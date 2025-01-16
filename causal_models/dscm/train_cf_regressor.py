import sys
import argparse
import random
import copy
import pyro
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
sys.path.append('..')
sys.path.append("../..")
from train_setup import setup_directories, setup_tensorboard, setup_logging
# From datasets import get_attr_max_min
from utils import EMA, seed_all
from train_setup import setup_dataloaders

from hvae2 import HVAE2
# from hvae4_attn import HVAE4_attn

from causal_models.dscm.dscm_regressor import DSCM, vae_preprocess

import torch.nn.functional as F
from pgm.train_pgm import sup_epoch, eval_epoch
from pgm.utils_pgm import check_nan, update_stats, calculate_loss, plot_cf

from pgm.layers import TraceStorage_ELBO
from causal_models.pgm.chest_pgm_regressor import FlowPGM
# from flow_pgm import FlowPGM_full as FlowPGM

def norm(batch):
    for k, v in batch.items():
        # if k == 'x':
        #     batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        if k in ['age']:
            batch[k] = batch[k].float()# Age normalised to [0,1]
            # batch[k] = batch[k] / 100.
            # batch[k] = batch[k] *2 -1 #[-1,1]
        elif k in ['race']:
            batch[k] = batch[k].float()
        elif k in ['finding']:
            batch[k] = batch[k].float()
        else:
            batch[k] = batch[k].float()
    return batch

def loginfo(title, logger, stats):
    logger.info(f'{title} | ' +
                ' - '.join(f'{k}: {v:.4f}' for k, v in stats.items()))

def preprocess(batch, device="cuda:0"):
    remove_k_list=[]
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = batch['x'].float().to(device)*2-1 # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().to(device) # age normlised to [0,1] already
        elif k in ['race']:
            batch[k] =batch[k].float().to(device)
#             print("batch[race]: ", batch[k])
        elif k in ['finding']:
            batch[k] = batch[k].float().to(device)
            # batch[k] = F.one_hot(batch[k], num_classes=2).squeeze().float().cuda()
        else:
            try:
                batch[k] = batch[k].float().to(device)
            except:
                remove_k_list.append(k)
    for k in remove_k_list:
        del batch[k]
    return batch

def inv_preprocess(pa):
    # Undo [0,1] parent preprocessing back to original range
    for k, v in pa.items():
        if k =='age':
            pa[k] = v * 100
    return pa


def save_plot(save_path, obs, cfs, do, var_cf_x=None, num_images=10):
    _ = plot_cf(obs['x'], cfs['x'], 
        inv_preprocess({k: v for k, v in obs.items() if k != 'x'}),  # pa
        inv_preprocess({k: v for k, v in cfs.items() if k != 'x'}),  # cf_pa
        inv_preprocess(do), # Counterfactual variance per pixel
        # cf_x_orig = cf_x_orig['x'],
        var_cf_x = var_cf_x,
        num_images=num_images,
        logger=logger,
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_metrics(preds, targets):
    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze().cpu()
    stats = {}
    for k in preds.keys():
        if k=="age":
            preds_k = (preds[k] + 1) / 2 *100  # [-1,1] -> [0,100]
            stats[k+'_mae'] = torch.mean(
                torch.abs(targets[k] - preds_k)).item() 
    return stats

@torch.no_grad()
def vae_epoch(args, vae, dataloader):
    vae.eval()
    stats = {k: 0 for k in ['elbo', 'nll', 'kl', 'n']}
    loader = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, batch in loader:
        # preprocessing
        batch = preprocess(batch) # [-1, 1]
        pa = {k: v for k, v in batch.items() if k != 'x' and k!='pa'}
        _pa = vae_preprocess(
            args, {k: v.clone() for k, v in pa.items()})
        # forward pass
        out = vae(batch['x'], _pa, beta=args.beta)
        # update stats
        bs = batch['x'].shape[0]
        stats['n'] += bs  # samples seen counter
        stats['elbo'] += out['elbo'] * bs
        stats['nll'] += out['nll'] * bs
        stats['kl'] += out['kl'] * bs
        loader.set_description(
            f' => eval | elbo: {stats["elbo"] / stats["n"]:.4f}'
            + f' - nll: {stats["nll"] / stats["n"]:.4f}'
            + f' - kl: {stats["kl"] / stats["n"]:.4f}'
        )
    return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}

def cf_epoch(args, model, ema, dataloaders, elbo_fn, optimizers, split='train', is_train=None):
    'counterfactual auxiliary training/eval epoch'
    if is_train==None:
        is_train = split == 'train'
    model.vae.train(is_train)
    model.pgm.eval()
    model.predictor.eval()
    stats = {k: 0 for k in ['loss', 'aux_loss', 'elbo', 'nll', 'kl', 'n']}
    steps_skipped = 0

    dag_vars = list(model.pgm.variables.keys())
    if is_train:
        optimizer, lagrange_opt = optimizers
    else:
        preds = {k: [] for k in dag_vars}
        targets = {k: [] for k in dag_vars}
        # train_samples = copy.deepcopy(dataloaders['train'].dataset.get_samples())

        # for k in dag_vars:
        #     if k!="x":
        #         train_samples[k]=torch.from_numpy(np.array(train_samples[k]))
        # n_train = len(dataloaders['train'].dataset)

    loader = tqdm(enumerate(dataloaders[split]), total=len(
        dataloaders[split]), mininterval=0.1)

    for i, batch in loader:
        bs = batch['x'].shape[0]
        # logger.info(f"bs: {bs}")
        # logger.info(f"before preprocess batch['x'].min(): {batch['x'].min()}, max: {batch['x'].max()}")
        batch = preprocess(batch)
        # logger.info(f"after preprocess batch['x'].min(): {batch['x'].min()}, max: {batch['x'].max()}")
        with torch.no_grad():
            # Randomly intervene on a single parent do(pa_k ~ p(pa_k))
            do = {}
            do_k = copy.deepcopy(args.do_pa) if args.do_pa else random.choice(dag_vars)
            if is_train:
                # Option 1: randomly shuffule the batch.
                # do[do_k] = batch[do_k].clone()[torch.randperm(bs)] 
                
                # Option 2: randomly sample from [0,1]
                do[do_k] = torch.rand(size=batch[do_k].size()).to("cuda:0")
            else:
                # do[do_k] = train_samples[do_k].clone()[torch.randperm(n_train)][:bs]
                do[do_k] = batch[do_k].clone()[torch.randperm(bs)]
                # do = preprocess(norm(do))

        with torch.set_grad_enabled(is_train):
            if not is_train:        
                args.cf_particles = 1

            out = model.forward(batch, do, elbo_fn, cf_particles=args.cf_particles)

            if torch.isnan(out['loss']):
                model.zero_grad(set_to_none=True)
                steps_skipped += 1
                continue
                
        if is_train:
            args.step = i + (args.epoch-1) * len(dataloaders[split])
            optimizer.zero_grad(set_to_none=True)
            lagrange_opt.zero_grad(set_to_none=True)
            out['loss'].backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)

            if grad_norm < args.grad_skip:
                optimizer.step()
                lagrange_opt.step()  # gradient ascent on lmbda
                model.lmbda.data.clamp_(min=0)
                ema.update()
            else:
                steps_skipped += 1
                print(
                    f'Steps skipped: {steps_skipped} - grad_norm: {grad_norm:.3f}')
            
            if args.step % 500==0:
                ckpt_path = os.path.join(args.save_dir, f'{args.step}_checkpoint.pt')
                torch.save({'epoch': args.epoch,
                            'step': args.step,
                            'best_loss': args.best_loss,
                            'model_state_dict': model.state_dict(),
                            'ema_model_state_dict': ema.ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lagrange_opt_state_dict': lagrange_opt.state_dict(),
                            'hparams': vars(args)}, ckpt_path)
                logger.info(f'Model saved: {ckpt_path}')
        else:  # Evaluation
            with torch.no_grad():
                preds_cf = ema.ema_model.predictor.predict(**out['cfs'])
                for k, v in preds_cf.items():
                    preds[k].extend(v)
                # Interventions are the targets for prediction
                for k in targets.keys():
                    t_k = do[k].clone() if k in do.keys() else out['cfs'][k].clone()
                    targets[k].extend(inv_preprocess({k: t_k})[k])

        if i % args.plot_freq == 0:
            if is_train:
                copy_do_pa = copy.deepcopy(args.do_pa)
                for pa_k in dag_vars + [None]:
                    args.do_pa = pa_k
                args.do_pa = copy_do_pa
            save_path = os.path.join(args.save_dir, f'{args.step}_{split}_{do_k}_cfs.png')
            save_plot(save_path, batch, out['cfs'], do,  
                      var_cf_x = out['var_cf_x'],
                      num_images=args.imgs_plot)

        stats['n'] += bs
        stats['loss'] += out['loss'].item() * bs
        stats['aux_loss'] += out['aux_loss'].item() * args.alpha * bs
        stats['elbo'] += out['elbo'] * bs
        stats['nll'] += out['nll'] * bs
        stats['kl'] += out['kl'] * bs
        del out
        
        try:
            stats = update_stats(stats, elbo_fn)
        except:
            pass

        loader.set_description(
            f'[{split}] lmbda: {model.lmbda.data.item():.3f}, ' +
            f', '.join(f'{k}: {v / stats["n"]:.3f}' for k, v in stats.items() if k != 'n') +
            (f', grad_norm: {grad_norm:.3f}' if is_train else '')
        )
    stats = {k: v / stats['n'] for k, v in stats.items() if k != 'n'}
    return stats if is_train else (stats, get_metrics(preds, targets))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',
                        help='experiment name.', type=str, default='')
    # parser.add_argument('--data_dir',
                        # help='data directory to load form.', type=str, default='')
    # parser.add_argument('--csv_dir',
    #                     help='CSV directory to load form.', type=str, default='')
    # parser.add_argument('--use_dataset',
    #                     help='Which dataset to use', type=str, default='')
    parser.add_argument('--load_path',
                        help='Path to load checkpoint.', type=str, default='')
    parser.add_argument('--pgm_path',
                        help='path to load pgm checkpoint.', type=str,
                        default='')
    parser.add_argument('--predictor_path',
                        help='path to load predictor checkpoint.', type=str,
                        default='')
    parser.add_argument('--vae_path',
                        help='path to load vae checkpoint.', type=str,
                        default='')
    parser.add_argument('--seed',
                        help='random seed.', type=int, default=7)
    parser.add_argument('--deterministic',
                        help='toggle cudNN determinism.', action='store_true', default=False)
    parser.add_argument('--testing',
                        help='test model.', action='store_true', default=False)
    # training
    parser.add_argument('--epochs',
                        help='num training epochs.', type=int, default=5000)
    parser.add_argument('--bs',
                        help='batch size.', type=int, default=32)
    parser.add_argument('--lr',
                        help='learning rate.', type=float, default=1e-4)
    parser.add_argument('--lr_lagrange',
                        help='learning rate for multipler.', type=float, default=1e-2)
    parser.add_argument('--ema_rate',
                        help='Exp. moving avg. model rate.', type=float, default=0.999)
    parser.add_argument('--alpha',
                        help='aux loss multiplier.', type=float, default=1)
    parser.add_argument('--loss_norm',
                        help='Loss norm for age.', type=str, default='l1')
    parser.add_argument('--lmbda_init',
                        help='lagrange multiplier init.', type=float, default=0)
    parser.add_argument('--damping',
                        help='lagrange damping scalar.', type=float, default=100)
    parser.add_argument('--do_pa',
                        help='intervened parent.', type=str, default=None)
    parser.add_argument('--device',
                        help='Which device ')
    parser.add_argument('--eval_freq',
                        help='epochs per eval.', type=int, default=1)
    parser.add_argument('--plot_freq',
                        help='steps per plot.', type=int, default=500)
    parser.add_argument('--imgs_plot',
                        help='num images to plot.', type=int, default=4)
    parser.add_argument('--cf_particles',
                        help='num counterfactual samples.', type=int, default=1)
    args = parser.parse_known_args()[0]
    args.cf_particles = 1

    # Update hparams if loading checkpoint
    if args.load_path:
        if os.path.isfile(args.load_path):
            print(f'\nLoading checkpoint: {args.load_path}')
            ckpt = torch.load(args.load_path)
            ckpt_args = {
                k: v for k, v in ckpt['hparams'].items() if k != 'load_path'}
            if args.testing:
                ckpt_args['testing'] = args.testing
            vars(args).update(ckpt_args)
        else:
            print(f'Checkpoint not found at: {args.load_path}')
    
    seed_all(args.seed, args.deterministic)


    class Hparams:
        def update(self, dict):
            for k, v in dict.items():
                setattr(self, k, v)

    # Load predictors
    print(f'\nLoading predictor checkpoint: {args.predictor_path}')
    predictor_checkpoint = torch.load(args.predictor_path)
    predictor_args = Hparams()
    predictor_args.update(predictor_checkpoint['hparams'])

    predictor_args.loss_norm = "l2"

    predictor = FlowPGM(predictor_args).cuda()
    predictor.load_state_dict(predictor_checkpoint['ema_model_state_dict'])
    args.loss_norm = predictor_args.loss_norm

    # Load PGM
    print(f'\nLoading PGM checkpoint: {args.pgm_path}')
    pgm_checkpoint = torch.load(args.pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint['hparams'])
    pgm = FlowPGM(pgm_args).cuda()
    pgm.load_state_dict(pgm_checkpoint['ema_model_state_dict'])

    # Load deep VAE
    print(f'\nLoading VAE checkpoint: {args.vae_path}')
    vae_checkpoint = torch.load(args.vae_path)
    vae_args = Hparams()
    vae_args.batch_size = args.bs

    vae_args.update(vae_checkpoint['hparams'])
    vae = HVAE2(vae_args).cuda()
    vae.load_state_dict(vae_checkpoint['ema_model_state_dict'])
    # vae_args.data_dir = None

    # setup data
    dataloaders = setup_dataloaders(vae_args, cache=False, shuffle_train=True)

    # Get the elbo constraint
    stats = vae_epoch(vae_args, vae, dataloaders['train'])
    print(f"elbo: {stats['elbo']}")  
    vae.train()  # Set it back for training


    # setup experiment args
    args.beta = vae_args.beta
    args.parents_x = vae_args.parents_x
    args.input_res = vae_args.input_res
    args.grad_clip = vae_args.grad_clip
    args.grad_skip = vae_args.grad_skip
    args.elbo_constraint = stats['elbo'].cpu()  # train set elbo 
    args.wd = vae_args.wd
    args.betas = vae_args.betas

    # init model
    model = DSCM(args, pgm, predictor, vae)
    # model_original = DSCM(args, copy.deepcopy(pgm), copy.deepcopy(predictor), copy.deepcopy(vae))
    # model_original.cuda()
    ema = EMA(model, beta=args.ema_rate)
    model.cuda()
    ema.cuda()

    # Freeze the parameters of pgm and predictor 
    for p in model.predictor.parameters():
        p.requires_grad = False
    for p in model.pgm.parameters():
        p.requires_grad = False

    # Train model
    if not args.testing:
        args.save_dir = setup_directories(args, ckpt_dir='checkpoints')
        writer = setup_tensorboard(args, model)
        logger = setup_logging(args)
        writer.add_custom_scalars({
            "loss": {"loss": ["Multiline", ["loss/train", "loss/valid"]]},
            "aux_loss": {"aux_loss": ["Multiline", ["aux_loss/train", "aux_loss/valid"]]}
        })

        # setup loss & optimizer
        elbo_fn = TraceStorage_ELBO(num_particles=1)
        optimizer = torch.optim.AdamW(
            # [p for n, p in model.named_parameters() if n != 'lmbda'],
            model.vae.parameters(), # Only update the VAE
            lr=args.lr, weight_decay=args.wd, betas=args.betas
        )
        lagrange_opt = torch.optim.AdamW(
            [model.lmbda], lr=args.lr_lagrange, betas=args.betas, 
            weight_decay=0, maximize=True
        )
        optimizers = (optimizer, lagrange_opt)
        
        # load checkpoint
        if args.load_path:
            if os.path.isfile(args.load_path):
                args.start_epoch = ckpt['epoch']
                args.step = ckpt['step']
                args.best_loss = ckpt['best_loss']
                model.load_state_dict(ckpt['model_state_dict'])
                ema.ema_model.load_state_dict(ckpt['ema_model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                lagrange_opt.load_state_dict(ckpt['lagrange_opt_state_dict'])
            else:
                print('Checkpoint not found: {}'.format(args.load_path))
        else:
            args.start_epoch, args.step = 0, 0
            args.best_loss = float('inf')

        for k in sorted(vars(args)):
            logger.info(f'--{k}={vars(args)[k]}')

        # training loop
        for epoch in range(args.start_epoch, args.epochs):
            args.epoch = epoch + 1
            logger.info(f'Epoch: {args.epoch}')
            stats = cf_epoch(
                args, model, ema, dataloaders, elbo_fn, optimizers,
                split='train',
            )
            loginfo('train', logger, stats)

            if epoch % args.eval_freq == 0:
                # evaluate single parent interventions
                copy_do_pa = copy.deepcopy(args.do_pa)
                for pa_k in list(model.pgm.variables.keys()) + [None]:
                    args.do_pa = pa_k
                    valid_stats, metrics = cf_epoch(
                        args, model, ema, dataloaders, elbo_fn, None,
                        split='valid',
                    )
                    loginfo(f'valid do({pa_k})', logger, valid_stats)
                    loginfo(f'valid do({pa_k})', logger, metrics)
                args.do_pa = copy_do_pa

                for k, v in stats.items():
                    writer.add_scalar('train/'+k, v, args.step)
                    writer.add_scalar('valid/'+k, valid_stats[k], args.step)
                
                for k, v in metrics.items():    
                    writer.add_scalar('valid/'+k, v, args.step)
                
                writer.add_scalar('loss/train', stats['loss'], args.step)
                writer.add_scalar('loss/valid', valid_stats['loss'], args.step)
                writer.add_scalar('aux_loss/train', stats['aux_loss'], args.step)
                writer.add_scalar('aux_loss/valid', valid_stats['aux_loss'], args.step)
            
                if valid_stats['loss'] < args.best_loss:
                    args.best_loss = valid_stats['loss'] 
                    ckpt_path = os.path.join(args.save_dir, f'checkpoint.pt')
                    torch.save({'epoch': args.epoch,
                                'step': args.step,
                                'best_loss': args.best_loss,
                                'model_state_dict': model.state_dict(),
                                'ema_model_state_dict': ema.ema_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lagrange_opt_state_dict': lagrange_opt.state_dict(),
                                'hparams': vars(args)}, ckpt_path)
                    logger.info(f'Model saved: {ckpt_path}')
                ckpt_path = os.path.join(args.save_dir, f'current_checkpoint.pt')
                torch.save({'epoch': args.epoch,
                            'step': args.step,
                            'best_loss': args.best_loss,
                            'model_state_dict': model.state_dict(),
                            'ema_model_state_dict': ema.ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lagrange_opt_state_dict': lagrange_opt.state_dict(),
                            'hparams': vars(args)}, ckpt_path)
                logger.info(f'Model saved: {ckpt_path}')
    else:
        # test model
        model.load_state_dict(ckpt['model_state_dict'])
        ema.ema_model.load_state_dict(ckpt['ema_model_state_dict'])
        elbo_fn = TraceStorage_ELBO(num_particles=1)
        stats, metrics = cf_epoch(
            args, model, ema, dataloaders, elbo_fn, None, 
            split='test',
        )
        print(f'\n[test] '+' - '.join(f'{k}: {v:.4f}' for k, v in stats.items()))
        print(f'[test] '+' - '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))