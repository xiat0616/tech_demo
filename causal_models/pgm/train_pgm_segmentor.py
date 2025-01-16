import sys
sys.path.append('..')
sys.path.append('../..')

from train_setup import *
# from datasets import ukbb, get_attr_max_min
from utils import seed_all, seed_worker, EMA
import os
import copy
import argparse
import pyro
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from pyro.infer import config_enumerate
from sklearn.metrics import roc_auc_score
from causal_models.pgm.chest_pgm_segmentor import FlowPGM_with_seg 
from causal_models.pgm.utils_pgm import update_stats, segmentation_loss, dice_coef, BCEloss
from causal_models.pgm.layers import TraceStorage_ELBO
from torch.optim.lr_scheduler import ReduceLROnPlateau

def preprocess(batch, device="cuda:0"):
    batch_new = {}
    for k, v in batch.items():
        if k == 'x':
            batch_new['x'] = batch['x'].float().to(device) * 2 -1  # [-1,1]
        elif k in ['age']:
            batch_new[k] = batch[k].float().to(device) 
        elif k in ['race']:
            batch_new[k] =batch[k].float().to(device) 
        elif k in ['finding']:
            batch_new[k] = batch[k].float().to(device) 
        else:
            try:
                batch_new[k] = batch[k].float().to(device) 
            except:
                continue
    return batch_new

def sup_epoch(model, ema, dataloader, elbo_fn, optimizer=None, setup=None, is_train=True, loss_norm="l1"):
    "supervised epoch"
    stats = {'loss': 0, 'n': 0}  # sample counter

    mininterval = 0.1
    loader = tqdm(
            enumerate(dataloader), total=len(dataloader), mininterval=mininterval
        )
    
    model.train(is_train)
    for i, batch in loader:
        bs = batch['x'].shape[0]
        batch = preprocess(batch)
        
        # logger.info(f"batch: {batch}")

        with torch.set_grad_enabled(is_train):
            if setup == 'sup_seg':
                pred_batch = model.predict_segmentations(**batch)
                loss = segmentation_loss(pred_batch=pred_batch,
                                         target_batch=batch)
            elif setup == 'sup_pgm':
                loss = elbo_fn.differentiable_loss(model.svi_model,
                               model.guide_pass, **batch) / bs
            else:
                NotImplementedError

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            ema.update()
        
        stats['loss'] += loss.item() * bs
        stats['n'] += bs
        try:
            stats = update_stats(stats, elbo_fn)
        except:
            pass
        loader.set_description(
        f' => {("train" if is_train else "eval")} | '+
        f', '.join(f'{k}: {v / stats["n"]:.4f}' for k, v in stats.items() if k != 'n')+
        (f', grad_norm: {grad_norm:.3f}' if is_train else "")
        )       
    return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}


from sklearn.metrics import roc_auc_score, accuracy_score
@torch.no_grad()
def eval_epoch(model, dataloader):
    ' this can consume lots of memory if dataset is large'
    _KEYS = ["Left-Lung", "Right-Lung", "Heart"]
    _VOL_KEYS = ["Left-Lung_volume", "Right-Lung_volume", "Heart_volume"]
    model.eval()
    preds = {k: [] for k in  _KEYS}
    preds_vol = {k: [] for k in _VOL_KEYS}
    targets = {k: [] for k in _KEYS}
    targets_vol =  {k: [] for k in _VOL_KEYS}

    for batch in tqdm(dataloader):
        for k in targets.keys():
            targets[k].extend(copy.deepcopy(batch[k]))
        for k in targets_vol.keys():
            targets_vol[k].extend(copy.deepcopy(batch[k]))
        # predict
        batch = preprocess(batch)
        out = model.predict_segmentations(**batch)
        vol_out = model.predict_volumes(**batch)

        for k, v in out.items():
            preds[k].extend(v)
        for k, v in vol_out.items():
            preds_vol[k].extend(v)

    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k])
    for k, v in preds_vol.items():
        preds_vol[k] = torch.stack(v).squeeze().cpu()
        targets_vol[k] = torch.stack(targets_vol[k])
    stats = {}

    for i, k in enumerate(_VOL_KEYS):
        if k in ['Left-Lung_volume','Right-Lung_volume', 'Heart_volume']:
            preds_k = preds_vol[k].unsqueeze(-1)
            stats[k] = torch.mean(torch.abs(targets_vol[k] - preds_k)).item() 

    for i, k in enumerate(_KEYS):
        if k in ['Left-Lung', 'Right-Lung', 'Heart']:
            stats[k+'_dice_coef'] = dice_coef(input=preds[k], target=targets[k])
            stats[k+'_BCE_loss'] = BCEloss(input=preds[k], target=targets[k])
            
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',
                        help='Experiment name.', type=str, default='')
    parser.add_argument('--hps',
                        help='Which dataset to use.', type=str, default='')     
    parser.add_argument('--batch_size',
                        help="Batch size", type=int, default=32)            
    parser.add_argument('--load_path',
                        help='Path to load checkpoint.', type=str, default='')
    parser.add_argument('--setup',  # sup_pgm/sup_aux/sup_determ
                        help='training setup.', type=str, default='sup_determ')
    parser.add_argument('--seed',
                        help='Set random seed.', type=int, default=7)
    parser.add_argument('--deterministic',
                        help='Toggle cudNN determinism.', action='store_true', default=False)
    parser.add_argument('--testing',
                        help='Test model.', action='store_true', default=False)
    parser.add_argument('--loss_norm',
                        help='Loss norm for age.', type=str, default='l1')
    # training
    parser.add_argument('--epochs',
                        help='Number of training epochs.', type=int, default=5000)
    parser.add_argument('--lr',
                        help='Learning rate.', type=float, default=1e-4)
    parser.add_argument('--lr_warmup_steps',
                        help='lr warmup steps.', type=int, default=1)
    parser.add_argument('--wd',
                        help='Weight decay penalty.', type=float, default=0.1)
    parser.add_argument('--input_res',
                        help='Input image crop resolution.', type=int, default=512)
    parser.add_argument('--input_channels',
                        help='Input image num channels.', type=int, default=1)
    parser.add_argument('--pad',
                        help='Input padding.', type=int, default=9)
    # parser.add_argument('--hflip',
    #                     help='Horizontal flip prob.', type=float, default=0.5)
    parser.add_argument('--eval_freq',
                        help='Num epochs per eval.', type=int, default=1)
    # model
    parser.add_argument('--enc_net',
                        help='encoder network architecture.', type=str, default='cnn')
    parser.add_argument('--widths',
                        help='Cond flow fc network width per layer.', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--parents_x',
                        help='Parents of x to load.', nargs='+', default=["age", "race", "sex", "finding" ])
    parser.add_argument('--alpha',
                        help='aux loss multiplier.', type=float, default=1e-3)
    parser.add_argument('--std_fixed',
                        help='Fix aux dist std value (0 is off).', type=float, default=0)
    args = parser.parse_known_args()[0]
    # args = parser.parse_args()

    seed_all(args.seed, args.deterministic)

    # update hparams if loading checkpoint
    if args.load_path:
        if os.path.isfile(args.load_path):
            print(f'\nLoading checkpoint: {args.load_path}')
            ckpt = torch.load(args.load_path)
            ckpt_args = {
                k: v for k, v in ckpt['hparams'].items() if k != 'load_path'}
            if args.data_dir is not None:
                ckpt_args['data_dir'] = args.data_dir
            if args.testing:
                ckpt_args['testing'] = args.testing
            vars(args).update(ckpt_args)
        else:
            print(f'Checkpoint not found at: {args.load_path}')

    # Load data
    dataloaders = setup_dataloaders(args, cache=False, shuffle_train=False)

    # Init model
    pyro.clear_param_store()
        
    model = FlowPGM_with_seg(args)
    ema = EMA(model, beta=0.999)
    model.cuda()
    ema.cuda()

    # Init loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=2)
    aux_elbo_fn = TraceStorage_ELBO(num_particles=2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler=ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=10,
                mode='min',
                min_lr=1e-5,)

    if not args.testing:
        # Train model
        args.save_dir = setup_directories(args, ckpt_dir='checkpoints')
        writer = setup_tensorboard(args, model)
        logger = setup_logging(args)

        for k in sorted(vars(args)):
            logger.info(f'--{k}={vars(args)[k]}')

        args.best_loss = float('inf')

        for epoch in range(args.epochs):
            logger.info(f'Epoch {epoch+1}:')
            # supervised training of PGM or aux models
            if args.setup in ['sup_pgm','sup_aux',"sup_determ",'sup_seg']:
                stats = sup_epoch(
                    model, ema,
                    dataloaders['train'],
                    elbo_fn,
                    optimizer,
                    setup=args.setup,
                    is_train=True,
                    loss_norm=args.loss_norm,
                )
                if epoch % args.eval_freq == 0:
                    valid_stats = sup_epoch(
                        ema.ema_model, None,
                        dataloaders['valid'],
                        elbo_fn,
                        setup=args.setup,
                        is_train=False,
                        loss_norm=args.loss_norm,
                    )
                    steps = (epoch + 1) * len(dataloaders['train'])
                    logger.info(f'loss | train: {stats["loss"]:.4f}' +
                                f' - valid: {valid_stats["loss"]:.4f} - steps: {steps}')
                    
                    for k, v in stats.items():
                        writer.add_scalar('train/'+k, v, steps)
                        writer.add_scalar('valid/'+k, valid_stats[k], steps)
                    
                    writer.add_custom_scalars({
                        "elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}
                    })
                    writer.add_scalar('elbo/train', stats['loss'], steps)
                    writer.add_scalar('elbo/valid', valid_stats['loss'], steps)

            else:
                NotImplementedError

            if epoch % args.eval_freq == 0:
                if not args.setup == 'sup_pgm':  # eval aux classifiers
                    metrics = eval_epoch(ema.ema_model, dataloaders['valid'])
                    logger.info(
                        'valid | '+' - '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))

            if valid_stats['loss'] < args.best_loss:
                args.best_loss = valid_stats['loss']
                ckpt_path = os.path.join(args.save_dir, 'checkpoint.pt')
                torch.save({'epoch': epoch + 1,
                            'step': steps,
                            'best_loss': args.best_loss,
                            'model_state_dict': model.state_dict(),
                            'ema_model_state_dict': ema.ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'hparams': vars(args)}, ckpt_path)
                logger.info(f'Model saved: {ckpt_path}')
            # Save current checkpoint
            ckpt_path = os.path.join(args.save_dir, 'checkpoint_current.pt')
            torch.save({'epoch': epoch + 1,
                            'step': steps,
                            'best_loss': args.best_loss,
                            'model_state_dict': model.state_dict(),
                            'ema_model_state_dict': ema.ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'hparams': vars(args)}, ckpt_path)
            logger.info(f'Model saved: {ckpt_path}')

    else:
        # test model
        model.load_state_dict(ckpt['model_state_dict'])
        ema.ema_model.load_state_dict(ckpt['ema_model_state_dict'])
        test_loss = sup_epoch(
            ema.ema_model, None,
            dataloaders['test'],
            elbo_fn,
            optimizer=None,
            setup=args.setup,
            is_train=False,
            loss_norm=args.loss_norm,
        )
        print(f'test | loss: {test_loss:.4f}')
        if not args.setup == 'sup_pgm':  # eval aux classifiers
            stats = eval_epoch(ema.ema_model, dataloaders['test'])
            print(
                'test | '+' - '.join(f'{k}: {v:.4f}' for k, v in stats.items()))