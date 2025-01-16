import sys
from venv import logger
import torch
import torch.nn as nn
sys.path.append('../..')
sys.path.append('..')
from causal_models.train_setup import setup_directories, setup_tensorboard, setup_logging
# From datasets import get_attr_max_min
# from hvae2 import HVAE2
# from hvae4_attn import HVAE4_attn
import torch.nn.functional as F
from pgm.utils_pgm import check_nan, update_stats, calculate_loss

def vae_preprocess(args, pa, device="cuda:0"):
    pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    pa = pa[..., None, None].repeat(
        1, 1, args.input_res[0], args.input_res[1]).to(device).float()
    return pa

class DSCM(nn.Module):
    def __init__(self, args, pgm, predictor, vae):
        super().__init__()
        self.args = args
        self.pgm = pgm
        self.pgm.eval()
        self.pgm.requires_grad = False
        self.predictor = predictor
        self.predictor.eval()
        self.predictor.requires_grad = False
        self.vae = vae
        # lagrange multiplier
        self.lmbda = nn.Parameter(args.lmbda_init * torch.ones(1))  
        self.register_buffer('eps', args.elbo_constraint * torch.ones(1))

    def forward(self, obs, do, elbo_fn, cf_particles=1):
        # print(f"obs : {obs}")
        # for k, v in obs.items():
        #     print(f"k: {k}, v: {v.size()}")
        pa = {k: v for k, v in obs.items() if k != 'x' and k!='pa'}
        # for k, v in pa.items():
        #     print(f"k: {k}, v: {v.size()}")
        # forward vae with factual parents

        _pa = vae_preprocess(
            self.args, {k: v.clone() for k, v in pa.items()})
        # print("vae_out = self.vae(obs['x'], _pa, beta=self.args.beta)")
        vae_out = self.vae(obs['x'], _pa, beta=self.args.beta)

        # Get soft labels, should be normalized values
        with torch.no_grad():
            soft_labels = self.predictor.predict(**obs)

        if cf_particles > 1:  # for calculating counterfactual variance
            cfs = {'x': torch.zeros_like(obs['x'])}
            cfs.update({'x2': torch.zeros_like(obs['x'])})

        for _ in range(cf_particles):
            # forward pgm, get counterfactual parents
            # logger.info(f"pa keys: {pa.keys()}; do keys: {do.keys()}")
            cf_pa = self.pgm.counterfactual(
                obs=pa, intervention=do, num_particles=1)
            _cf_pa = vae_preprocess(
                self.args, {k: v.clone() for k, v in cf_pa.items()})
            # print("zs = self.vae.abduct(obs['x'], parents=_pa)")
            # forward vae with counterfactual parents
            zs = self.vae.abduct(obs['x'], parents=_pa)  # z ~ q(z|x,pa)

            # To get z
            if self.vae.cond_prior:
                zs = [zs[j]["z"] for j in range(len(zs))]
            # for z in zs:
            #     print(f"z: {type(z)}")
            # print(f"zs: {type(zs)}, cond_prior: {self.vae.cond_prior}, _cf_pa: {_cf_pa.size()}")
            cf_loc, cf_scale = self.vae.forward_latents(zs, parents=_cf_pa, t=0.1)
            rec_loc, rec_scale = self.vae.forward_latents(zs, parents=_pa)
            # cf_x = obs['x'] + (cf_loc - rec_loc)
            u = (obs['x'] - rec_loc) / rec_scale.clamp(min=1e-12)
            cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)
            
            if cf_particles > 1:
                cfs['x'] += cf_x   
                with torch.no_grad():
                    cfs['x2'] += cf_x**2
            else:
                cfs = {'x': cf_x}
        
        # Var[X] = E[X^2] - E[X]^2
        if cf_particles > 1:
            with torch.no_grad():
                var_cf_x = (cfs['x2'] - cfs['x']**2 / cf_particles) / cf_particles
                cfs.pop('x2', None)
            cfs['x'] = cfs['x'] / cf_particles
        else:
            var_cf_x = None
            
        cfs.update(cf_pa)
        if check_nan(vae_out) > 0 or check_nan(cfs) > 0:
            return {'loss': torch.tensor(float('nan'))}
        
        # For age, we do not use predicted values, as it is already soft labels
        soft_labels['age'] = cfs['age']

        for do_key in do.keys():
            # When do(age), it causally affect disease, 
            # so we do not use the soft label of disease as it might be already changed. 
            # The same for age, but we do not use soft label for age in any case.
            if do_key in ['age']:
                soft_labels['finding'] = cfs['finding']
            # Set soft_labels[do_key] as the intervented label
            soft_labels[do_key] = cfs[do_key]

        # Use other sup loss, here target batch should be the soft labels
        pred_batch = self.predictor.predict_unnorm(**cfs)
        aux_loss = calculate_loss(pred_batch=pred_batch, 
                                target_batch=soft_labels, 
                                loss_norm="l2")

        with torch.no_grad():
            sg = self.eps - vae_out['elbo']
        damp = self.args.damping * sg
        loss = aux_loss - (self.lmbda - damp) * (self.eps - vae_out['elbo'])
        # loss =vae_out['elbo'] + aux_loss * args.alpha
        
        out = {}
        out.update(vae_out)
        out.update({
            'loss': loss, 
            'aux_loss': aux_loss,
            'cfs': cfs,
            'var_cf_x': var_cf_x,
            'cf_pa': cf_pa,
        })
        del loss, aux_loss, vae_out
        return out