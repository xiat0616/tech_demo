import numpy as np

import sys

sys.path.append("..")
from causal_models.pgm import unet
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from pyro.nn import DenseNN
from pyro.infer.reparam.transform import TransformReparam

from pgm.layers import (
    CNN,  # fmt: skip
    ConditionalGumbelMax,
    ConditionalTransformedDistributionGumbelMax,
)
from pgm.unet import ResUnet

def get_min_max_valumes(volume_name=None, target_size=(256,64)):
    _min, _max  = None, None
    if target_size==(256,64):
        if "Left-Lung" in volume_name:
            _min, _max = 303, 5931
        elif "Right-Lung" in volume_name:
            _min, _max  = 302, 5948
        elif "Heart" in volume_name:
            _min, _max  = 300, 3756
        else: 
            print(f"wrong volume name: {volume_name}")
    elif target_size==(224,224):
        if "Left-Lung" in volume_name:
            _min, _max = 934, 18153
        elif "Right-Lung" in volume_name:
            _min, _max  = 912, 18218
        elif "Heart" in volume_name:
            _min, _max  = 897, 11522
        else: 
            print(f"wrong volume name: {volume_name}")
    return  _min, _max 

def get_norm_volumes(
        seg,
        volume_name,
):
    _min, _max = get_min_max_valumes(
        volume_name = volume_name,
        target_size=seg.size()[-2:],
    )

    sum_seg = torch.sum(seg, dim=(1,2,3), keepdim=False) # dim is defined for (B, C, H, W) segmentation

    norm_seg = (sum_seg-_min) / _max
    norm_seg = norm_seg[:, None] # Add one dim
    return norm_seg


class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)  # model defined in parent class
        return samples

    def infer_exogeneous(self, obs):
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()
        # assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)

            if hasattr(self, "discrete_variables"):  # hack for MIMIC
                # Check if we should change "finding", i.e. if its parents and/or
                # itself are not intervened, then we use its observed value.
                # This is needed due to stochastic abduction of discrete variables
                if (
                    "age" not in intervention.keys()
                    and "finding" not in intervention.keys()
                ):
                    counterfactuals["finding"] = obs["finding"]

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class FlowPGM_with_seg(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "Left-Lung_volume": "continuous",
            "Right-Lung_volume": "continuous",
            "Heart_volume": "continuous",
        }
        # define base distributions
        for k in ["llv","rlv","hv"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))
        # LLV, RLV, HV spline flow
        self.llv_flow_components = T.ComposeTransformModule([T.Spline(1)])
        self.rlv_flow_components = T.ComposeTransformModule([T.Spline(1)])
        self.hv_flow_components = T.ComposeTransformModule([T.Spline(1)])

        self.llv_flow = T.ComposeTransform([self.llv_flow_components,])
        self.rlv_flow = T.ComposeTransform([self.rlv_flow_components,])
        self.hv_flow = T.ComposeTransform([self.hv_flow_components,])

        if args.setup == "sup_seg":
            ### Do segmentations ###
            # segmentor for llv
            self.encoder_llv = ResUnet(channel=1)
            # segmentor for rlv
            self.encoder_rlv = ResUnet(channel=1)
            # segmentor for hv
            self.encoder_hv = ResUnet(channel=1)
            
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )
        else:
            NotImplementedError

    def model(self, t=None):
        pyro.module("ChestPGM", self)
        # p(llv), llv flow
        pllv_base = dist.Normal(self.llv_base_loc, self.llv_base_scale).to_event(1)
        pllv = dist.TransformedDistribution(pllv_base, self.llv_flow)
        llv = pyro.sample("Left-Lung_volume", pllv)
        _ = self.llv_flow_components  # register with pyro

        # p(rlv), llv flow
        prlv_base = dist.Normal(self.rlv_base_loc, self.rlv_base_scale).to_event(1)
        prlv = dist.TransformedDistribution(prlv_base, self.rlv_flow)
        rlv = pyro.sample("Right-Lung_volume", prlv)
        _ = self.rlv_flow_components  # register with pyro

        # p(hv), hv flow
        phv_base = dist.Normal(self.hv_base_loc, self.hv_base_scale).to_event(1)
        phv = dist.TransformedDistribution(phv_base, self.hv_flow)
        hv = pyro.sample("Heart_volume", phv)
        _ = self.hv_flow_components  # register with pyro
    
        return {
            "Left-Lung_volume": llv,
            "Right-Lung_volume": rlv,
            "Heart_volume": hv,
        }

    def guide(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            if obs["Left-Lung_volume"] is None:
                llv = get_norm_volumes(seg=self.encoder_llv(obs["x"]),volume_name="Left-Lung_volume")
                pyro.sample("llv_aux", llv)
            if obs["Right-Lung_volume"] is None:
                rlv = get_norm_volumes(seg=self.encoder_rlv(obs["x"]),volume_name="Right-Lung_volume")
                pyro.sample("rlv_aux", rlv)
            if obs["Heart_volume"] is None:
                hv = get_norm_volumes(seg=self.encoder_hv(obs["x"]),volume_name="Heart_volume")
                pyro.sample("hv_aux", hv)

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(llv | x)
            llv = get_norm_volumes(seg=self.encoder_llv(obs["x"]),volume_name="Left-Lung_volume")
            pyro.sample("Left-Lung_volume_aux", llv, obs=obs["Left-Lung_volume"])
            
            # q(rlv | x)
            rlv = get_norm_volumes(seg=self.encoder_rlv(obs["x"]),volume_name="Right-Lung_volume")
            pyro.sample("Right-Lung_volume_aux", rlv, obs=obs["Right-Lung_volume"])

            # q(hv | x)
            hv = get_norm_volumes(seg=self.encoder_hv(obs["x"]),volume_name="Heart_volume")
            pyro.sample("Heart_volume_aux", hv, obs=obs["Heart_volume"])

    def predict_volumes(self, **obs):
        # q(llv | x)
        llv = get_norm_volumes(seg=self.encoder_llv(obs["x"]),volume_name="Left-Lung_volume")
        # q(rlv | x)
        rlv = get_norm_volumes(seg=self.encoder_rlv(obs["x"]),volume_name="Right-Lung_volume")
        # q(hv | x)
        hv = get_norm_volumes(seg=self.encoder_hv(obs["x"]),volume_name="Heart_volume")
        return {
            "Left-Lung_volume": llv,
            "Right-Lung_volume": rlv,
            "Heart_volume": hv,
        }
    
    def predict_segmentations(self, **obs):
        llv_seg = self.encoder_llv(obs["x"])
        rlv_seg = self.encoder_rlv(obs["x"])
        hv_seg = self.encoder_hv(obs["x"])

        return {
            "Left-Lung": llv_seg,
            "Right-Lung": rlv_seg,
            "Heart": hv_seg,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass

