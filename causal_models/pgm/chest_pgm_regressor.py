import numpy as np

import sys

sys.path.append("..")
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
from pgm.resnet import  ResNet, ResNet18, CustomBlock


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

class FlowPGM(BasePGM):
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

        if args.setup != "sup_pgm":
            shared_model = ResNet(
                CustomBlock,
                layers=[2, 2, 2, 2],
                widths=[64, 128, 256, 512],
                norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),
            )

            shared_model.conv1 = nn.Conv2d(
                args.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            kwargs = {
                "in_shape": (args.input_channels, *(args.input_res,) * 2),
                "base_model": shared_model,
            }
            # q(llv | x) ~  Normal(mu(x), sigma(x))
            self.encoder_llv = ResNet18(num_outputs=2, **kwargs)
            # q(rlv | x) ~  Normal(mu(x), sigma(x))
            self.encoder_rlv = ResNet18(num_outputs=2, **kwargs)
            # q(hv | x) ~ Normal(mu(x), sigma(x))
            self.encoder_hv = ResNet18(num_outputs=2, **kwargs)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
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
                llv_loc, llv_logscale = self.encoder_llv(obs["x"]).chunk(
                    2, dim=-1
                )
                qllv_xf = dist.Normal(llv_loc, self.f(llv_logscale)).to_event(1)
                pyro.sample("llv_aux", qllv_xf)
            if obs["Right-Lung_volume"] is None:
                rlv_loc, rlv_logscale = self.encoder_rlv(obs["x"]).chunk(
                    2, dim=-1
                )
                qrlv_xf = dist.Normal(rlv_loc, self.f(rlv_logscale)).to_event(1)
                pyro.sample("rlv_aux", qrlv_xf)
            if obs["Heart_volume"] is None:
                hv_loc, hv_logscale = self.encoder_hv(obs["x"]).chunk(
                    2, dim=-1
                )
                qhv_xf = dist.Normal(hv_loc, self.f(hv_logscale)).to_event(1)
                pyro.sample("hv_aux", qhv_xf)

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(llv | x)
            llv_loc, llv_logscale = self.encoder_llv(obs["x"]).chunk(
                2, dim=-1
            )
            qllv_xf = dist.Normal(llv_loc, self.f(llv_logscale)).to_event(1)
            pyro.sample("Left-Lung_volume_aux", qllv_xf, obs=obs["Left-Lung_volume"])
            
            # q(rlv | x)
            rlv_loc, rlv_logscale = self.encoder_rlv(obs["x"]).chunk(
                2, dim=-1
            )
            qrlv_xf = dist.Normal(rlv_loc, self.f(rlv_logscale)).to_event(1)
            pyro.sample("Right-Lung_volume_aux", qrlv_xf, obs=obs["Right-Lung_volume"])

            # q(hv | x)
            hv_loc, hv_logscale = self.encoder_hv(obs["x"]).chunk(
                2, dim=-1
            )
            qhv_xf = dist.Normal(hv_loc, self.f(hv_logscale)).to_event(1)
            pyro.sample("Heart_volume_aux", qhv_xf, obs=obs["Heart_volume"])


    def predict(self, **obs):
        # q(llv | x, f)
        llv_loc, _ = self.encoder_llv(obs["x"]).chunk(2, dim=-1)
        # q(llv | x, f)
        rlv_loc, _ = self.encoder_rlv(obs["x"]).chunk(2, dim=-1)
        # q(llv | x, f)
        hv_loc, _ = self.encoder_hv(obs["x"]).chunk(2, dim=-1)
        return {
            "Left-Lung_volume": llv_loc,
            "Right-Lung_volume": rlv_loc,
            "Heart_volume": hv_loc,
        }

    def predict_unnorm(self, **obs):
        # q(llv | x, f)
        llv_loc, _ = self.encoder_llv(obs["x"]).chunk(2, dim=-1)
        # q(llv | x, f)
        rlv_loc, _ = self.encoder_rlv(obs["x"]).chunk(2, dim=-1)
        # q(llv | x, f)
        hv_loc, _ = self.encoder_hv(obs["x"]).chunk(2, dim=-1)
        return {
            "Left-Lung_volume": llv_loc,
            "Right-Lung_volume": rlv_loc,
            "Heart_volume": hv_loc,
        }
        
    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass

