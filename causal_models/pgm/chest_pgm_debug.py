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

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class FlowPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "sex": "binary",
            "finding": "binary",
        }
        # log space for sex 
        self.sex_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))
        self.finding_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))
        
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
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNet18(num_outputs=1, **kwargs)
            # q(finding | x) ~ Bernoulli(f(x))
            self.encoder_finding = ResNet18(num_outputs=1, **kwargs)

            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )
        else:
            if args.enc_net == "cnn":
                input_shape=(1,args.input_res,args.input_res)
                self.encoder_s = CNN(input_shape, num_outputs=1)
                self.encoder_finding = CNN(input_shape, num_outputs=1)

    def model(self, t=None):
        pyro.module("ChestPGM", self)
        # p(s), sex dist
        ps = dist.Bernoulli(logits=self.sex_logit).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(finding), independent finding dist
        pfinding = dist.Bernoulli(logits=self.finding_logit).to_event(1)
        finding = pyro.sample("finding", pfinding)

        return {
            "sex": sex,
            "finding": finding,
        }

    def guide(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
                pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            if obs["finding"] is None:
                finding_prob = torch.sigmoid(self.encoder_finding(obs["x"]))
                pyro.sample("finding", dist.Bernoulli(probs=finding_prob).to_event(1))

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classifiers
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
            qs_x = dist.Bernoulli(probs=s_prob).to_event(1)
            pyro.sample("sex_aux", qs_x, obs=obs["sex"])

            # q(finding | x)
            finding_prob = torch.sigmoid(self.encoder_finding(obs["x"]))
            qfinding_x = dist.Bernoulli(probs=finding_prob).to_event(1)
            pyro.sample("finding_aux", qfinding_x, obs=obs["finding"])


    def predict(self, **obs):
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
        # q(finding | x)
        finding_prob = torch.sigmoid(self.encoder_finding(obs["x"]))

        return {
            "sex": s_prob,
            "finding": finding_prob,
        }

    def predict_unnorm(self, **obs):
        # q(s | x)
        s_logits = self.encoder_s(obs["x"])
        # q(finding | x)
        finding_logits = self.encoder_finding(obs["x"])

        return {
            "sex": s_logits,
            "finding": finding_logits,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass
