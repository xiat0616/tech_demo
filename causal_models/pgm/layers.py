import pyro
import torch
import torch.nn as nn
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions.conditional import *
from typing import Dict
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost
from torch.distributions.transforms import  Transform
import torch.nn.functional as F
import torch.nn as nn
from pyro.distributions.torch import Gumbel
from pyro.distributions.torch import Categorical
# from torch.distributions.categorical import Categorical
from pyro.distributions.torch_distribution import ExpandedDistribution
from torch import Tensor

class TraceStorage_ELBO(pyro.infer.Trace_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace

class SoftmaxCentered(Transform):
    """
    Implements softmax as a bijection, the forward transformation appends a value to the
    input and the inverse removes it. The appended coordinate represents a pivot, e.g., 
    softmax(x) = exp(x-c) / sum(exp(x-c)) where c is the implicit last coordinate.

    Adapted from a Tensorflow implementation: https://tinyurl.com/48vuh7yw 
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, temperature: float = 1.):
        super().__init__()
        self.temperature = temperature

    def __call__(self, x: Tensor):
        zero_pad = torch.zeros(*x.shape[:-1], 1, device=x.device)
        x_padded = torch.cat([x, zero_pad], dim=-1)
        return (x_padded / self.temperature).softmax(dim=-1)

    def _inverse(self, y: Tensor):
        log_y = torch.log(y.clamp(min=1e-12))
        unorm_log_probs = log_y[..., :-1] - log_y[..., -1:]
        return unorm_log_probs * self.temperature

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor):
        """ log|det(dy/dx)| """
        Kplus1 = torch.tensor(y.size(-1), dtype=y.dtype, device=y.device)
        return 0.5 * Kplus1.log() + torch.sum(torch.log(y.clamp(min=1e-12)), dim=-1)

    def forward_shape(self, shape: torch.Size):
        return shape[:-1] + (shape[-1] + 1,)  # forward appends one dim

    def inverse_shape(self, shape: torch.Size):
        if shape[-1] <= 1:
            raise ValueError
        return shape[:-1] + (shape[-1] - 1,)  # inverse removes last dim

class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context):
        # print(f"self.context_nn(context): {self.context_nn(context)}")
        loc, log_scale = self.context_nn(context)
        return torch.distributions.transforms.AffineTransform(
            loc, log_scale.exp(), event_dim=self.event_dim)


class MLP(nn.Module):
    def __init__(self, num_inputs=1, width=32, num_outputs=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, num_outputs),
        )

    def forward(self, x):
        return self.mlp(x)


class CNN(nn.Module):
    def __init__(self, in_shape=(1, 192, 192), width=16, num_outputs=1, context_dim=0):
        super().__init__()
        in_channels = in_shape[0]
        s = 2 if in_shape[1] > 64 else 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, s, 3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(width, 2*width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2*width),
            nn.ReLU(),
            nn.Conv2d(2*width, 2*width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*width),
            nn.ReLU(),
            nn.Conv2d(2*width, 4*width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4*width),
            nn.ReLU(),
            nn.Conv2d(4*width, 4*width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4*width),
            nn.ReLU(),
            nn.Conv2d(4*width, 8*width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8*width),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(8*width + context_dim, 8*width, bias=False),
            nn.BatchNorm1d(8*width),
            nn.ReLU(),
            nn.Linear(8*width, num_outputs)
        )

    def forward(self, x, y=None):
        x = self.cnn(x).mean(dim=(-2, -1))  # avg pool
        if y is not None:
            x = torch.cat([x, y], dim=-1)
        return self.fc(x)

class ArgMaxGumbelMax(Transform):
    r"""ArgMax as Transform, but inv conditioned on logits"""
    def __init__(self, logits, event_dim=0, cache_size=0):
        super(ArgMaxGumbelMax, self).__init__(cache_size=cache_size)
        self.logits = logits
        self._event_dim = event_dim
        self._categorical = Categorical(logits=self.logits).to_event(0)

    @property
    def event_dim(self):
        return self._event_dim

    def __call__(self, gumbels):
        """
        Computes the forward transform 
        """
        assert self.logits!=None, "Logits not defined."

        if self._cache_size == 0:
            return self._call(gumbels)

        y = self._call(gumbels)
        return y

    def _call(self, gumbels):
        """
        Abstract method to compute forward transformation.
        """
        assert self.logits!=None, "Logits not defined."
        y = gumbels  + self.logits
        # print(f'y: {y}')
        # print(f'logits: {self.logits}')
        return y.argmax(-1, keepdim=True)

    @property
    def domain(self):
        """"
        Domain of input(gumbel variables), Real 
        """
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)
        
    @property
    def codomain(self):
        """"
        Domain of output(categorical variables), should be natural numbers, but set to Real for now 
        """
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def inv(self, k):
        """Infer the gumbels noises given k and logits."""
        assert self.logits!=None, "Logits not defined."
        
        uniforms = torch.rand(self.logits.shape, dtype=self.logits.dtype, device=self.logits.device)
        gumbels = -((-(uniforms.log())).log())
        # (batch_size, num_classes) mask to select kth class
        mask = F.one_hot(k.squeeze(-1).to(torch.int64), num_classes=self.logits.shape[-1])
        # (batch_size, 1) select topgumbel for truncation of other classes
        topgumbel = (mask * gumbels).sum(dim=-1, keepdim=True) - (mask * self.logits).sum(dim=-1, keepdim=True)
        mask = 1 - mask  # invert mask to select other != k classes
        g = gumbels + self.logits
        # (batch_size, num_classes)
        epsilons = -torch.log(mask * torch.exp(-g) + torch.exp(-topgumbel)) - (mask * self.logits)
        return epsilons
    
    def log_abs_det_jacobian(self, x, y):
        """ We use the log_abs_det_jacobian to account for the categorical prob
            x: Gumbels; y: argmax(x+logits)
            return log prob of p(y=y).
        """
        return -self._categorical.log_prob(y.squeeze(-1)).unsqueeze(-1)

class ConditionalGumbelMax(ConditionalTransformModule):
    r"""Given gumbels+logits, output the OneHot Categorical"""
    def __init__(self, context_nn, event_dim=0, **kwargs):
        # The logits_nn which predict the logits given ages: 
        super().__init__(**kwargs)
        self.context_nn = context_nn  
        self.event_dim = event_dim

    def condition(self, context):
        """Given context (age), output the Categorical results"""
        logits = self.context_nn(context) # The logits for calculating argmax(Gumbel + logits)
        return ArgMaxGumbelMax(logits)
    
    def _logits(self, context):
        """Return logits given context"""
        return self.context_nn(context)

    @property
    def domain(self):
        """"
        Domain of input(gumbel variables), Real 
        """
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)
        
    @property
    def codomain(self):
        """"
        Domain of output(categorical variables), should be natural numbers, but set to Real for now 
        """
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)
    

class TransformedDistributionGumbelMax(TransformedDistribution, TorchDistributionMixin):
    r""" Define a TransformedDistribution class for Gumbel max
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def log_prob(self, value):
        """
        We do not use the log_prob() of the base Gumbel distribution, because the likelihood for
        each class for Gumbel Max sampling is determined by the logits.
        """
        # print("This happens")
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                                 event_dim - transform.domain.event_dim)
            y = x
        return log_prob
        
class ConditionalTransformedDistributionGumbelMax(ConditionalTransformedDistribution):

    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        transforms = [t.condition(context) for t in self.transforms]
        return TransformedDistributionGumbelMax(base_dist, transforms)

    def clear_cache(self):
        pass