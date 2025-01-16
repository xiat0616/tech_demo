import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

act = nn.Softplus(beta=np.log(2))


def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    q_scale = act(q_logscale)
    p_scale = act(p_logscale)
    out = dist.kl_divergence(
        dist.Normal(q_loc, q_scale),
        dist.Normal(p_loc, p_scale),
    )
    return out

@torch.jit.script
def sample_gaussian(loc, logscale):
    return loc + act(logscale) * torch.randn_like(loc)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class Block(nn.Module):
    def __init__(
        self,
        args,
        in_width,
        out_width,
        kernel_size=3,
        residual=True,
        down_rate_h=None,
        down_rate_w=None,
        condition=True,
    ):
        super().__init__()
        self.down_h = down_rate_h
        self.down_w = down_rate_w
        self.residual = residual
        self.condition = condition
        bottleneck = int(in_width / args.bottleneck)
        padding = 0 if kernel_size == 1 else 1

        activation = nn.SiLU()
        self.conv = nn.Sequential(
            nn.GroupNorm(min(32, in_width // 4), in_width),
            activation,
            nn.Conv2d(in_width, bottleneck, kernel_size, 1, padding),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(32, bottleneck // 4), bottleneck),
            activation,
            nn.Dropout(p=args.p_dropout),
            zero_module(nn.Conv2d(bottleneck, out_width, kernel_size, 1, padding)),
        )

        if self.condition:
            self.pa_proj = nn.Sequential(
                activation, nn.Linear(args.embd_dim, bottleneck)
            )

        if self.residual:
            if in_width != out_width:
                self.skip_conn = nn.Conv2d(in_width, out_width, 1, 1)
            else:
                self.skip_conn = nn.Identity()

    def forward(self, x, pa=None):
        out = self.conv(x)
        out = out + self.pa_proj(pa)[..., None, None] if self.condition else out
        out = self.conv2(out)
        out = out + self.skip_conn(x) if self.residual else out
        if self.down_h: # If self.down_h is not None
            if isinstance(self.down_h, float):
                out = F.adaptive_avg_pool2d(out, (int(out.shape[-2] / self.down_h), int(out.shape[-1]/ self.down_w)))
            else:
                out = F.avg_pool2d(out, kernel_size=(self.down_h,self.down_w), stride=(self.down_h, self.down_w))
        return out

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # parse architecture
        stages = []
        for i, stage in enumerate(args.enc_arch.split(",")):
            start = stage.index("b") + 1
            end = stage.index("d") if "d" in stage else None
            n_blocks = int(stage[start:end])

            if i == 0:  # define network stem
                if n_blocks == 0 and "d" not in stage:
                    print("Using stride=2 conv encoder stem.")
                    stem_width, stem_stride = args.widths[1], 2
                    continue
                else:
                    stem_width, stem_stride = args.widths[0], 1
                self.stem = nn.Conv2d(
                    args.input_channels,
                    stem_width,
                    kernel_size=7,
                    stride=stem_stride,
                    padding=3,
                )

            stages += [(args.widths[i], None, None) for _ in range(n_blocks)]
            if "d" in stage:  # downsampling block
                _down = stage.split("d")[1]
                down_h = _down.split("_")[0]
                down_w = _down.split("_")[1]
                down_h = float(down_h) if "." in down_h else int(down_h)
                down_w = float(down_w) if "." in down_w else int(down_w)
                stages += [(args.widths[i + 1], down_h, down_w)]
        blocks = []
        for i, (width, d_h, d_w) in enumerate(stages):
            prev_width = stages[max(0, i - 1)][0]
            blocks.append(Block(args, prev_width, width, down_rate_h=d_h, down_rate_w=d_w, condition=False))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.stem(x)
        acts = {}
        for block in self.blocks:
            x = block(x)
            acts[x.size(-2)] = x # Use h as key
        return acts

class DecoderBlock(nn.Module):
    def __init__(self, args, in_width, out_width, z_dim, res_h, res_w):
        super().__init__()
        self.res_h = res_h
        self.res_w = res_w

        self.stochastic = self.res_w <= args.z_max_res
        self.z_dim = z_dim
        self.cond_prior = args.cond_prior
        self.q_correction = args.q_correction
        k = 3 if self.res_w > 2 else 1

        self.prior = Block(
            args,
            in_width,
            2 * self.z_dim + in_width,
            kernel_size=k,
            residual=False,
            condition=self.cond_prior,
        )
        if self.stochastic:
            self.posterior = Block(
                args,
                2 * in_width,
                2 * self.z_dim,
                kernel_size=k,
                residual=False,
            )
        self.z_proj = nn.Conv2d(self.z_dim, in_width, 1)
        if not self.q_correction:  # for no posterior correction
            self.z_feat_proj = nn.Conv2d(self.z_dim + in_width, out_width, 1)
        self.conv = Block(args, in_width, out_width, kernel_size=k)

    def forward_prior(self, z, pa=None, t=None):
        # print(f"z: {z.size()}, pa: {pa.size()}")
        pz = self.prior(z, pa)
        p_loc = pz[:, : self.z_dim, ...]
        p_logscale = pz[:, self.z_dim : 2 * self.z_dim, ...]  # noqa
        p_features = pz[:, 2 * self.z_dim :, ...]  # noqa
        if t is not None:
            p_logscale = p_logscale + torch.tensor(t, device=z.device).log()
        return p_loc, p_logscale, p_features

    def forward_posterior(self, z, x, pa, t=None):
        h = torch.cat([z, x], dim=1)  # previous z and obs x
        q_loc, q_logscale = self.posterior(h, pa).chunk(2, dim=1)
        if t is not None:
            q_logscale = q_logscale + torch.tensor(t, device=z.device).log()
        return q_loc, q_logscale


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # parse architecture
        stages = []
        rev_widths = args.widths[::-1]
        for i, stage in enumerate(args.dec_arch.split(",")):
            res_h = int(stage.split('b')[0].split("_")[0])
            res_w = int(stage.split('b')[0].split("_")[1])
            n_blocks = int(stage[stage.index("b") + 1 :])  # noqa
            stages += [(res_h, res_w, rev_widths[i], args.z_dim[i]) for _ in range(n_blocks)]
        self.blocks = []
        for i, (res_h, res_w, width, z_dim) in enumerate(stages):
            next_width = stages[min(len(stages) - 1, i + 1)][2]
            self.blocks.append(DecoderBlock(args=args, 
                                            in_width=width, 
                                            out_width=next_width, 
                                            z_dim=z_dim, 
                                            res_h=res_h, 
                                            res_w=res_w))
        self.blocks = nn.ModuleList(self.blocks)
        # bias params
        self.all_res_h = list(np.unique([stages[i][0] for i in range(len(stages))]))
        self.all_res_w = list(np.unique([stages[i][1] for i in range(len(stages))]))
        bias = []
        for i, (res_h, res_w) in enumerate(zip(self.all_res_h, self.all_res_w)):
            if res_w <= args.bias_max_res:
                bias.append(nn.Parameter(torch.zeros(1, rev_widths[i], res_h, res_w)))
        self.bias = nn.ParameterList(bias)
        self.cond_prior = args.cond_prior

        self.pa_embd = nn.Sequential(
            nn.Linear(args.context_dim, args.embd_dim),
            nn.SiLU(),
            nn.Linear(args.embd_dim, args.embd_dim),
        )

    def forward(self, parents, x=None, t=None, abduct=False, latents=[]):
        # learnt params for each resolution r
        bias = {r.shape[2]: r for r in self.bias} # bias use res_h as key
        h = bias[self.all_res_h[0]].repeat(parents.shape[0], 1, 1, 1)  # h_init
        z = h  # for exogenous prior
        pa = self.pa_embd(parents[:, :, 0, 0]) # pa: [bs, embed_dim]
        pa_sto = pa_det = pa

        stats = []
        for i, block in enumerate(self.blocks):
            res_h, res_w = block.res_h, block.res_w  # current block resolution, e.g. 64x48
            # print(f"i: {i}, res_h: {res_h}, res_w: {res_h}")
            if h.size(-1) < res_h:  # upsample previous layer output
                b = bias[res_h] if res_h in bias.keys() else 0  # broadcasting
                h = b + F.interpolate(h, scale_factor=(res_h / h.shape[-2], res_w / h.shape[-1]))

            if block.q_correction:
                p_input = h  # current prior depends on previous posterior
            else:  # current prior depends on previous prior only, upsample previous prior latent z
                p_input = (
                    b + F.interpolate(z, scale_factor=(res_h / z.shape[-2], res_w / z.shape[-1]))
                    if z.size(-2) < res_h
                    else z
                )
            p_loc, p_logscale, p_feat = block.forward_prior(p_input, pa_sto, t=t)

            if block.stochastic:
                if x is not None:  # z_i ~ q(z_i | z_<i, x, pa_x)
                    # print(f"x.keys: {x.keys()}, res_h: {res_h}")
                    q_loc, q_logscale = block.forward_posterior(h, x[res_h], pa, t=t)
                    z = sample_gaussian(q_loc, q_logscale)
                    stat = dict(kl=gaussian_kl(q_loc, q_logscale, p_loc, p_logscale))
                    if abduct:  # abduct exogenous noise
                        if block.cond_prior:  # z* if conditional prior
                            stat.update(
                                dict(
                                    z={"z": z, "q_loc": q_loc, "q_logscale": q_logscale}
                                )
                            )
                        else:
                            stat.update(dict(z=z))  # z if exogenous prior
                    stats.append(stat)
                else:  # x is not given
                    try:  # forward fixed latents z or z*
                        z = latents[i]
                    except:  # noqa E722
                        z = sample_gaussian(p_loc, p_logscale)
                        if abduct and block.cond_prior:  # for abducting z*
                            stats.append(
                                dict(z={"p_loc": p_loc, "p_logscale": p_logscale})
                            )
            else:  # deterministic path
                z = p_loc
            h = h + p_feat  # merge features from prior
            h = h + block.z_proj(z)  # merge stochastic latent variable z
            h = block.conv(h, pa_det)
            if not block.q_correction:
                if (i + 1) < len(
                    self.blocks
                ):  # z independent of pa_x for next layer prior
                    z = block.z_feat_proj(torch.cat([z, p_feat], dim=1))
        return h, stats


class DGaussNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        width = args.widths[0]
        self.x_loc = nn.Sequential(
            nn.GroupNorm(min(32, width // 4), width),
            nn.SiLU(),
            zero_module(nn.Conv2d(width, args.input_channels, 3, padding=1)),
        )
        self.x_logscale = nn.Sequential(
            nn.GroupNorm(min(32, width // 4), width),
            nn.SiLU(),
            zero_module(nn.Conv2d(width, args.input_channels, 3, padding=1)),
        )

        self.covariance = args.x_like.split("_")[0]
        if self.covariance == "fixed":
            assert args.std_init > 0
            self.logscale = torch.log(torch.tensor(args.std_init)).float()

    def forward(self, h, x=None, t=None):
        loc, log_scale = self.x_loc(h), self.x_logscale(h)
        if self.covariance != "fixed":
            log_scale = self.x_logscale(h)
        else:
            log_scale = (
                torch.ones_like(loc, requires_grad=False, device=loc.device)
                * self.logscale
            )

        if t is not None:
            log_scale = log_scale + torch.tensor(t, device=h.device).log()
        return loc, log_scale

    def approx_cdf(self, x):
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, h, x):
        loc, logscale = self.forward(h, x)
        centered_x = x - loc
        std = act(logscale)
        prec = 1 / std
        plus_in = prec * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = prec * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(self, h, return_loc=True, t=None):
        if return_loc:
            x, log_scale = self.forward(h)
        else:
            loc, log_scale = self.forward(h, t)
            x = loc + act(log_scale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, act(log_scale)


class HVAE2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        if args.x_like.split("_")[1] == "dgauss":
            self.likelihood = DGaussNet(args)
        else:
            NotImplementedError(f"{args.x_like} not implemented.")
        self.cond_prior = args.cond_prior
        self.free_bits = args.kl_free_bits
        self.register_buffer("log2", torch.tensor(2.0).log())

    def forward(self, x, parents, beta=1, cr_transf=None):
        acts = self.encoder(x)
        h, stats = self.decoder(parents=parents, x=acts)
        nll_pp = self.likelihood.nll(h, x)
        if self.free_bits > 0:
            free_bits = torch.tensor(self.free_bits).type_as(nll_pp)
            kl_pp = 0.0
            for stat in stats:
                kl_pp += torch.maximum(
                    free_bits, stat["kl"].sum(dim=(2, 3)).mean(dim=0)
                ).sum()
        else:
            kl_pp = torch.zeros_like(nll_pp)
            for _, stat in enumerate(stats):
                kl_pp += stat["kl"].sum(dim=(1, 2, 3))
        kl_pp = kl_pp / np.prod(x.shape[1:])  # per pixel
        kl_pp = kl_pp.mean()
        nll_pp = nll_pp.mean()
        nelbo = nll_pp + beta * kl_pp  # negative elbo (free energy)
        return dict(elbo=nelbo, nll=nll_pp, kl=kl_pp)

    def sample(self, parents, return_loc=True, t=None):
        h, _ = self.decoder(parents=parents, t=t)
        return self.likelihood.sample(h, return_loc, t=t)

    def abduct(self, x, parents, cf_parents=None, t=None, alpha=0.8):
        acts = self.encoder(x)
        _, q_stats = self.decoder(
            x=acts, parents=parents, abduct=True, t=t
        )  # q(z|x,pa)
        q_stats = [s["z"] for s in q_stats]

        if self.cond_prior and cf_parents is not None:
            _, p_stats = self.decoder(parents=cf_parents, abduct=True, t=t)  # p(z|pa*)
            p_stats = [s["z"] for s in p_stats]

            cf_zs = []
            t = torch.tensor(t, device=x.device)  # z* sampling temperature

            for i in range(len(q_stats)):
                # from z_i ~ q(z_i | z_{<i}, x, pa)
                q_loc = q_stats[i]["q_loc"]
                q_scale = act(q_stats[i]["q_logscale"])

                # abduct exogenouse noise u ~ N(0,I)
                u = (q_stats[i]["z"] - q_loc) / q_scale

                # p(z_i | z_{<i}, pa*)
                p_loc = p_stats[i]["p_loc"]
                act(p_stats[i]["p_logscale"]).pow(2)

                # mixture distribution: r(z_i | z_{<i}, x, pa, pa*)
                #  = a*q(z_i | z_{<i}, x, pa) + (1-a)*p(z_i | z_{<i}, pa*)
                r_loc = alpha * q_loc + (1 - alpha) * p_loc

                # sample: z_i* ~ r(z_i | z_{<i}, x, pa, pa*)
                r_scale = q_scale.pow(2)
                r_scale = r_scale * t if t is not None else r_scale
                cf_zs.append(r_loc + r_scale * u)
            return cf_zs
        else:
            return q_stats  # zs

    def forward_latents(self, latents, parents, t=None):
        h, _ = self.decoder(latents=latents, parents=parents, t=t)
        return self.likelihood.sample(h, t=t)
