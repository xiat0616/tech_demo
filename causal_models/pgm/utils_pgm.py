import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# plt.rcParams['figure.facecolor'] = 'white'

def dice_coef(input, target):
    pred = input.view(-1).type(torch.DoubleTensor)
    truth = target.view(-1).type(torch.DoubleTensor)
    # dice coef
    dice_coef = (2.0 * (pred * truth).float().sum() + 1) / (
        pred.float().sum() + truth.float().sum() + 1
    )
    return dice_coef

def BCEloss(input, target):
    pred = input.view(-1).type(torch.DoubleTensor)
    truth = target.view(-1).type(torch.DoubleTensor)
    # BCE loss
    bce_loss = torch.nn.BCELoss()(pred, truth)
    return bce_loss

def BCEDiceloss(input, target):
    pred = input.view(-1).type(torch.DoubleTensor)
    truth = target.view(-1).type(torch.DoubleTensor)
    # BCE loss
    bce_loss = torch.nn.BCELoss()(pred, truth)
    # Dice Loss
    dice_coef = (2.0 * (pred * truth).float().sum() + 1) / (
        pred.float().sum() + truth.float().sum() + 1
    )
    return bce_loss + (1 - dice_coef)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def check_nan(input_dict):
    nans = 0
    for k, v in input_dict.items():
        k_nans = torch.isnan(v).sum()
        nans += k_nans
        if k_nans > 0:
            print(f'\nFound {k_nans} nan(s) in {k}, skipping step.')
    return nans


def update_stats(stats, elbo_fn):
    """Accumulate tracked summary statistics."""

    def _update(trace, dist='p'):
        for name, node in trace.nodes.items():
            if node['type'] == 'sample':
                k = 'log' + dist + '(' + name + ')'
                if k not in stats:
                    stats[k] = 0
                stats[k] += node['log_prob'].sum().item()
        return stats

    _update(elbo_fn.trace_storage['model'], dist='p')
    _update(elbo_fn.trace_storage['guide'], dist='q')
    return stats


def plot(x, fig=None, ax=None, nrows=1, cmap='Greys_r', norm=None, cbar=False, set_cbar_ticks=True, logger=None):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n * 4, 16))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j
            ax = [ax] if n == 1 else ax
            _x = x[i * n + j].squeeze()
            if norm is not None:
                norm = MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max())
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])

    # plt.tight_layout()

    if cbar:
        if fig:
            fig.subplots_adjust(wspace=-0.275, hspace=0.25)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                # cbar_ax = fig.add_axes([
                #     ax[idx].get_position().x0 + 0.0025, # left
                #     ax[idx].get_position().y1, # bottom
                #     0.003, # width
                #     ax[idx].get_position().height # height
                # ])
                cbar_ax = fig.add_axes([
                    ax[idx].get_position().x0,
                    ax[idx].get_position().y0 - 0.015,
                    ax[idx].get_position().width,
                    0.0075
                ])
                cbar = plt.colorbar(im[i * n + j], cax=cbar_ax,
                                    orientation="horizontal")  # , ticks=mticker.MultipleLocator(25)) #, ticks=mticker.AutoLocator())
                # cbar.ax.tick_params(rotation=0)
                # cbar.ax.locator_params(nbins=5)
                _x = x[i * n + j].squeeze()

                if set_cbar_ticks:
                    d = 20
                    _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                    _vmin = -(_vmin - (_vmin % d))
                    _vmax = _vmax - (_vmax % d)
                    lt = [_vmin, 0, _vmax]

                    if (np.abs(_vmin) - 0) > d or (_vmax - 0) > d:
                        lt.insert(1, _vmin // 2)
                        lt.insert(-2, _vmax // 2)
                    cbar.set_ticks(lt)
                else:
                    cbar.ax.locator_params(nbins=5)
                    cbar.formatter.set_powerlimits((0, 0))

                cbar.outline.set_visible(False)
    return fig, ax


@torch.no_grad()
def plot_cf(x, cf_x, pa, cf_pa, do, var_cf_x=None, num_images=8):
    n = num_images  # Number of columns
    x = (x[:n].detach().cpu() + 1) * 127.5
    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5

    fs = 16  # Font size
    m = 3 if var_cf_x is None else 4  # Number of rows
    s = 5
    fig, ax = plt.subplots(m, n, figsize=(n * s + 6, m * s))

    _, _ = plot(x, ax=ax[0])
    _, _ = plot(cf_x, ax=ax[1])
    _, _ = plot(cf_x - x, ax=ax[2], fig=fig, cmap='RdBu_r', cbar=True, 
                norm=MidpointNormalize(midpoint=0))
    if var_cf_x is not None:
        _, _ = plot(var_cf_x[:n].detach().sqrt().cpu(), fig=fig, cmap='jet', 
                    ax=ax[3], cbar=True, set_cbar_ticks=False)

    sex_categories = ['male', 'female']  # 0, 1
    finding_categories = ['No finding', 'Finding']  # 0, 1
    scanner_categories = ['Phillips', 'Imaging']  # 0, 1

    for j in range(n):
        msg = ''
        for i, (k, v) in enumerate(do.items()):
            if k == 'sex':
                vv = sex_categories[int(v[j].item())]
                kk = 's'
            elif k == 'finding':
                vv = finding_categories[int(v[j].item())]
                kk = 'f'
            elif k == 'scanner':
                vv = scanner_categories[int(v[j].item())]
                kk = 'sc'
            else:
                continue
            msg += kk + '{{=}}' + vv
            msg += ', ' if (i + 1) < len(list(do.keys())) else ''

        if 'sex' in pa.keys():
            s = str(sex_categories[int(pa['sex'][j].item())])
            f = str(finding_categories[int(pa['finding'][j].item())])
            sc = str(scanner_categories[int(pa['scanner'][j].item())])

        if 'sex' in pa.keys():
            ax[0, j].set_title(f's={s}, f={f}, sc={sc}',
                               pad=8, fontsize=fs - 4, 
                               multialignment='center', linespacing=1.5)
            ax[1, j].set_title(f'do(${msg}$)', fontsize=fs, pad=10)

        # Plot counterfactual
        if 'sex' in cf_pa.keys():
            cf_s = str(sex_categories[int(cf_pa['sex'][j].item())])
            cf_f = str(finding_categories[int(cf_pa['finding'][j].item())])
            cf_sc = str(scanner_categories[int(cf_pa['scanner'][j].item())])

            ax[1, j].set_xlabel(
                rf'$\widetilde{{s}}{{=}}{cf_s}, \ \widetilde{{f}}{{=}}{cf_f}, \ \widetilde{{sc}}{{=}}{cf_sc}$',
                labelpad=9, fontsize=fs - 4, multialignment='center', linespacing=1.25)

    ax[0, 0].set_ylabel('Observation', fontsize=fs + 2, labelpad=8)
    ax[1, 0].set_ylabel('Counterfactual', fontsize=fs + 2, labelpad=8)
    ax[2, 0].set_ylabel('Treatment Effect', fontsize=fs + 2, labelpad=8)
    if var_cf_x is not None:
        ax[3, 0].set_ylabel('Uncertainty', fontsize=fs + 2, labelpad=8)

    return fig


def calculate_loss(pred_batch, target_batch, loss_norm="l1", soft_loss="BCElogits"):
    """Calculate the losses for pred_batch"""
    loss = 0
    for k in pred_batch.keys():
        assert pred_batch[k].size() == target_batch[k].size(), (
            f"{k} size does not match, pred_batch size {pred_batch[k].size()}; target batch size {target_batch[k].size()}"
        )
        if k in ["sex", "finding", "scanner"]:
            if soft_loss == "BCElogits":
                loss += torch.nn.BCEWithLogitsLoss()(pred_batch[k], target_batch[k])
            elif soft_loss == "l1":
                loss += torch.nn.L1Loss()(pred_batch[k], target_batch[k])
    return loss
