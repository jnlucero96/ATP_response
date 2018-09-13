#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 13 September 2018 08:38:43

# Import internal python libraries
from sys import exit
# Import external python libraries
from numpy import nanmin, nanmax
from matplotlib import cm, colors, rcParams, rc
from matplotlib.style import use
from matplotlib.pyplot import subplots, close

# Import self-made libraries
from misc import get_unit_label

# define matplotlib settings:
use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}"
    ]

def plot_heatmap(
    data, var, var_name, var_unit, axes, 
    title=None, filename=None, maxval=None, minval=None
    ):
    """\
    Description:

    Inputs:

    Outputs:
    """

    n_grid = data.shape[0]
    units = get_unit_label(var_unit)

    if not title:
        title = var_name
    if not filename:
        filename = var + '_hmap.pdf'
    if not maxval:
        maxval = nanmax(data)
    if not minval:
        minval = min(nanmin(data), 0)
    
    cmap = cm.Blues
    cmap.set_bad(color='k')

    fig, ax = subplots(1, 1, figsize=(10, 10))
    image = ax.imshow(
        data, aspect='auto', interpolation='nearest', vmin=minval,
        vmax=maxval, cmap=cmap, extent=[0, 1, 0, 1], origin='lower'
    )

    ax.set_xlabel(axes[1], fontsize=24)
    ax.set_ylabel(axes[0], fontsize=24)
    ax.tick_params(labelsize=18)
    
    fig.tight_layout()

    left = 0.125  # the left side of the subplots of the figure
    right = 0.87    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.94      # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    cbar_axes = fig.add_axes([0.88, 0.1, 0.01, 0.84])
    cbar_axes.tick_params(labelsize=20)

    cbar = fig.colorbar(image, cax=cbar_axes)
    cbar.set_label(var_name + units, labelpad=10, fontsize=18)
    fig.text(0.5, 0.965, title, va='center', ha='center', fontsize=24)
    fig.savefig(filename)
    close('all')

    exit(0)


def plot_heatmap_grid(
    data, var, var_name, var_unit, axes, log_params,
    title=None, filename=None, maxval=None, minval=None
    ):
    """\
    Description:

    Inputs:

    Outputs:
    """
    n_grid = data.shape[0]
    
    units = get_unit_label(var_unit)

    if not title:
        title = var_name
    if not filename:
        filename = var + '_grid_hmap.pdf'
    if not maxval:
        maxval = nanmax(data)
    if not minval:
        minval = min(nanmin(data), 0)

    cmap = cm.Blues
    cmap.set_bad(color='k')

    fig, ax = subplots(
        n_grid, n_grid, sharex='all', sharey='all', figsize=(10,10)
        )

    if log_params[0]:
        y_min = 2**(1 - n_grid)
    else:
        y_min = 0
    y_max = 1

    if log_params[1]:
        x_min = 2**(1 - n_grid)
    else:
        x_min = 0
    x_max = 1

    for index1 in range(n_grid):
        for index2 in range(n_grid):
            image = ax[n_grid - index1 - 1][index2].imshow(
                data[index1, index2, :, :], aspect='auto', 
                interpolation='nearest', vmin=minval, vmax=maxval,
                cmap=cmap, extent=[x_min, x_max, y_min, y_max],
                origin='lower'
                )
    ax[0][0].set_yticks([y_min, y_max])
    ax[n_grid-1][n_grid-1].set_xticks([x_min, x_max])

    if log_params[0]:
        ax[0][0].set_yticklabels(
            [r'$2^{' + str(1-n_grid) + '}$', r'$2^{0}$']
        )
    if log_params[1]:
        ax[n_grid - 1][n_grid - 1].set_xticklabels(
            [r'$2^{' + str(1-n_grid) + '}$', r'$2^{0}$']
        )

    fig.text(0.5, 0.04, axes[1], va='center', ha='center', fontsize=24)
    fig.text(
        0.04, 0.5, axes[0], va='center', ha='center', 
        rotation='vertical', size=16
        )
    ax[n_grid-1][0].set_ylabel(axes[2], fontsize=24)
    ax[n_grid-1][0].set_xlabel(axes[3], fontsize=24)

    fig.tight_layout()

    left = 0.125  # the left side of the subplots of the figure
    right = 0.88    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.88      # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    cbar_axes = fig.add_axes([0.82, 0.15, 0.05, 0.7])
    cbar_axes.tick_params(labelsize=20)

    cbar = fig.colorbar(image, cax=cbar_axes)
    cbar.set_label(var_name + units, labelpad=10, fontsize=24)
    fig.suptitle(title)
    fig.savefig(filename)
    close('all')

    

