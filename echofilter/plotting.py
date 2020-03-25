'''
Plotting utilities.
'''

import copy

import numpy as np
import matplotlib.pyplot as plt

from . import utils


TOP_COLOR = 'c'
BOT_COLOR = '#00ee00'

# ColorBrewer Paired values
TOP_COLOR = '#a6cee3'
BOT_COLOR = '#b2df8a'
TOP_COLOR_DARK = '#1f78b4'
BOT_COLOR_DARK = '#33a02c'

PASSIVE_COLOR = [.4, .4, .4]
REMOVED_COLOR = [0, 0, 1]


def plot_transect(
    transect,
    signal_type='Sv',
    x_scale='index',
    show_regions=True,
    top_color=TOP_COLOR,
    bot_color=BOT_COLOR,
    passive_color=PASSIVE_COLOR,
    removed_color=REMOVED_COLOR,
    linewidth=2,
):
    '''
    Plot a transect.

    Parameters
    ----------
    transect : dict
        Transect values.
    '''
    x_scale = x_scale.lower()

    if x_scale == 'index':
        tt = np.arange(transect['timestamps'].shape[0])
        xlabel = 'Sample index'
    elif x_scale == 'timestamp':
        tt = transect['timestamps']
        xlabel = 'Timestamp (s)'
    elif x_scale == 'time':
        tt = transect['timestamps'] - transect['timestamps'][0]
        xlabel = 'Time (s)'
    else:
        raise ValueError('Unsupported x_scale: {}'.format(x_scale))

    if signal_type == 'Sv_masked':
        signal = copy.deepcopy(transect['Sv'])
        signal[~transect['mask']] = np.nan
    else:
        signal = transect[signal_type]
    plt.pcolormesh(
        tt,
        transect['depths'],
        signal.T,
    )
    plt.plot(tt, transect['top'], top_color, linewidth=linewidth)
    plt.plot(tt, transect['bottom'], bot_color, linewidth=linewidth)

    if not show_regions or 'is_passive' not in transect:
        indices = []
    else:
        indices = np.nonzero(transect['is_passive'])[0]
    if len(indices) > 0:
        r_starts = [indices[0]]
        r_ends = []
        breaks = np.nonzero(indices[1:] - indices[:-1] > 1)[0]
        for break_idx in breaks:
            r_ends.append(indices[break_idx])
            r_starts.append(indices[break_idx + 1])
        r_ends.append(indices[-1])
        for r_start, r_end in zip(r_starts, r_ends):
            plt.fill_between(
                tt[[r_start, r_end]],
                transect['depths'][[0, 0]],
                transect['depths'][[-1, -1]],
                facecolor='none',
                hatch='//',
                edgecolor=passive_color,
                linewidth=0.0,
            )

    if not show_regions or 'is_removed' not in transect:
        indices = []
    else:
        indices = np.nonzero(transect['is_removed'])[0]
    if len(indices) > 0:
        r_starts = [indices[0]]
        r_ends = []
        breaks = np.nonzero(indices[1:] - indices[:-1] > 1)[0]
        for break_idx in breaks:
            r_ends.append(indices[break_idx])
            r_starts.append(indices[break_idx + 1])
        r_ends.append(indices[-1])
        for r_start, r_end in zip(r_starts, r_ends):
            plt.fill_between(
                tt[[r_start, r_end]],
                transect['depths'][[0, 0]],
                transect['depths'][[-1, -1]],
                facecolor='none',
                hatch='\\\\',
                edgecolor=removed_color,
                linewidth=0.0,
            )

    plt.tick_params(reset=True, color=(.2, .2, .2))
    plt.tick_params(labelsize=14)

    plt.gca().invert_yaxis()
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Depth (m)', fontsize=18)
