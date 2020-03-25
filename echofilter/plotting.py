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


def plot_indicator_hatch(indicator, xx=None, ymin=None, ymax=None, hatch='//', color='k'):
    '''
    Plots a hatch across indicated segments along the x-axis of a plot.

    Parameters
    ----------
    indicator : numpy.ndarray vector
        Whether to include or exclude each column along the x-axis. Included
        columns are indicated with non-zero values.
    xx : numpy.ndarray vector, optional
        Values taken by indicator along the x-axis. If `None` (default), the
        indices of `indicator` are used: `arange(len(indicator))`.
    ymin : float, optional
        The lower y-value of the extent of the hatching. If `None` (default),
        the minimum y-value of the current axes is used.
    ymax : float, optional
        The upper y-value of the extent of the hatching. If `None` (default),
        the maximum y-value of the current axes is used.
    hatch : str, optional
        Hatching pattern to use. Default is `'//'`.
    color : color, optional
        Color of the hatching pattern. Default is black.
    '''

    if xx is None:
        xx = np.arange(len(indicator))
    if ymin is None or ymax is None:
        ylim = plt.gca().get_ylim()
    if ymin is None:
        ymin = ylim[0]
    if ymax is None:
        ymax = ylim[1]

    indices = np.nonzero(indicator)[0]

    if len(indices) == 0:
        return

    r_starts = [indices[0]]
    r_ends = []
    breaks = np.nonzero(indices[1:] - indices[:-1] > 1)[0]
    for break_idx in breaks:
        r_ends.append(indices[break_idx])
        r_starts.append(indices[break_idx + 1])
    r_ends.append(indices[-1])
    for r_start, r_end in zip(r_starts, r_ends):
        plt.fill_between(
            xx[[r_start, r_end]],
            [ymin, ymin],
            [ymax, ymax],
            facecolor='none',
            hatch=hatch,
            edgecolor=color,
            linewidth=0.0,
        )


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
    signal_type : str, optional
        The signal to plot as a heatmap. Default is `'Sv'`. If this is
        `'Sv_masked'`, the mask (given by `transect['mask']`) is used to
        mask `'transect['Sv']` before plotting.
    x_scale : {'index', 'timestamp' 'time'}, optional
        Scaling for x-axis. If `'timestamp'`, the number of seconds since the
        Unix epoch is shown; if `'time'`, the amount of time in seconds since
        the start of the transect is shown. Default is `'index'`.
    show_regions : bool, optional
        Whether to show segments of data maked as removed or passive with
        hatching. Passive data is shown with `'/'` oriented lines, other removed
        timestamps with `'\'` oriented lines. Default is `True`.
    top_color : color, optional
        Color of top line. Default is `'#a6cee3'`.
    bot_color : color, optional
        Color of bottom line. Default is `'#b2df8a'`.
    passive_color : color, optional
        Color of passive segment hatching. Default is `[.4, .4, .4]`.
    removed_color : color, optional
        Color of removed segment hatching. Default is `[0, 0, 1]`.
    linewidth : int
        Width of lines. Default is `2`.
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

    if show_regions:
        plot_indicator_hatch(
            transect.get('is_passive', []),
            xx=tt,
            ymin=transect['depths'][0],
            ymax=transect['depths'][-1],
            hatch='//',
            color=passive_color,
        )
        plot_indicator_hatch(
            transect.get('is_removed', []),
            xx=tt,
            ymin=transect['depths'][0],
            ymax=transect['depths'][-1],
            hatch='\\\\',
            color=removed_color,
        )

    plt.tick_params(reset=True, color=(.2, .2, .2))
    plt.tick_params(labelsize=14)

    plt.gca().invert_yaxis()
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Depth (m)', fontsize=18)
