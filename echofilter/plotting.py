"""
Plotting utilities.
"""

import copy

import numpy as np
import matplotlib.pyplot as plt

from . import utils


TURBULENCE_COLOR = "c"
BOTTOM_COLOR = "#00ee00"

# ColorBrewer Paired values
TURBULENCE_COLOR = "#a6cee3"
TURBULENCE_COLOR_DARK = "#1f78b4"
BOTTOM_COLOR = "#b2df8a"
BOTTOM_COLOR_DARK = "#33a02c"
SURFACE_COLOR = "#4ba82a"
SURFACE_COLOR_DARK = "#255e11"

PASSIVE_COLOR = [0.4, 0.4, 0.4]
REMOVED_COLOR = "b"


def ensure_axes_inverted(axes=None, dir="y"):
    """
    Invert axis direction, if not already inverted.

    Parameters
    ----------
    axes : matplotlib.axes or None
        The axes to invert. If `None`, the current axes are used (default).
    dir : {"x", "y", "xy"}
        The axis to invert. Default is `"y"`.
    """
    if axes is None:
        axes = plt.gca()
    if "y" in dir:
        lim = axes.get_ylim()
        if lim[1] > lim[0]:
            axes.invert_yaxis()
    if "x" in dir:
        lim = axes.get_xlim()
        if lim[1] > lim[0]:
            axes.invert_xaxis()


def plot_indicator_hatch(
    indicator, xx=None, ymin=None, ymax=None, hatch="//", color="k"
):
    """
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
        Hatching pattern to use. Default is `"//"`.
    color : color, optional
        Color of the hatching pattern. Default is black.
    """

    if xx is None:
        xx = np.arange(len(indicator))
    if ymin is None or ymax is None:
        ylim = plt.gca().get_ylim()
    if ymin is None:
        ymin = ylim[0]
    if ymax is None:
        ymax = ylim[1]

    r_starts, r_ends = utils.get_indicator_onoffsets(indicator)

    if len(r_starts) == 0:
        return

    for r_start, r_end in zip(r_starts, r_ends):
        plt.fill_between(
            xx[[r_start, r_end]],
            [ymin, ymin],
            [ymax, ymax],
            facecolor="none",
            hatch=hatch,
            edgecolor=color,
            linewidth=0.0,
        )


def plot_mask_hatch(*args, hatch="//", color="k", border=False):
    """
    Plot hatching according to a mask shape.

    Parameters
    ----------
    X, Y : array-like, optional
        The coordinates of the values in Z.

        X and Y must both be 2-D with the same shape as `Z` (e.g. created via
        numpy.meshgrid), or they must both be 1-D such that ``len(X) == M`` is
        the number of columns in `Z` and ``len(Y) == N`` is the number of rows
        in `Z`.

        If not given, they are assumed to be integer indices, i.e.
        ``X = range(M)``, ``Y = range(N)``.
    Z : array-like(N, M)
        Indicator for which locations should be hatched. If `Z` is not a boolean
        array, any location where ``Z > 0`` will be hatched.
    hatch : str, optional
        The hatching pattern to apply. Default is "//".
    color : color, optional
        The color of the hatch. Default is black.
    border : bool, optional
        Whether to include border around hatch. Default is `False`.
    """
    args = list(args)
    args[-1] = args[-1] > 0
    if len(args) == 3:
        # Transpose Z if necessary, as it expects its shape the opposite way
        # around to what you might expect.
        if (
            args[0].shape[0] == args[2].shape[1]
            and args[1].shape[0] == args[2].shape[0]
        ):
            pass
        elif (
            args[0].shape[0] == args[2].shape[0]
            and args[1].shape[0] == args[2].shape[1]
        ):
            args[2] = args[2].T
    cs = plt.contourf(
        *args,
        levels=[-3.4e38, 0.5, 3.4e38],
        colors=["none", "none"],
        hatches=[None, hatch],
    )
    # Change the color of the hatches in each layer
    for collection in cs.collections:
        collection.set_edgecolor(color)
    # Doing this also colors in the box around each level.
    # We can remove the colored line around the levels by setting the
    # linewidth to 0.
    if not border:
        for collection in cs.collections:
            collection.set_linewidth(0.0)


def plot_transect(
    transect,
    signal_type=None,
    x_scale="index",
    show_regions=True,
    turbulence_color=TURBULENCE_COLOR,
    bottom_color=BOTTOM_COLOR,
    surface_color=SURFACE_COLOR,
    passive_color=PASSIVE_COLOR,
    removed_color=None,
    linewidth=1,
    cmap=None,
):
    r"""
    Plot a transect.

    Parameters
    ----------
    transect : dict
        Transect values.
    signal_type : str, optional
        The signal to plot as a heatmap. Default is `"Sv"` if present, or
        "signals" if not. If this is `"Sv_masked"`, the mask (given by
        `transect["mask"]`) is used to mask `transect["Sv"]` before plotting.
    x_scale : {"index", "timestamp" "time"}, optional
        Scaling for x-axis. If `"timestamp"`, the number of seconds since the
        Unix epoch is shown; if `"time"`, the amount of time in seconds since
        the start of the transect is shown. Default is `"index"`.
    show_regions : bool, optional
        Whether to show segments of data maked as removed or passive with
        hatching. Passive data is shown with `"/"` oriented lines, other removed
        timestamps with `"\"` oriented lines. Default is `True`.
    turbulence_color : color, optional
        Color of turbulence line. Default is `"#a6cee3"`.
    bottom_color : color, optional
        Color of bottom line. Default is `"#b2df8a"`.
    surface_color : color, optional
        Color of surface line. Default is `"#d68ade"`.
    passive_color : color, optional
        Color of passive segment hatching. Default is `[.4, .4, .4]`.
    removed_color : color, optional
        Color of removed segment hatching. Default is `"r"` if `cmap` is
        `"viridis"`, and `"b"` otherwise.
    linewidth : int
        Width of lines. Default is `2`.
    cmap : str, optional
        Name of a registered matplotlib colormap. If `None` (default), the
        current default colormap is used.
    """
    x_scale = x_scale.lower()

    if removed_color is not None:
        pass
    elif cmap is "viridis":
        removed_color = "r"
    else:
        removed_color = REMOVED_COLOR

    if signal_type is not None:
        pass
    elif "Sv" in transect:
        signal_type = "Sv"
    elif "signals" in transect:
        signal_type = "signals"
    else:
        raise ValueError(
            "No signal key found in transect. Available keys were: {}".format(
                transect.keys()
            )
        )

    turbulence_key = None
    for k in ("turbulence", "d_turbulence", "top", "d_top"):
        if k in transect:
            turbulence_key = k
            break

    bottom_key = None
    for k in ("bottom", "d_bottom", "bot", "d_bot"):
        if k in transect:
            bottom_key = k
            break

    surface_key = None
    for k in ("surface", "d_surface", "d_surf", "surf"):
        if k in transect:
            surface_key = k
            break

    if x_scale == "index":
        tt = np.arange(transect["timestamps"].shape[0])
        xlabel = "Sample index"
    elif x_scale == "timestamp":
        tt = transect["timestamps"]
        xlabel = "Timestamp (s)"
    elif x_scale == "time":
        tt = transect["timestamps"] - transect["timestamps"][0]
        xlabel = "Time (s)"
    else:
        raise ValueError("Unsupported x_scale: {}".format(x_scale))

    if signal_type == "Sv_masked":
        signal = copy.deepcopy(transect["Sv"])
        signal[~transect["mask"]] = np.nan
    else:
        signal = transect[signal_type]
    plt.pcolormesh(tt, transect["depths"], signal.T)
    if cmap is not None:
        plt.set_cmap(cmap)
    if surface_key is not None:
        plt.plot(tt, transect[surface_key], surface_color, linewidth=linewidth)
    if turbulence_key is not None:
        plt.plot(tt, transect[turbulence_key], turbulence_color, linewidth=linewidth)
    if bottom_key is not None:
        plt.plot(tt, transect[bottom_key], bottom_color, linewidth=linewidth)

    if show_regions:
        plot_indicator_hatch(
            transect.get("is_passive", []),
            xx=tt,
            ymin=transect["depths"][0],
            ymax=transect["depths"][-1],
            hatch="//",
            color=passive_color,
        )
        plot_indicator_hatch(
            transect.get("is_removed", []),
            xx=tt,
            ymin=transect["depths"][0],
            ymax=transect["depths"][-1],
            hatch="\\\\",
            color=removed_color,
        )
    if show_regions and "mask_patches" in transect:
        plot_mask_hatch(
            tt,
            transect["depths"],
            transect["mask_patches"],
            hatch="\\\\",
            color=removed_color,
            border=True,
        )

    plt.tick_params(reset=True, color=(0.2, 0.2, 0.2))
    plt.tick_params(labelsize=14)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("Depth (m)", fontsize=18)

    # Make sure y-axis is inverted (lowest depth at the top)
    ensure_axes_inverted(dir="y")


def plot_transect_predictions(transect, prediction, linewidth=1, cmap=None):
    """
    Plot the generated output for a transect against its ground truth data.

        - Ground truth data is shown in black, predictions in white.
        - Passive regions are hatched in / direction for ground truth, \\ for
          prediciton.
        - Removed regions are hatched in \\ direction for ground truth, / for
          prediction.

    Parameters
    ----------
    transect : dict
        Ground truth data for the transect.
    prediction : dict
        Predictions for the transect.
    linewidth : int
        Width of lines. Default is `2`.
    cmap : str, optional
        Name of a registered matplotlib colormap. If `None` (default), the
        current default colormap is used.
    """

    plot_transect(
        transect,
        x_scale="index",
        turbulence_color="k",
        bottom_color="k",
        surface_color="k",
        passive_color="k",
        removed_color="k",
        linewidth=linewidth,
        cmap=cmap,
    )

    # Convert output into lines
    for shape_y in (
        prediction["p_is_above_turbulence"].shape[-1],
        prediction["p_is_below_bottom"].shape[-1],
    ):
        if not np.allclose(prediction["depths"].shape, shape_y):
            print("Shape mismatch: {} {}".format(prediction["depths"].shape, shape_y))
    surface_depths = prediction["depths"][
        utils.last_nonzero(prediction["p_is_above_surface"] > 0.5, -1)
    ]
    turbulence_depths = prediction["depths"][
        utils.last_nonzero(prediction["p_is_above_turbulence"] > 0.5, -1)
    ]
    bottom_depths = prediction["depths"][
        utils.first_nonzero(prediction["p_is_below_bottom"] > 0.5, -1)
    ]

    tt = np.linspace(0, len(transect["timestamps"]) - 1, len(turbulence_depths))
    plt.plot(tt, surface_depths, "w", linewidth=linewidth)
    plt.plot(tt, turbulence_depths, "w", linewidth=linewidth)
    plt.plot(tt, bottom_depths, "w", linewidth=linewidth)

    # Mark removed areas
    plot_indicator_hatch(
        prediction.get("p_is_passive", np.array([])) > 0.5,
        xx=tt,
        ymin=transect["depths"][0],
        ymax=transect["depths"][-1],
        hatch="\\\\",
        color="w",
    )
    plot_indicator_hatch(
        prediction.get("p_is_removed", np.array([])) > 0.5,
        xx=tt,
        ymin=transect["depths"][0],
        ymax=transect["depths"][-1],
        hatch="//",
        color="w",
    )
    if "p_is_patch" in prediction:
        plot_mask_hatch(
            tt,
            prediction["depths"],
            prediction["p_is_patch"] > 0.5,
            hatch="//",
            color="w",
            border=True,
        )
    # Make sure y-axis is inverted (lowest depth at the top)
    ensure_axes_inverted(dir="y")
