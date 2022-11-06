"""
Input/Output handling for raw Echoview files.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import OrderedDict
import csv
import datetime
import os
import textwrap
import warnings

import numpy as np
import scipy.interpolate
import scipy.ndimage
import skimage.measure
import pandas as pd

from . import utils
from ..ui import style


ROOT_DATA_DIR = "/data/dsforce/surveyExports"

TRANSECT_FIELD_TYPES = {
    "Ping_index": int,
    "Distance_gps": float,
    "Distance_vl": float,
    "Ping_date": str,
    "Ping_time": str,
    "Ping_milliseconds": float,
    "Latitude": float,
    "Longitude": float,
    "Depth_start": float,
    "Depth_stop": float,
    "Range_start": float,
    "Range_stop": float,
    "Sample_count": int,
}


def transect_reader(fname):
    """
    Creates a generator which iterates through a survey csv file.

    Parameters
    ----------
    fname: str
        Path to survey CSV file.

    Returns
    -------
    generator
        Yields a tupule of `(metadata, data)`, where metadata is a dict,
        and data is a :class:`numpy.ndarray`. Each yield corresponds to a single
        row in the data. Every row (except for the header) is yielded.
    """
    metadata_header = []
    with open(fname, "rb") as hf:
        for i_row, row in enumerate(hf):
            try:
                row = row.decode("utf-8-sig" if i_row == 0 else "utf-8")
            except:
                if i_row == 0:
                    raise
                print(
                    "Row {} of {} contained a byte which is not in UTF-8"
                    " and will be skipped.".format(i_row, fname)
                )
                continue
            row = row.split(",")
            row = [entry.strip() for entry in row]
            if i_row == 0:
                metadata_header = row
                continue
            metadata = row[: len(metadata_header)]
            metadata_d = OrderedDict()
            for k, v in zip(metadata_header, metadata):
                if k in TRANSECT_FIELD_TYPES:
                    metadata_d[k] = TRANSECT_FIELD_TYPES[k](v)
                else:
                    metadata_d[k] = v
            data = np.array([float(x) for x in row[len(metadata_header) :]])
            yield metadata_d, data


def count_lines(filename):
    """
    Count the number of lines in a file.

    Parameters
    ----------
    filename : str
        Path to file.

    Returns
    -------
    int
        Number of lines in file.
    """
    with open(filename, "rb") as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def transect_loader(
    fname,
    skip_lines=0,
    warn_row_overflow=None,
    row_len_selector="mode",
):
    """
    Loads an entire survey transect CSV.

    Parameters
    ----------
    fname : str
        Path to survey CSV file.
    skip_lines : int, optional
        Number of initial entries to skip. Default is 0.
    warn_row_overflow : bool or int, optional
        Whether to print a warning message if the number of elements in a
        row exceeds the expected number. If this is an int, this is the number
        of times to display the warnings before they are supressed. If this
        is `True`, the number of outputs is unlimited. If `None`, the
        maximum number of underflow and overflow warnings differ: if
        `row_len_selector` is `"init"` or `"min"`, underflow always produces a
        message and the overflow messages stop at 2; otherwise the values are
        reversed. Default is `None`.
    row_len_selector : {"init", "min", "max", "median", "mode"}, optional
        The method used to determine which row length (number of depth samples)
        to use. Default is `"mode"`, the most common row length across all
        the measurement timepoints.

    Returns
    -------
    numpy.ndarray
        Timestamps for each row, in seconds. Note: not corrected for timezone
        (so make sure your timezones are internally consistent).
    numpy.ndarray
        Depth of each column, in metres.
    numpy.ndarray
        Survey signal (Sv, for instance). Units match that of the file.
    """

    row_len_selector = row_len_selector.lower()
    if row_len_selector in {"init", "min"}:
        expand_for_overflow = False
    else:
        expand_for_overflow = True

    if warn_row_overflow is True:
        warn_row_overflow = np.inf

    if warn_row_overflow is not None:
        warn_row_underflow = warn_row_overflow
    elif expand_for_overflow:
        warn_row_underflow = 2
        warn_row_overflow = np.inf
    else:
        warn_row_underflow = np.inf
        warn_row_overflow = 2

    # We remove one from the line count because of the header
    # which is excluded from output
    n_lines = count_lines(fname) - 1
    n_distances = 0

    # Initialise output array
    for i_line, (meta, row) in enumerate(transect_reader(fname)):
        if i_line < min(n_lines, max(1, skip_lines)):
            continue
        n_depths_init = len(row)
        depth_start_init = meta["Depth_start"]
        depth_stop_init = meta["Depth_stop"]
        break

    n_depth_exp = n_depths_init

    data = np.empty((n_lines - skip_lines, n_depth_exp))
    data[:] = np.nan
    timestamps = np.empty((n_lines - skip_lines))
    timestamps[:] = np.nan

    row_lengths = np.empty((n_lines - skip_lines), dtype=np.int)
    row_depth_starts = np.empty((n_lines - skip_lines))
    row_depth_ends = np.empty((n_lines - skip_lines))

    n_warn_overflow = 0
    n_warn_underflow = 0

    n_entry = 0
    for i_line, (meta, row) in enumerate(transect_reader(fname)):
        if i_line < skip_lines:
            continue
        i_entry = i_line - skip_lines

        # Track the range of depths used in the row with this length
        row_lengths[i_entry] = len(row)
        row_depth_starts[i_entry] = meta["Depth_start"]
        row_depth_ends[i_entry] = meta["Depth_stop"]

        if len(row) > n_depth_exp:
            if n_warn_overflow < warn_row_overflow:
                print(
                    "Row {} of {} exceeds expected n_depth of {} with {}".format(
                        i_line, fname, n_depth_exp, len(row)
                    )
                )
                n_warn_overflow += 1
            if expand_for_overflow:
                data = np.pad(
                    data,
                    ((0, 0), (0, len(row) - n_depth_exp)),
                    mode="constant",
                    constant_values=np.nan,
                )
                n_depth_exp = len(row)

        if len(row) < n_depth_exp:
            if n_warn_underflow < warn_row_underflow:
                print(
                    "Row {} of {} shorter than expected n_depth_exp of {} with {}".format(
                        i_line, fname, n_depth_exp, len(row)
                    )
                )
                n_warn_underflow += 1
            data[i_entry, : len(row)] = row
        else:
            data[i_entry, :] = row[:n_depth_exp]

        timestamps[i_entry] = datetime.datetime.strptime(
            "{}T{}.{:06d}".format(
                meta["Ping_date"],
                meta["Ping_time"],
                int(1000 * float(meta["Ping_milliseconds"])),
            ),
            "%Y-%m-%dT%H:%M:%S.%f",
        ).timestamp()
        n_entry += 1

    # Turn NaNs into NaNs (instead of extremely negative number)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in less")
        warnings.filterwarnings("ignore", "invalid value encountered in greater")
        # 9.9e+37 and -9.9e+37 are special values indicating missing data
        # https://support.echoview.com/WebHelp/Reference/File_formats/Export_file_formats/Special_Export_Values.htm
        data[data < -1e37] = np.nan
        data[data > 1e37] = np.nan

    # Trim timestamps dimension down to size
    timestamps = timestamps[:n_entry]
    data = data[:n_entry]
    row_lengths = row_lengths[:n_entry]
    row_depth_starts = row_depth_starts[:n_entry]
    row_depth_ends = row_depth_ends[:n_entry]

    # Work out what row length we should return
    if row_len_selector == "init":
        n_depth_use = n_depths_init
    elif row_len_selector == "min":
        n_depth_use = np.min(row_lengths)
    elif row_len_selector == "max":
        n_depth_use = np.max(row_lengths)
    elif row_len_selector == "median":
        n_depth_use = np.median(row_lengths)
        # If the median is half-way between two values, round up
        if n_depth_use not in row_depth_starts:
            n_depth_use = int(np.round(n_depth_use))
        # If the median is still not between values, drop the last value
        # to make the array be odd, guaranteeing the median is an observed
        # value, not an intermediary.
        if n_depth_use not in row_depth_starts:
            n_depth_use = np.median(row_lengths[:-1])
    elif row_len_selector == "mode":
        n_depth_use = utils.mode(row_lengths)
    else:
        raise ValueError(
            "Unsupported row_len_selector value: {}".format(row_len_selector)
        )

    # Use depths corresponding to that declared in the rows which had the
    # number of entries used.
    if row_len_selector == "median":
        d_start = np.median(row_depth_starts[row_lengths == n_depth_use])
        d_stop = np.median(row_depth_ends[row_lengths == n_depth_use])
    else:
        d_start = utils.mode(row_depth_starts[row_lengths == n_depth_use])
        d_stop = utils.mode(row_depth_ends[row_lengths == n_depth_use])
    depths = np.linspace(d_start, d_stop, n_depth_use)

    # Interpolate depths to get a consistent sampling grid
    interp_kwargs = dict(nan_threshold=0.3, assume_sorted=True)
    for i_entry, (nd, d0, d1) in enumerate(
        zip(row_lengths, row_depth_starts, row_depth_ends)
    ):
        if d0 < d1:
            data[i_entry, :n_depth_use] = utils.interp1d_preserve_nan(
                np.linspace(d0, d1, nd),
                data[i_entry, :nd],
                depths,
                **interp_kwargs,
            )
        else:
            data[i_entry, :n_depth_use] = utils.interp1d_preserve_nan(
                np.linspace(d1, d0, nd),
                data[i_entry, :nd][::-1],
                depths,
                **interp_kwargs,
            )

    # Crop the data down to size
    data = data[:, :n_depth_use]

    return timestamps, depths, data


def evl_reader(fname):
    """
    EVL file reader

    Parameters
    ----------
    fname : str
        Path to .evl file.

    Returns
    -------
    generator
        A generator which yields the timestamp (in seconds), depth (in
        metres), and status (int) for each entry. Note that the timestamp is
        not corrected for timezone (so make sure your timezones are internally
        consistent).
    """
    with open(fname, "r") as hf:
        continuance = True
        for i_row, row in enumerate(csv.reader(hf, delimiter=" ")):
            if i_row == 0:
                continue
            if len(row) < 4:
                if not continuance:
                    raise ValueError("Trying to skip data after parsing began")
                continue
            continuance = False

            timestamp = datetime.datetime.strptime(
                row[0] + "T" + row[1],
                "%Y%m%dT%H%M%S%f",
            ).timestamp()

            if len(row[2]) > 0:
                raise ValueError("row[2] was non-empty: {}".format(row[2]))

            yield timestamp, float(row[3]), int(row[4])


def evl_loader(fname, special_to_nan=True, return_status=False):
    """
    EVL file loader

    Parameters
    ----------
    fname : str
        Path to .evl file.
    special_to_nan : bool, optional
        Whether to replace the special value, `-10000.99`, which indicates no
        depth value, with NaN.
        https://support.echoview.com/WebHelp/Reference/File_formats/Export_file_formats/Special_Export_Values.htm

    Returns
    -------
    numpy.ndarray of floats
        Timestamps, in seconds.
    numpy.ndarary of floats
        Depth, in metres.
    numpy.ndarary of ints, optional
        Status codes.
    """
    timestamps = []
    values = []
    statuses = []
    for timestamp, value, status in evl_reader(fname):
        timestamps.append(timestamp)
        values.append(value)
        statuses.append(status)
    timestamps = np.array(timestamps)
    values = np.array(values)
    statuses = np.array(statuses)
    if special_to_nan:
        # Replace the special value -10000.99 with NaN
        # https://support.echoview.com/WebHelp/Reference/File_formats/Export_file_formats/Special_Export_Values.htm
        values[np.isclose(values, -10000.99)] = np.nan
    if return_status:
        return timestamps, values, statuses
    return timestamps, values


def timestamp2evdtstr(timestamp):
    """
    Converts a timestamp into an Echoview-compatible datetime string, in the
    format "CCYYMMDD HHmmSSssss", where:

    | CC: century
    | YY: year
    | MM: month
    | DD: day
    | HH: hour
    | mm: minute
    | SS: second
    | ssss: 0.1 milliseconds

    Parameters
    ----------
    timestamp : float
        Number of seconds since Unix epoch.

    Returns
    -------
    datetimestring : str
        Datetime string in the Echoview-compatible format
        "CCYYMMDD HHmmSSssss".
    """
    # Datetime must be in the format CCYYMMDD HHmmSSssss
    # where ssss = 0.1 milliseconds.
    # We have to manually determine the number of "0.1 milliseconds"
    # from the microsecond component.
    dt = datetime.datetime.fromtimestamp(timestamp)
    return "{}{:04d}".format(dt.strftime("%Y%m%d %H%M%S"), round(dt.microsecond / 100))


def evl_writer(fname, timestamps, depths, status=1, line_ending="\r\n", pad=False):
    r"""
    EVL file writer

    Parameters
    ----------
    fname : str
        Destination of output file.
    timestamps : array_like
        Timestamps for each node in the line.
    depths : array_like
        Depths (in meters) for each node in the line.
    status : 0, 1, 2, or 3; optional
        Status for the line.

        - `0` : none
        - `1` : unverified
        - `2` : bad
        - `3` : good

        Default is `1` (unverified). For more details on line status, see
        https://support.echoview.com/WebHelp/Using_Echoview/Echogram/Lines/About_Line_Status.htm
    pad : bool, optional
        Whether to pad the line with an extra datapoint half a pixel before the
        first and after the last given timestamp. Default is `False`.
    line_ending : str, optional
        Line ending. Default is `"\r\n"` the standard line ending on Windows/DOS,
        as per the specification for the file format.
        https://support.echoview.com/WebHelp/Using_Echoview/Exporting/Exporting_data/Exporting_line_data.htm
        Set to `"\n"` to get Unix-style line endings instead.

    Notes
    -----
    For more details on the format specification, see
    https://support.echoview.com/WebHelp/Using_Echoview/Exporting/Exporting_data/Exporting_line_data.htm#Line_definition_file_format
    """
    if len(timestamps) != len(depths):
        raise ValueError(
            "Number of timestamps ({}) and depths ({}) are not equal".format(
                len(timestamps), len(depths)
            )
        )
    if pad and len(timestamps) > 1:
        timestamps = timestamps[:]
        timestamps = np.r_[
            timestamps[0] - (timestamps[1] - timestamps[0]) / 2,
            timestamps,
            timestamps[-1] + (timestamps[-2] - timestamps[-1]) / 2,
        ]
        depths = np.r_[depths[0], depths, depths[-1]]
    # The file object will automatically replace \n with our chosen line ending
    with open(fname, "w+", encoding="utf-8-sig", newline=line_ending) as hf:
        # Write header
        hf.write("EVBD 3 10.0.270.37090" + "\n")
        n_row = len(depths)
        hf.write(str(n_row) + "\n")
        # Write each row
        for i_row, (timestamp, depth) in enumerate(zip(timestamps, depths)):
            # Datetime must be in the format CCYYMMDD HHmmSSssss
            # where ssss = 0.1 milliseconds.
            # We have to manually determine the number of "0.1 milliseconds"
            # from the microsecond component.
            dt = datetime.datetime.fromtimestamp(timestamp)
            hf.write("{}  {} {} \n".format(timestamp2evdtstr(timestamp), depth, status))


def evr_writer(
    fname,
    rectangles=[],
    contours=[],
    common_notes="",
    default_region_type=0,
    line_ending="\r\n",
):
    r"""
    EVR file writer.

    Writes regions to an Echoview region file.

    Parameters
    ----------
    fname : str
        Destination of output file.
    rectangles : list of dictionaries, optional
        Rectangle region definitions. Default is an empty list. Each rectangle
        region must implement fields `"depths"` and `"timestamps"`, which
        indicate the extent of the rectangle. Optionally, `"creation_type"`,
        `"region_name"`, `"region_type"`, and `"notes"` may be set.
        If these are not given, the default creation_type is 4 and region_type
        is set by `default_region_type`.
    contours : list of dictionaries
        Contour region definitions. Default is an empty list. Each contour
        region must implement a `"points"` field containing a :class:`numpy.ndarray`
        shaped `(n, 2)` defining the co-ordinates of nodes along the (open)
        contour in units of timestamp and depth. Optionally, `"creation_type"`,
        `"region_name"`, `"region_type"`, and `"notes"` may be set.
        If these are not given, the default creation_type is 2 and region_type
        is set by `default_region_type`.
    common_notes : str, optional
        Notes to include for every region. Default is `""`, an empty string.
    default_region_type : int, optional
        The region type to use for rectangles and contours which do not define
        a `"region_type"` field. Possible region types are

        - `0` : bad (no data)
        - `1` : analysis
        - `2` : marker
        - `3` : fishtracks
        - `4` : bad (empty water)

        Default is `0`.
    line_ending : str, optional
        Line ending. Default is `"\r\n"` the standard line ending on Windows/DOS,
        as per the specification for the file format.
        https://support.echoview.com/WebHelp/Using_Echoview/Exporting/Exporting_data/Exporting_line_data.htm
        Set to `"\n"` to get Unix-style line endings instead.

    Notes
    -----
    For more details on the format specification, see:
    https://support.echoview.com/WebHelp/Reference/File_formats/Export_file_formats/2D_Region_definition_file_format.htm
    """
    # Remove leading/trailing new lines, since we will join with our own line ending
    common_notes = common_notes.strip("\r\n")
    # Standardize line endings to be \n, regardless of input
    common_notes = common_notes.replace("\r\n", "\n").replace("\r", "\n")
    if len(common_notes) == 0:
        n_lines_common_notes = 0
    else:
        n_lines_common_notes = 1 + common_notes.count(line_ending)
    n_regions = len(rectangles) + len(contours)
    i_region = 0
    # The file object will automatically replace \n with our chosen line ending
    with open(fname, "w+", encoding="utf-8-sig", newline=line_ending) as hf:
        # Write header
        hf.write("EVRG 7 10.0.283.37689" + "\n")
        hf.write(str(n_regions) + "\n")

        # Write each rectangle
        for region in rectangles:
            # Regions are indexed from 1, so increment the counter first
            i_region += 1
            hf.write("\n")  # Blank line separates regions
            # Determine extent of rectangle
            left = timestamp2evdtstr(np.min(region["timestamps"]))
            right = timestamp2evdtstr(np.max(region["timestamps"]))
            top = np.min(region["depths"])
            bottom = np.max(region["depths"])
            # Region header
            hf.write(
                "13 4 {i} 0 {type} -1 1 {left}  {top} {right}  {bottom}".format(
                    i=i_region,
                    type=region.get("creation_type", 4),
                    left=left,
                    right=right,
                    top=top,
                    bottom=bottom,
                )
                + "\n"
            )
            # Notes
            notes = region.get("notes", "")
            if len(notes) == 0:
                notes = common_notes
                n_lines_notes = n_lines_common_notes
            else:
                notes = notes.strip("\n")
                if len(common_notes) > 0:
                    notes += "\n" + common_notes
                n_lines_notes = 1 + notes.count("\n")
            hf.write(str(n_lines_notes) + "\n")  # Number of lines of notes
            if len(notes) > 0:
                hf.write(notes + "\n")
            # Detection settings
            hf.write("0" + "\n")  # Number of lines of detection settings
            # Region classification string
            hf.write("Unclassified regions" + "\n")
            # The points defining the region itself
            hf.write(
                "{left} {top} {left} {bottom} {right} {bottom} {right} {top} ".format(
                    left=left,
                    right=right,
                    top=top,
                    bottom=bottom,
                )  # Terminates with a space, not a new line
            )
            # Region type
            hf.write(str(region.get("region_type", default_region_type)) + "\n")
            # Region name
            hf.write(
                str(region.get("region_name", "Region {}".format(i_region))) + "\n"
            )

        # Write each contour
        for region in contours:
            # Regions are indexed from 1, so increment the counter first
            i_region += 1
            hf.write("\n")  # Blank line separates regions
            # Header line
            hf.write(
                "13 {n} {i} 0 {type} -1 1 {left}  {top} {right}  {bottom}".format(
                    n=region["points"].shape[0],
                    i=i_region,
                    type=region.get("creation_type", 2),
                    left=timestamp2evdtstr(np.min(region["points"][:, 0])),
                    right=timestamp2evdtstr(np.max(region["points"][:, 0])),
                    top=np.min(region["points"][:, 1]),
                    bottom=np.max(region["points"][:, 1]),
                )
                + "\n"
            )
            # Notes
            notes = region.get("notes", "")
            if len(notes) == 0:
                notes = common_notes
                n_lines_notes = n_lines_common_notes
            else:
                notes = notes.strip("\n")
                if len(common_notes) > 0:
                    notes += "\n" + common_notes
                n_lines_notes = 1 + notes.count("\n")
            hf.write(str(n_lines_notes) + "\n")  # Number of lines of notes
            if len(notes) > 0:
                hf.write(notes + "\n")
            # Detection settings
            hf.write("0" + "\n")  # Number of lines of detection settings
            # Region classification string
            hf.write("Unclassified regions" + "\n")
            # The region itself
            for point in region["points"]:
                hf.write("{} {} ".format(timestamp2evdtstr(point[0]), point[1]))
            # Region type
            hf.write(str(region.get("region_type", default_region_type)) + "\n")
            # Region name
            hf.write(
                str(region.get("region_name", "Region {}".format(i_region))) + "\n"
            )


def write_transect_regions(
    fname,
    transect,
    depth_range=None,
    passive_key="is_passive",
    removed_key="is_removed",
    patches_key="mask_patches",
    collate_passive_length=0,
    collate_removed_length=0,
    minimum_passive_length=0,
    minimum_removed_length=0,
    minimum_patch_area=0,
    name_suffix="",
    common_notes="",
    line_ending="\r\n",
    verbose=0,
    verbose_indent=0,
):
    r"""
    Convert a transect dictionary to a set of regions and write as an EVR file.

    Parameters
    ----------
    fname : str
        Destination of output file.
    transect : dict
        Transect dictionary.
    depth_range : array_like or None, optional
        The minimum and maximum depth extents (in any order) of the passive and
        removed block regions. If this is `None` (default), the minimum and
        maximum of `transect["depths"]` is used.
    passive_key : str, optional
        Field name to use for passive data identification. Default is
        `"is_passive"`.
    removed_key : str, optional
        Field name to use for removed blocks. Default is `"is_removed"`.
    patches_key : str, optional
        Field name to use for the mask of patch regions. Default is
        `"mask_patches"`.
    collate_passive_length : int, optional
        Maximum distance (in indices) over which passive regions should be
        merged together, closing small gaps between them. Default is `0`.
    collate_removed_length : int, optional
        Maximum distance (in indices) over which removed blocks should be
        merged together, closing small gaps between them. Default is `0`.
    minimum_passive_length : int, optional
        Minimum length (in indices) a passive region must have to be included
        in the output. Set to -1 to omit all passive regions from the output.
        Default is `0`.
    minimum_removed_length : int, optional
        Minimum length (in indices) a removed block must have to be included in
        the output. Set to -1 to omit all removed regions from the output.
        Default is `0`.
    minimum_patch_area : float, optional
        Minimum amount of area (in input pixel space) that a patch must occupy
        in order to be included in the output. Set to `0` to include all
        patches, no matter their area. Set to `-1` to omit all patches.
        Default is `0`.
    name_suffix : str, optional
        Suffix to append to variable names. Default is `""`, an empty string.
    common_notes : str, optional
        Notes to include for every region. Default is `""`, an empty string.
    line_ending : str, optional
        Line ending. Default is `"\r\n"` the standard line ending on Windows/DOS,
        as per the specification for the file format,
        https://support.echoview.com/WebHelp/Using_Echoview/Exporting/Exporting_data/Exporting_line_data.htm
        Set to `"\n"` to get Unix-style line endings instead.
    verbose : int, optional
        Verbosity level. Default is `0`.
    verbose_indent : int, optional
        Level of indentation (number of preceding spaces) before verbosity
        messages. Default is `0`.
    """
    if depth_range is None:
        depth_range = transect["depths"]
    depth_range = [np.min(depth_range), np.max(depth_range)]

    rectangles = []
    contours = []
    # Regions around each period of passive data
    key = passive_key
    if key not in transect:
        key = "p_" + key
    if key not in transect:
        raise ValueError("Key {} and {} not found in transect.".format(key[2:], key))
    is_passive = transect[key] > 0.5
    is_passive = ~utils.squash_gaps(~is_passive, collate_passive_length)
    passive_starts, passive_ends = utils.get_indicator_onoffsets(is_passive)
    i_passive = 1
    n_passive_skipped = 0
    for start_index, end_index in zip(passive_starts, passive_ends):
        start_index -= 0.5
        end_index += 0.5
        if minimum_passive_length == -1:
            # No passive regions
            break
        if end_index - start_index <= minimum_passive_length:
            n_passive_skipped += 1
            continue
        region = {}
        region["region_name"] = "Passive{} {}".format(name_suffix, i_passive)
        region["creation_type"] = 4
        region["region_type"] = 0
        region["depths"] = depth_range
        region["timestamps"] = scipy.interpolate.interp1d(
            np.arange(len(transect["timestamps"])),
            transect["timestamps"],
            fill_value="extrapolate",
        )([start_index, end_index])
        region["notes"] = textwrap.dedent(
            """
            Passive data
            Length in pixels: {}
            Duration in seconds: {}
            """.format(
                end_index - start_index,
                region["timestamps"][1] - region["timestamps"][0],
            )
        )
        rectangles.append(region)
        i_passive += 1
    # Regions around each period of removed data
    key = removed_key
    if key not in transect:
        key = "p_" + key
    if key not in transect:
        raise ValueError("Key {} and {} not found in transect.".format(key[2:], key))
    is_removed = transect[key] > 0.5
    is_removed = ~utils.squash_gaps(~is_removed, collate_removed_length)
    removed_starts, removed_ends = utils.get_indicator_onoffsets(is_removed)
    i_removed = 1
    n_removed_skipped = 0
    for start_index, end_index in zip(removed_starts, removed_ends):
        start_index -= 0.5
        end_index += 0.5
        if minimum_removed_length == -1:
            # No passive regions
            break
        if end_index - start_index <= minimum_removed_length:
            n_removed_skipped += 1
            continue
        region = {}
        region["region_name"] = "Removed block{} {}".format(name_suffix, i_removed)
        region["creation_type"] = 4
        region["region_type"] = 0
        region["depths"] = depth_range
        region["timestamps"] = scipy.interpolate.interp1d(
            np.arange(len(transect["timestamps"])),
            transect["timestamps"],
            fill_value="extrapolate",
        )([start_index, end_index])
        region["notes"] = textwrap.dedent(
            """
            Removed data block
            Length in pixels: {}
            Duration in seconds: {}
            """.format(
                end_index - start_index,
                region["timestamps"][1] - region["timestamps"][0],
            )
        )
        rectangles.append(region)
        i_removed += 1
    # Contours around each removed patch
    if patches_key not in transect:
        raise ValueError("Key {} not found in transect.".format(patches_key))
    patches = transect[patches_key]
    patches = scipy.ndimage.binary_fill_holes(patches > 0.5)
    contours_coords = skimage.measure.find_contours(patches, 0.5)
    contour_dicts = []
    i_contour = 1
    n_contour_skipped = 0
    for contour in contours_coords:
        if minimum_patch_area == -1:
            # No patches
            break
        area = utils.integrate_area_of_contour(
            contour[:, 0], contour[:, 1], closed=False
        )
        if area < minimum_patch_area:
            n_contour_skipped += 1
            continue
        region = {}
        region["region_name"] = "Removed patch{} {}".format(name_suffix, i_contour)
        region["creation_type"] = 2
        region["region_type"] = 0
        x = scipy.interpolate.interp1d(
            np.arange(len(transect["timestamps"])),
            transect["timestamps"],
            fill_value="extrapolate",
        )(contour[:, 0])
        y = scipy.interpolate.interp1d(
            np.arange(len(transect["depths"])),
            transect["depths"],
            fill_value="extrapolate",
        )(contour[:, 1])
        region["points"] = np.stack([x, y], axis=-1)
        region["notes"] = textwrap.dedent(
            """
            Removed patch
            Area in pixels: {}
            Area in meter-seconds: {}
            """.format(
                area, utils.integrate_area_of_contour(x, y, closed=False)
            )
        )
        contour_dicts.append(region)
        i_contour += 1
    if verbose >= 1:
        print(
            " " * verbose_indent + "Outputting {} region{}:"
            " {} passive, {} removed blocks, {} removed patches".format(
                len(rectangles) + len(contour_dicts),
                "" if len(rectangles) + len(contour_dicts) == 1 else "s",
                i_passive - 1,
                i_removed - 1,
                i_contour - 1,
            )
        )
        n_skipped = n_passive_skipped + n_removed_skipped + n_contour_skipped
        if n_skipped > 0:
            print(
                " " * verbose_indent
                + style.skip_fmt(
                    "There {} {} skipped (too small) region{}:"
                    " {} passive, {} removed blocks, {} removed patches".format(
                        "was" if n_skipped == 1 else "were",
                        n_skipped,
                        "" if n_skipped == 1 else "s",
                        n_passive_skipped,
                        n_removed_skipped,
                        n_contour_skipped,
                    )
                )
            )

    # Write the output
    return evr_writer(
        fname,
        rectangles=rectangles,
        contours=contour_dicts,
        common_notes=common_notes,
        line_ending=line_ending,
    )


def load_transect_data(transect_pth, dataset="mobile", root_data_dir=ROOT_DATA_DIR):
    """
    Load all data for one transect.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding `"_Sv_raw.csv"`.
    dataset : str, optional
        Name of dataset. Default is `"mobile"`.
    root_data_dir : str
        Path to root directory where data is located.

    Returns
    -------
    timestamps : numpy.ndarray
        Timestamps (in seconds since Unix epoch), with each entry
        corresponding to each row in the `signals` data.
    depths : numpy.ndarray
        Depths from the surface (in metres), with each entry corresponding
        to each column in the `signals` data.
    signals : numpy.ndarray
        Echogram Sv data, shaped (num_timestamps, num_depths).
    turbulence : numpy.ndarray
        Depth of turbulence line, shaped (num_timestamps, ).
    bottom : numpy.ndarray
        Depth of bottom line, shaped (num_timestamps, ).
    """
    dirname = os.path.join(root_data_dir, dataset)
    raw_fname = os.path.join(dirname, transect_pth + "_Sv_raw.csv")
    bottom_fname = os.path.join(dirname, transect_pth + "_bottom.evl")
    turbulence_fname = os.path.join(dirname, transect_pth + "_turbulence.evl")

    timestamps, depths, signals = transect_loader(raw_fname)
    t_bottom, d_bottom = evl_loader(bottom_fname)
    t_turbulence, d_turbulence = evl_loader(turbulence_fname)

    return (
        timestamps,
        depths,
        signals,
        np.interp(timestamps, t_turbulence, d_turbulence),
        np.interp(timestamps, t_bottom, d_bottom),
    )


def get_partition_data(
    partition,
    dataset="mobile",
    partitioning_version="firstpass",
    root_data_dir=ROOT_DATA_DIR,
):
    """
    Loads partition metadata.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding `"_Sv_raw.csv"`.
    dataset : str, optional
        Name of dataset. Default is `"mobile"`.
    partitioning_version : str, optional
        Name of partitioning method.
    root_data_dir : str
        Path to root directory where data is located.

    Returns
    -------
    pandas.DataFrame
        Metadata for all transects in the partition. Each row is a single
        sample.
    """
    dirname = os.path.join(root_data_dir, dataset, "sets", partitioning_version)
    fname_partition = os.path.join(dirname, partition + ".txt")
    fname_header = os.path.join(dirname, "header" + ".txt")

    with open(fname_header, "r") as hf:
        for row in csv.reader(hf):
            header = [entry.strip() for entry in row]
            break

    df = pd.read_csv(fname_partition, header=None, names=header)
    return df


def remove_trailing_slash(s):
    """
    Remove trailing forward slashes from a string.

    Parameters
    ----------
    s : str
        String representing a path, possibly with trailing slashes.

    Returns
    -------
    str
        Same as `s`, but without trailing forward slashes.
    """
    while s[-1] == "/" or s[-1] == os.path.sep:
        s = s[:-1]
    return s


def list_from_file(fname):
    """
    Get a list from a file.

    Parameters
    ----------
    fname : str
        Path to file.

    Returns
    -------
    list
        Contents of the file, one line per entry in the list. Trailing
        whitespace is removed from each end of each line.
    """
    with open(fname, "r") as hf:
        contents = hf.readlines()
    contents = [x.strip() for x in contents]
    return contents


def get_partition_list(
    partition,
    dataset="mobile",
    full_path=False,
    partitioning_version="firstpass",
    root_data_dir=ROOT_DATA_DIR,
    sharded=False,
):
    """
    Get a list of transects in a single partition.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding `"_Sv_raw.csv"`.
    dataset : str, optional
        Name of dataset. Default is `"mobile"`.
    full_path : bool, optional
        Whether to return the full path to the sample. If `False`, only the
        relative path (from the dataset directory) is returned.
        Default is `False`.
    partitioning_version : str, optional
        Name of partitioning method.
    root_data_dir : str, optional
        Path to root directory where data is located.
    sharded : bool, optional
        Whether to return path to sharded version of data. Default is `False`.

    Returns
    -------
    list
        Path for each sample in the partition.
    """
    if dataset == "mobile":
        df = get_partition_data(
            partition,
            dataset=dataset,
            partitioning_version=partitioning_version,
            root_data_dir=root_data_dir,
        )
        fnames = df["Filename"]
        fnames = [os.path.join(f.split("_")[0], f.strip()) for f in fnames]
    else:
        partition_file = os.path.join(
            root_data_dir,
            dataset,
            "sets",
            partitioning_version,
            partition + ".txt",
        )
        fnames = list_from_file(partition_file)

    fnames = [f.replace("_Sv_raw.csv", "") for f in fnames]
    if full_path and sharded:
        root_data_dir = remove_trailing_slash(root_data_dir)
        fnames = [os.path.join(root_data_dir + "_sharded", dataset, f) for f in fnames]
    elif full_path:
        fnames = [os.path.join(root_data_dir, dataset, f) for f in fnames]
    return fnames
