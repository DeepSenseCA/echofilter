'''
Input/Output handling for raw echoview files.
'''

from collections import OrderedDict
import csv
import datetime
import os

import numpy as np
import pandas as pd


ROOT_DATA_DIR = '/data/dsforce/surveyExports'

TRANSECT_FIELD_TYPES = {
    'Ping_index': int,
    'Distance_gps': float,
    'Distance_vl': float,
    'Ping_date': str,
    'Ping_time': str,
    'Ping_milliseconds': float,
    'Latitude': float,
    'Longitude': float,
    'Depth_start': float,
    'Depth_stop': float,
    'Range_start': float,
    'Range_stop': float,
    'Sample_count': int,
}


def transect_reader(fname):
    '''
    Creates a generator which iterates through a survey csv file.

    Parameters
    ----------
    fname: str
        Path to survey CSV file.

    Returns
    -------
    generator
        Yields a tupule of `(metadata, data)`, where metadata is a dict,
        and data is a `numpy.ndarray`. Each yield corresponds to a single
        row in the data. Every row (except for the header) is yielded.
    '''
    metadata_header = []
    with open(fname, 'r', encoding='utf-8-sig') as hf:
        for i_row, row in enumerate(csv.reader(hf)):
            row = [entry.strip() for entry in row]
            if i_row == 0:
                metadata_header = row
                continue;
            metadata = row[:len(metadata_header)]
            metadata_d = OrderedDict()
            for k, v in zip(metadata_header, metadata):
                if k in TRANSECT_FIELD_TYPES:
                    metadata_d[k] = TRANSECT_FIELD_TYPES[k](v)
                else:
                    metadata_d[k] = v
            data = np.array([float(x) for x in row[len(metadata_header):]])
            yield metadata_d, data


def count_lines(filename):
    '''
    Count the number of lines in a file.

    Credit: https://stackoverflow.com/a/27518377

    Parameters
    ----------
    filename : str
        Path to file.

    Returns
    int
        Number of lines in file.
    '''
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read  # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def transect_loader(fname, skip_lines=1, warn_row_overflow=1):
    '''
    Loads an entire survey CSV.

    Parameters
    ----------
    fname : str
        Path to survey CSV file.
    skip_lines : int, optional
        Number of initial entries to skip. Default is 1.
    warn_row_overflow : bool or int, optional
        Whether to print a warning message if the number of elements in a
        row exceeds the expected number. If this is an int, this is the number
        of times to display the warnings before they are supressed. If this
        is `True`, the number of outputs is unlimited. Default is `1`.

    Returns
    -------
    numpy.ndarray
        Timestamps for each row, in seconds. Note: not corrected for timezone
        (so make sure your timezones are internally consistent).
    numpy.ndarray
        Depth of each column, in metres.
    numpy.ndarray
        Survey signal (echo strength, units unknown).
    '''

    if warn_row_overflow is True:
        warn_row_overflow = np.inf

    # We remove one from the line count because of the header
    # which is excluded from output
    n_lines = count_lines(fname) - 1
    n_distances = 0
    depth_start = None
    depth_stop = None

    # Initialise output array
    for i_line, (meta, row) in enumerate(transect_reader(fname)):
        if i_line < skip_lines:
            continue
        n_depths = len(row)
        depth_start = meta['Depth_start']
        depth_stop = meta['Depth_stop']
        break

    data = np.empty((n_lines - skip_lines, n_depths))
    timestamps = np.empty((n_lines - skip_lines))
    depths = np.linspace(depth_start, depth_stop, n_depths)

    n_warn_overflow = 0
    n_warn_underflow = 0

    for i_line, (meta, row) in enumerate(transect_reader(fname)):
        if i_line < skip_lines:
            continue
        i_entry = i_line - skip_lines
        if len(row) > n_depths and n_warn_overflow < warn_row_overflow:
            print(
                'Row {} of {} exceeds expected n_depths of {} with {}'
                .format(i_line, fname, n_depths, len(row))
            )
            n_warn_overflow += 1
        if len(row) < n_depths:
            if n_warn_underflow < warn_row_overflow:
                print(
                    'Row {} of {} shorter than expected n_depths of {} with {}'
                    .format(i_line, fname, n_depths, len(row))
                )
                n_warn_underflow += 1
            data[i_entry, :] = np.nan
            data[i_entry, :len(row)] = row
        else:
            data[i_entry, :] = row[:n_depths]
        timestamps[i_entry] = datetime.datetime.strptime(
            '{}T{}.{:06d}'.format(
                meta['Ping_date'],
                meta['Ping_time'],
                int(1000 * float(meta['Ping_milliseconds'])),
            ),
            '%Y-%m-%dT%H:%M:%S.%f',
        ).timestamp()

    # Turn NaNs into NaNs (instead of extremely negative number)
    data[data < -1e6] = np.nan

    return timestamps, depths, data


def evl_reader(fname):
    '''
    EVL file reader

    Parameters
    ----------
    fname : str
        Path to .evl file.

    Returns
    -------
    generator
        A generator which yields the timestamp (in seconds) and depth (in
        metres) for each entry. Note that the timestamp is not corrected for
        timezone (so make sure your timezones are internally consistent).
    '''
    with open(fname, 'r') as hf:
        continuance = True
        for i_row, row in enumerate(csv.reader(hf, delimiter=' ')):
            if i_row == 0:
                continue
            if len(row) < 4:
                if not continuance:
                    raise ValueError('Trying to skip data after parsing began')
                continue
            continuance = False

            timestamp = datetime.datetime.strptime(
                row[0] + 'T' + row[1],
                '%Y%m%dT%H%M%S%f',
            ).timestamp()

            if len(row[2]) > 0:
                raise ValueError('row[2] was non-empty: {}'.format(row[2]))

            yield timestamp, float(row[3])


def evl_loader(fname):
    '''
    EVL file loader

    Parameters
    ----------
    fname : str
        Path to .evl file.

    Returns
    -------
    numpy.ndarray
        Timestamps, in seconds.
    numpy.ndarary
        Depth, in metres.
    '''
    timestamps = []
    values = []
    for timestamp, value in evl_reader(fname):
        timestamps.append(timestamp)
        values.append(value)
    return np.array(timestamps), np.array(values)


def evl_writer(fname, timestamps, depths, status=1):
    '''
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
            `0` : none
            `1` : unverified
            `2` : bad
            `3` : good
        Default is `1` (unverified). For more details on line status, see:
        https://support.echoview.com/WebHelp/Using_Echoview/Echogram/Lines/About_Line_Status.htm

    Notes
    -----
    For more details on the format specification, see:
    https://support.echoview.com/WebHelp/Using_Echoview/Exporting/Exporting_data/Exporting_line_data.htm#Line_definition_file_format
    '''
    with open(fname, 'w+') as hf:
        # Write header
        print('﻿EVBD 3 10.0.270.37090', file=hf)
        n_row = len(depths)
        print(n_row, file=hf)
        # Write each row
        for i_row, (timestamp, depth) in enumerate(zip(timestamps, depths)):
            print(
                '{}  {} {} '.format(
                    datetime.datetime.fromtimestamp(timestamp).strftime('%Y%m%d %H%M%S%f'),
                    depth,
                    0 if i_row == n_row - 1 else status,
                ),
                file=hf,
            )


def load_transect_data(
        transect_pth,
        dataset='mobile',
        root_data_dir=ROOT_DATA_DIR
        ):
    '''
    Load all data for one transect.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding '_Sv_raw.csv'.
    dataset : str, optional
        Name of dataset. Default is `'mobile'`.
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
    top : numpy.ndarray
        Depth of top line, shaped (num_timestamps, ).
    bottom : numpy.ndarray
        Depth of bottom line, shaped (num_timestamps, ).
    '''
    dirname = os.path.join(root_data_dir, dataset)
    raw_fname = os.path.join(dirname, transect_pth + '_Sv_raw.csv')
    bot_fname = os.path.join(dirname, transect_pth + '_bottom.evl')
    top_fname = os.path.join(dirname, transect_pth + '_turbulence.evl')

    timestamps, depths, signals = transect_loader(raw_fname)
    t_bot, d_bot = evl_loader(bot_fname)
    t_top, d_top = evl_loader(top_fname)

    return (
        timestamps,
        depths,
        signals,
        np.interp(timestamps, t_top, d_top),
        np.interp(timestamps, t_bot, d_bot),
    )


def get_partition_data(
        partition,
        dataset='mobile',
        partitioning_version='firstpass',
        root_data_dir=ROOT_DATA_DIR,
        ):
    '''
    Loads partition metadata.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding '_Sv_raw.csv'.
    dataset : str, optional
        Name of dataset. Default is `'mobile'`.
    partitioning_version : str, optional
        Name of partitioning method.
    root_data_dir : str
        Path to root directory where data is located.

    Returns
    -------
    pandas.DataFrame
        Metadata for all transects in the partition. Each row is a single
        sample.
    '''
    dirname = os.path.join(root_data_dir, dataset, 'sets', partitioning_version)
    fname_partition = os.path.join(dirname, partition + '.txt')
    fname_header = os.path.join(dirname, 'header' + '.txt')

    with open(fname_header, 'r') as hf:
        for row in csv.reader(hf):
            header = [entry.strip() for entry in row]
            break

    df = pd.read_csv(fname_partition, header=None, names=header)
    return df


def remove_trailing_slash(s):
    '''
    Remove trailing forward slashes from a string.

    Parameters
    ----------
    s : str
        String representing a path, possibly with trailing slashes.

    Returns
    -------
    str
        Same as `s`, but without trailing forward slashes.
    '''
    while s[-1] == '/':
        s = s[:-1]
    return s


def list_from_file(fname):
    '''
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
    '''
    with open(fname, 'r') as hf:
        contents = hf.readlines()
    contents = [x.strip() for x in contents]
    return contents


def get_partition_list(
        partition,
        dataset='mobile',
        full_path=False,
        partitioning_version='firstpass',
        root_data_dir=ROOT_DATA_DIR,
        sharded=False,
    ):
    '''
    Get a list of transects in a single partition.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding '_Sv_raw.csv'.
    dataset : str, optional
        Name of dataset. Default is `'mobile'`.
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
    '''
    if dataset == 'mobile':
        df = get_partition_data(
            partition,
            dataset=dataset,
            partitioning_version=partitioning_version,
            root_data_dir=root_data_dir,
        )
        fnames = df['Filename']
        fnames = [os.path.join(f.split('_')[0], f.strip()) for f in fnames]
    else:
        partition_file = os.path.join(
            root_data_dir,
            dataset,
            'sets',
            partitioning_version,
            partition + '.txt',
        )
        fnames = list_from_file(partition_file)

    fnames = [f.replace('_Sv_raw.csv', '') for f in fnames]
    if full_path and sharded:
        root_data_dir = remove_trailing_slash(root_data_dir)
        fnames = [os.path.join(root_data_dir + '_sharded', dataset, f) for f in fnames]
    elif full_path:
        fnames = [os.path.join(root_data_dir, dataset, f) for f in fnames]
    return fnames
