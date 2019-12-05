from collections import OrderedDict
import csv
import datetime
import os

import numpy as np
import pandas as pd


ROOT_DATA_DIR = '/data/dsforce'

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


def transect_loader(fname, skip_lines=1, warn_row_overflow=True):
    '''
    Loads an entire survey CSV.

    Parameters
    ----------
    fname : str
        Path to survey CSV file.
    skip_lines : int, optional
        Number of initial entries to skip. Default is 1.
    warn_row_overflow : bool, optional
        Whether to print a warning message if the number of elements in a
        row exceeds the expected number. Default is `True`. Overflowing
        datapoints are dropped.

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

    for i_line, (meta, row) in enumerate(transect_reader(fname)):
        if i_line < skip_lines:
            continue
        i_entry = i_line - skip_lines
        if warn_row_overflow and len(row) > n_depths:
            print(
                'Row {} of {} exceeds expected n_depths of {} with {}'
                .format(i_line, fname, n_depths, len(row))
            )
        if len(row) < n_depths:
            if warn_row_overflow:
                print(
                    'Row {} of {} shorter than expected n_depths of {} with {}'
                    .format(i_line, fname, n_depths, len(row))
                )
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


def load_transect_data(
        transect_pth,
        dataset='surveyExports',
        root_data_dir=ROOT_DATA_DIR
        ):
    '''
    Load all data for one transect.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding '_Sv_raw.csv'.
    dataset : str, optional
        Name of dataset. Default is `'surveyExports'`.
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
        dataset='surveyExports',
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
        Name of dataset. Default is `'surveyExports'`.
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


def get_partition_list(
        partition,
        dataset='surveyExports',
        full_path=False,
        partitioning_version='firstpass',
        root_data_dir=ROOT_DATA_DIR,
    ):
    '''
    Get a list of transects in a single partition.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding '_Sv_raw.csv'.
    dataset : str, optional
        Name of dataset. Default is `'surveyExports'`.
    full_path : bool, optional
        Whether to return the full path to the sample. If `False`, only the
        relative path (from the dataset directory) is returned.
        Default is `False`.
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
    df = get_partition_data(
        partition,
        dataset=dataset,
        partitioning_version=partitioning_version,
        root_data_dir=root_data_dir,
    )
    fnames = df['Filename']
    fnames = [os.path.join(f.split('_')[0], f.strip().replace('_Sv_raw.csv', '')) for f in fnames]
    if full_path:
        fnames = [os.path.join(root_data_dir, dataset, f) for f in fnames]
    return fnames
