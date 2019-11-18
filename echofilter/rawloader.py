from collections import OrderedDict
import csv
import datetime
import os

import numpy as np


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


def transect_loader(fname, skip_lines=1):
    '''
    Loads an entire survey CSV.

    Parameters
    ----------
    fname : str
        Path to survey CSV file.
    skip_lines : int, optional
        Number of initial entries to skip. Default is 1.

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
        data[i_entry, :] = row
        timestamps[i_entry] = datetime.datetime.strptime(
            '{}T{}.{:6d}'.format(
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
