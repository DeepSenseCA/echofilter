'''
Path utilities.
'''

import os


def parse_files_in_folders(files_or_folders, data_dir, extension):
    '''
    Walk through folders and find suitable files.

    Parameters
    ----------
    files_or_folders : iterable
        List of files and folders.
    data_dir : str
        Root directory within which elements of `files_or_folders` may
        be found.
    extension : str
        Extension which files within directories must bear to be included,
        without leading `'.'`, for instance `'.csv'`. Note that explicitly
        given files are always used.

    Yields
    ------
    str
        Paths to explicitly given files and files within directories with
        extension `extension`.
    '''
    if extension is not None:
        extension = extension.lower()
    for path in files_or_folders:
        if os.path.isfile(path) or os.path.isfile(os.path.join(data_dir, path)):
            yield path
            continue
        elif os.path.isdir(path):
            folder = path
        elif os.path.isdir(os.path.join(data_dir, path)):
            folder = os.path.join(data_dir, path)
        else:
            raise EnvironmentError('Missing file or directory: {}'.format(path))
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                rel_file = os.path.join(dirpath, filename)
                if not os.path.isfile(rel_file):
                    continue
                if extension is None or os.path.splitext(filename)[1][1:].lower() == extension:
                    yield rel_file


def determine_file_path(fname, data_dir):
    '''
    Determine the path to use to an input file.

    Parameters
    ----------
    fname : str
        Path to an input file. Either an absolute path, or a path relative to
        to `data_dir`, or a path relative to the working directory.
    data_dir : str
        Path to a directory where the file bearing name `fname` is expected to
        be located.

    Returns
    -------
    str
        Path to where file can be found, either absolute or relative.
    '''
    # Check what the full path should be
    if os.path.isabs(fname) and os.path.isfile(fname):
        fname_full = fname
    elif os.path.isfile(os.path.join(data_dir, fname)):
        fname_full = os.path.join(data_dir, fname)
    elif os.path.isfile(fname):
        fname_full = fname
    else:
        raise EnvironmentError('Could not locate file {}'.format(fname))

    return fname_full


def determine_destination(fname, fname_full, data_dir, output_dir):
    '''
    Determine where destination should be placed for a file, preserving subtree
    paths.

    Parameters
    ----------
    fname : str
        Original input path.
    fname_full : str
        Path to file, either absolute or relative; possibly containing
        `data_dir`.
    data_dir : str
        Path to a directory where the file bearing name `fname` is expected to
        be located.
    output_dir : str
        Path to root output directory.

    Returns
    -------
    str
        Path to where file can be found, either absolute or relative.
    '''
    # Determine where destination should be placed
    if output_dir is None or output_dir == '':
        destination = fname_full
    elif os.path.isabs(fname):
        destination = os.path.join(output_dir, os.path.split(fname)[1])
    elif os.path.abspath(fname).startswith(os.path.join(os.path.abspath(data_dir), '')):
        destination = os.path.join(
            output_dir,
            os.path.abspath(fname)[len(os.path.join(os.path.abspath(data_dir), '')):],
        )
    else:
        destination = os.path.join(output_dir, fname)

    return destination
