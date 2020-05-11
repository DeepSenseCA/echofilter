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
        Extension which files within directories must bear to be included.
        Explicitly given files are always used.

    Yields
    ------
    str
        Paths to explicitly given files and files within directories with
        extension `extension`.
    '''
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
                if extension is None or os.path.splitext(filename)[1][1:] == extension:
                    yield rel_file
