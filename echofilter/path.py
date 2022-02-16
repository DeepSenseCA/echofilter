"""
Path utilities.
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

import os
import sys


def check_if_windows():
    """
    Check if the operating system is Windows.

    Returns
    -------
    bool
        Whether the OS is Windows.
    """
    return sys.platform.startswith("win")


def parse_files_in_folders(files_or_folders, source_dir, extension, recursive=True):
    """
    Walk through folders and find suitable files.

    Parameters
    ----------
    files_or_folders : iterable
        List of files and folders.
    source_dir : str
        Root directory within which elements of `files_or_folders` may
        be found.
    extension : str or Collection
        Extension (or list of extensions) which files within directories must
        bear to be included, without leading `'.'`, for instance `'.csv'`.
        Note that explicitly given files are always used.
    recursive : bool, optional
        Whether to walk through the tree of files in a subfolders of a
        directory input. If `False`, only files in the folder itself and
        not its child folders will be included.

    Yields
    ------
    str
        Paths to explicitly given files and files within directories with
        extension `extension`.
    """
    if extension is None or not isinstance(extension, str):
        extensions = extension
    else:
        extensions = {extension}
    if extensions is not None:
        extensions = {ext.lower() for ext in extensions}
    for path in files_or_folders:
        if check_if_windows():
            # Remove trailing slashes on Windows, because isdir("some\path\")
            # fails even though isdir("some\path") would succeed.
            path = path.rstrip("\\")

        if os.path.isfile(path) or os.path.isfile(os.path.join(source_dir, path)):
            yield path
            continue
        elif os.path.isdir(path):
            folder = path
        elif os.path.isdir(os.path.join(source_dir, path)):
            folder = os.path.join(source_dir, path)
        else:
            raise EnvironmentError("Missing file or directory: {}".format(path))

        if recursive:
            full_filenames = []
            for dirpath, dirnames, fnames in os.walk(folder):
                for fname in fnames:
                    full_filenames.append(os.path.join(dirpath, fname))
        else:
            full_filenames = (
                os.path.join(folder, filename) for filename in os.listdir(folder)
            )

        for filename in full_filenames:
            if not os.path.isfile(filename):
                continue
            ext = os.path.splitext(filename)[1]
            if extensions is None or (len(ext) > 0 and ext[1:].lower() in extensions):
                yield filename


def determine_file_path(fname, source_dir):
    """
    Determine the path to use to an input file.

    Parameters
    ----------
    fname : str
        Path to an input file. Either an absolute path, or a path relative to
        to `source_dir`, or a path relative to the working directory.
    source_dir : str
        Path to a directory where the file bearing name `fname` is expected to
        be located.

    Returns
    -------
    str
        Path to where file can be found, either absolute or relative.
    """
    # Check what the full path should be
    if os.path.isabs(fname) and os.path.isfile(fname):
        fname_full = fname
    elif os.path.isfile(os.path.join(source_dir, fname)):
        fname_full = os.path.join(source_dir, fname)
    elif os.path.isfile(fname):
        fname_full = fname
    else:
        raise EnvironmentError("Could not locate file {}".format(fname))

    return fname_full


def determine_destination(fname, fname_full, source_dir, output_dir):
    """
    Determine where destination should be placed for a file, preserving subtree
    paths.

    Parameters
    ----------
    fname : str
        Original input path.
    fname_full : str
        Path to file, either absolute or relative; possibly containing
        `source_dir`.
    source_dir : str
        Path to a directory where the file bearing name `fname` is expected to
        be located.
    output_dir : str
        Path to root output directory.

    Returns
    -------
    str
        Path to where file can be found, either absolute or relative.
    """
    # Determine where destination should be placed
    if output_dir is None or output_dir == "":
        return fname_full
    if os.path.isabs(fname):
        return os.path.join(output_dir, os.path.split(fname)[1])
    full_source_dir = os.path.join(os.path.abspath(source_dir), "")
    if os.path.abspath(fname).startswith(full_source_dir):
        return os.path.join(output_dir, os.path.abspath(fname)[len(full_source_dir) :])
    return os.path.join(output_dir, fname)
