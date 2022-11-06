"""
Window management for Windows.
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

from contextlib import contextmanager
import re

from .. import ui

try:
    import pywintypes
    import win32com.client
    import win32con
    import win32gui
except ImportError:
    from ..path import check_if_windows

    if check_if_windows():
        raise

    import warnings

    msg = "The Windows management module is only for Windows operating" " systems."
    with ui.style.warning_message(msg) as msg:
        warnings.warn(msg, category=RuntimeWarning)


__all__ = ["opencom", "WindowManager"]


class WindowManager:
    """
    Encapsulates calls to window management using the Windows api.

    Notes
    -----
    Based on: https://stackoverflow.com/a/2091530 and https://stackoverflow.com/a/4440622
    """

    def __init__(self, title=None, class_name=None, title_pattern=None):
        self.handle = None
        if title is not None:
            self.find_window(class_name=class_name, title=title)
        elif title_pattern is not None:
            self.find_window_regex(title_pattern)

    def find_window(self, class_name=None, title=None):
        """Find a window by its exact title"""
        handle = win32gui.FindWindow(class_name, title)
        if handle == 0:
            raise EnvironmentError(
                "Couldn't find window with class={}, title={}.".format(
                    class_name, title
                )
            )
        else:
            self.handle = handle

    def _window_enum_callback(self, hwnd, pattern):
        """Pass to win32gui.EnumWindows() to check all the opened windows"""
        if re.match(pattern, str(win32gui.GetWindowText(hwnd))) is not None:
            self.handle = hwnd

    def find_window_regex(self, pattern):
        """Find a window whose title matches a regular expression."""
        self.handle = None
        win32gui.EnumWindows(self._window_enum_callback, pattern)
        if self.handle is None:
            raise EnvironmentError(
                "Couldn't find a window with title matching pattern {}.".format(pattern)
            )

    def set_foreground(self):
        """Bring the window to the foreground."""
        win32gui.SetForegroundWindow(self.handle)

    def hide(self):
        """Hide the window."""
        win32gui.ShowWindow(self.handle, win32con.SW_HIDE)

    def show(self):
        """Show the window."""
        win32gui.ShowWindow(self.handle, win32con.SW_SHOW)


@contextmanager
def opencom(
    com_name,
    can_make_anew=False,
    title=None,
    title_pattern=None,
    minimize=False,
    hide="never",
):
    """
    Open a connection to an application with a COM object.

    The application may or may not be open before this context begins. If it
    was not already open, the application is closed when leaving the context.

    Parameters
    ----------
    com_name : str
        Name of COM object to dispatch.
    can_make_anew : bool, optional
        Whether arbitrarily many sessions of the COM object can be created, and
        if so whether they should be. Default is `False`, in which case the
        context manager will check to see if the application is already running
        before connecting to it. If it was already running, it will not be
        closed when this context closes.
    title : str, optional
        Exact title of window. If the title can not be determined exactly, use
        `title_pattern` instead.
    title_pattern : str, optional
        Regular expression for the window title.
    minimize : bool, optional
        If `True`, the application will be minimized while the code runs.
        Default is `False`.
    hide : {"never", "new", "always"}, optional
        Whether to hide the application window entirely. Default is `"never"`.
        If this is enabled, at least one of `title` and `title_pattern` must
        be specified.  If `hide="new"`, the application is only hidden if it
        was created by this context, and not if it was already running.
        If `hide="always"`, the application is hidden even if it was already
        running. In the latter case, the window will be revealed again when
        leaving this context.

    Yields
    ------
    win32com.gen_py
        Interface to COM object.
    """
    HIDE_OPTIONS = {"never", "new", "always"}
    if hide not in HIDE_OPTIONS:
        raise ValueError(
            "Unsupported hide value: {}. Must be one of {}".format(hide, HIDE_OPTIONS)
        )

    make_anew = can_make_anew
    if not make_anew:
        # Try to fetch existing session
        try:
            app = win32com.client.GetActiveObject(com_name)
            existing_session = True
        except pywintypes.com_error as err:
            # No existing session, make a new session
            make_anew = True
    if make_anew:
        # Create a new session
        existing_session = False
        app = win32com.client.Dispatch(com_name)

    was_minimized = False
    if minimize:
        # Tell the app to minimize itself
        try:
            app.Minimize()
            was_minimized = True
        except:
            print(ui.style.warning_fmt("Could not minimize {} window".format(com_name)))

    was_hidden = False
    if hide == "always" or (hide == "new" and not existing_session):
        # Find handle to the app based on the window title, and hide it
        winman = WindowManager()
        if title is None and title_pattern is None:
            raise ValueError(
                "One of the arguments title or title_pattern must be set in"
                " order to hide the window."
            )
        if title is not None:
            try:
                winman.find_window(title=title)
            except:
                pass
        if winman.handle is None and title_pattern is not None:
            try:
                winman.find_window_regex(title_pattern)
            except:
                pass
        if winman.handle is None:
            print(
                ui.style.warning_fmt(
                    "Could not hide {} window with title {}".format(
                        com_name, title if title_pattern is None else title_pattern
                    )
                )
            )
        else:
            winman.hide()
            was_hidden = True

    try:
        yield app
    finally:
        # As we leave the context, fix the state of the app to how it was
        # before we started
        command = ""
        try:
            if not existing_session:
                # If we opened it, tell the application to quit
                command = "exit"
                app.Quit()
            else:
                if was_minimized:
                    # Restore the window from being minimised
                    command = "restore"
                    app.Restore()
                if was_hidden:
                    # Show the window again
                    command = "unhide"
                    winman.show()
        except:
            # We'll get an error if the application was already closed, etc
            print(
                ui.style.warning_fmt(
                    "Could not {} the {} window with handle {}.".format(
                        command, com_name, app
                    )
                )
            )
