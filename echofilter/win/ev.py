"""
Echoview interface management.
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
import os

from .. import ui


__all__ = ["ECHOVIEW_COM_NAME", "maybe_open_echoview", "open_ev_file"]

ECHOVIEW_COM_NAME = "EchoviewCom.EvApplication"


@contextmanager
def maybe_open_echoview(
    app=None,
    do_open=True,
    minimize=False,
    hide="new",
):
    """
    If the current pointer to the Echoview is invalid, open an Echoview window.

    Parameters
    ----------
    app : COM object or None, optional
        Existing COM object to interface with Echoview.
    do_open : bool, optional
        If `False` (dry-run mode), we don't actually need Echoview open and so
        don't try to open it. In this case, `None` is yielded. Present so a
        context manager can be used even if the application isn't opened.
        Default is `True`, do open Echoview.
    minimize : bool, optional
        If `True`, the Echoview window being used will be minimized while the
        code runs. Default is `False`.
    hide : {"never", "new", "always"}, optional
        Whether to hide the Echoview window entirely. If `hide="new"`, the
        application is only hidden if it was created by this context, and not
        if it was already running. If `hide="always"`, the application is
        hidden even if it was already running. In the latter case, the window
        will be revealed again when leaving this context. Default is `"new"`.
    """
    if not do_open:
        yield None
        return

    if app is None:
        need_to_open = True
    else:
        try:
            # If we can check for a license, the COM handle is working
            is_licensed = app.IsLicensed()
            need_to_open = False
        except:
            need_to_open = True
    if not need_to_open:
        yield app
    else:
        # Need to import the context which actually opens COM windows
        from .manager import opencom

        with opencom(
            ECHOVIEW_COM_NAME,
            title="Echoview",
            title_pattern="Echoview.*",
            minimize=minimize,
            hide=hide,
        ) as app:
            yield app


@contextmanager
def open_ev_file(filename, app=None):
    """
    Open an EV file within a context.

    Parameters
    ----------
    filename : str
        Path to file to open.
    app : COM object or None, optional
        Existing COM object to interface with Echoview. If `None`, a new
        COM interface is created. If that requires opening a new instance
        of Echoview, it is hidden while the file is in use.
    """
    if not os.path.isfile(filename):
        raise EnvironmentError("Missing file: {}".format(filename))
    # Ensure the path is an absolute path
    filename = os.path.abspath(filename)
    with maybe_open_echoview(app, hide="new") as app:
        ev_file = app.OpenFile(filename)
        if ev_file is None or ev_file is "None" or ev_file is "NoneValue":
            raise EnvironmentError(
                "Could not open file '{}'."
                "\nIt appears that you already have a file open in Echoview."
                " {}Please close Echoview before running echofilter{}."
                " You can re-open Echoview once echofilter is running and"
                " continue your work without distrupting echofilter.".format(
                    filename,
                    ui.style.HighlightStyle.start,
                    ui.style.HighlightStyle.reset,
                )
            )
        try:
            yield ev_file
        finally:
            try:
                ev_file.Close()
            except:
                print(
                    ui.style.warning_fmt(
                        "Could not close Echoview file {}".format(filename)
                    )
                )
