#!/usr/bin/env python

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


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


if __name__ == "__main__":
    try:
        import warnings

        warnings.filterwarnings(
            "ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning
        )

        import torch.jit
        from echofilter.ui.inference_cli import main

        torch.jit.script_method = script_method
        torch.jit.script = script

        main()

    except KeyboardInterrupt as err:
        import sys

        # Don't show stack traceback when KeyboardInterrupt is given.
        print("Interrupted by user during: {}".format(" ".join(sys.argv)))
        try:
            sys.exit(1)
        except SystemExit:
            import os

            os._exit(1)
