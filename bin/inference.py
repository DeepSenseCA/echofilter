#!/usr/bin/env python


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
