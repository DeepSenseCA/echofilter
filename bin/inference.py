#!/usr/bin/env python

import torch.jit


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


torch.jit.script_method = script_method
torch.jit.script = script


if __name__ == "__main__":
    import echofilter.inference

    echofilter.inference.main()
