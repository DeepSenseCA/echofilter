#!/usr/bin/env python


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit

torch.jit.script_method = script_method
torch.jit.script = script

import echofilter.inference

if __name__ == "__main__":
    echofilter.inference.main()
