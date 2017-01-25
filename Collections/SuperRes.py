# SuperRes1(): Single Image Super Resolution
# SuperRes2(): Single Image Super Resolution with Bilateral filtering

# SuperRes(): Single Image Super Resolution with Bilateral filtering and user-defined resampling
"""Example of using nnedi3() as an upsampling filter:

import nnedi3_resample as nnrs
from functools import partial

input = ...
target_width = ...
target_height = ...
upsampleFilter = partial(nnrs.nnedi3_resample, target_width=target_width, target_height=target_height)
superResolution = SuperRes(input, target_width, target_width, upsampleFilter=upsampleFilter)

"""

# Appears to behave naturally when used to enhance textures during upsampling, though there would be lots of aliasing after filtering

# 16bit integer clip is required

def SuperRes1(lowRes, w, h, fltPass=3, **fmtc_args):
    def computeError(input):
        return core.std.MakeDiff(lowRes, core.fmtc.resample(input, lowRes.width, lowRes.height, **fmtc_args))

    highRes = core.fmtc.resample(lowRes, w, h, **fmtc_args)
    for i in range(fltPass):
        highRes = core.std.MergeDiff(highRes, core.fmtc.resample(computeError(highRes), w, h, **fmtc_args))
    return highRes

def SuperRes2(lowRes, w, h, fltPass=3, bilateral_args=dict(sigmaS=3.0, sigmaR=0.02), **fmtc_args):
    def computeError(input):
        return core.std.MakeDiff(lowRes, core.fmtc.resample(input, lowRes.width, lowRes.height, **fmtc_args))

    highRes = core.fmtc.resample(lowRes, w, h, **fmtc_args)
    for i in range(fltPass):
        highRes = core.std.MergeDiff(highRes, core.bilateral.Bilateral(core.fmtc.resample(computeError(highRes), w, h, **fmtc_args), ref=highRes, **bilateral_args))
    return highRes

def SuperRes(lowRes, w, h, fltPass=3, upsampleFilter=None, downsampleFilter=None, useBilateral=True, **bilateral_args):
    if upsampleFilter is None:
        def upsampleFilter(input):
            return core.fmtc.resample(input, w, h)
    if downsampleFilter is None:
        def downsampleFilter(input):
            return core.fmtc.resample(input, lowRes.width, lowRes.height)

    def computeError(input):
        return core.std.MakeDiff(lowRes, downsampleFilter(input))

    highRes = upsampleFilter(lowRes)
    for i in range(fltPass):
        diff = upsampleFilter(computeError(highRes))
        if useBilateral:
            diff = core.bilateral.Bilateral(diff, ref=highRes, **bilateral_args)
        highRes = core.std.MergeDiff(highRes, diff)
    return highRes