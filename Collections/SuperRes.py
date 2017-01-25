# SuperRes1(): Single Image Super Resolution
# SuperRes2(): Single Image Super Resolution with nnedi3 upsampling

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

# Main function
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


# Wrap functions
def SuperRes1(lowRes, w, h, fltPass=3, useBilateral=True, **fmtc_args):
    from functools import partial

    upsampleFilter = partial(core.fmtc.resample, w=w, h=h, **fmtc_args)
        
    downsampleFilter = partial(core.fmtc.resample, w=lowRes.width, h=lowRes.height, **fmtc_args)
    
    return SuperRes(lowRes, w, h, fltPass, upsampleFilter, downsampleFilter, useBilateral)

def SuperRes2(lowRes, w, h, fltPass=3, useBilateral=True, nnedi3_args=dict(), **fmtc_args):
    from functools import partial
    import nnedi3_resample as nnrs
    
    upsampleFilter = partial(nnrs.nnedi3_resample, target_width=w, target_height=h, **nnedi3_args)
    
    downsampleFilter = partial(core.fmtc.resample, w=lowRes.width, h=lowRes.height, **fmtc_args)
    
    return SuperRes(lowRes, w, h, fltPass, upsampleFilter, downsampleFilter, useBilateral)