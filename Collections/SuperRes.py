# SuperRes1(): Super Resolution
# SuperRes2(): Super Resolution with nnedi3 upsampling

# SuperRes(): Super Resolution with NLMeans filtering and user-defined resampling
"""Example of using nnedi3() as a main upsampling filter:

import nnedi3_resample as nnrs
from functools import partial

input = ...
target_width = ...
target_height = ...
upsampleFilter = partial(nnrs.nnedi3_resample, target_width=target_width, target_height=target_height)
superResolution = SuperRes(input, target_width, target_width, upsampleFilter1=upsampleFilter)

"""

# Appears to behave naturally when used to enhance textures during upsampling, though there would be lots of aliasing after filtering

# 16bit integer clip is required

# Main function
def SuperRes(lowRes, width, height, fltPass=3, upsampleFilter1=None, upsampleFilter2=None, downsampleFilter=None, useNLMeans=True, **knlm_args):
    if upsampleFilter1 is None:
        def upsampleFilter1(input):
            return core.fmtc.resample(input, width, height)
    if upsampleFilter2 is None:
        def upsampleFilter2(input):
            return core.fmtc.resample(input, width, height)
    if downsampleFilter is None:
        def downsampleFilter(input):
            return core.fmtc.resample(input, lowRes.width, lowRes.height)

    def computeError(input):
        return core.std.MakeDiff(lowRes, downsampleFilter(input))

    highRes = upsampleFilter1(lowRes)
    for i in range(fltPass):
        diff = upsampleFilter2(computeError(highRes))
        if useNLMeans:
            diff = core.knlm.KNLMeansCL(diff, rclip=highRes, **knlm_args)
        highRes = core.std.MergeDiff(highRes, diff)
    return highRes


# Wrap functions
def SuperRes1(lowRes, w, h, fltPass=3, useNLMeans=True, knlm_args=dict(), **fmtc_args):
    from functools import partial

    upsampleFilter = partial(core.fmtc.resample, w=w, h=h, **fmtc_args)
        
    downsampleFilter = partial(core.fmtc.resample, w=lowRes.width, h=lowRes.height, **fmtc_args)
    
    return SuperRes(lowRes, w, h, fltPass, upsampleFilter, upsampleFilter, downsampleFilter, useNLMeans, **knlm_args)

def SuperRes2(lowRes, w, h, fltPass=3, useNLMeans=True, nnedi3_args=dict(), knlm_args=dict(), **fmtc_args):
    from functools import partial
    import nnedi3_resample as nnrs
    
    upsampleFilter1 = partial(nnrs.nnedi3_resample, target_width=w, target_height=h, **nnedi3_args)
    
    upsampleFilter2 = partial(core.fmtc.resample, w=w, h=h, **fmtc_args)
    
    downsampleFilter = partial(core.fmtc.resample, w=lowRes.width, h=lowRes.height, **fmtc_args)
    
    return SuperRes(lowRes, w, h, fltPass, upsampleFilter1, upsampleFilter2, downsampleFilter, useNLMeans, **knlm_args)