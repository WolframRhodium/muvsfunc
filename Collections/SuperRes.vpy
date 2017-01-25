# SuperRes1(): Single Image Super Resolution
# SuperRes2(): Single Image Super Resolution with Bilateral filtering

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