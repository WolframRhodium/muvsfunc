"""
Miscellaneous functions:
    GPS
    gauss
    freq_merge
    band_merge
    detail_enhancement
    SSR
"""

import vapoursynth as vs
import muvsfunc as muf
import mvsfunc as mvf
import functools

def GPS(clip, gamma=None):
    """Get Power Spectrum

    Args:
        gamma: It enables viewing small valued responses in the spectral display.

    """
    core = vs.get_core()

    w = clip.width
    h = clip.height
    max_w_h = max(w, h)

    clip = core.std.AddBorders(clip, right=max_w_h - w, bottom=max_w_h - h)
    clip = core.vcfreq.F2Quiver(clip, test=1, frad=16, fspec=[1,2,0,1,7], gamma=gamma)
    clip = core.std.CropRel(clip, 0, max_w_h // 2).resize.Bicubic(w, h)
    return clip


def gauss(clip, sigma=None, algo=0):
    """Gaussian filter using tcanny
    Borrowed from https://github.com/IFeelBloated/Oyster

    Args:
        sigma: Standard deviation of gaussian.
        algo: (int) Algorithm. 0:auto, 1:tcanny.TCanny(mode=-1), 2:bilateral.Gaussian()

    """

    core = vs.get_core()

    if (algo == 0 and sigma is not None and sigma >= 10) or algo == 2:
        return core.bilateral.Gaussian(clip, sigma=sigma)
    else: # algo == 1 or (algo == 0 and (sigma is None or sigma < 10))
        return core.tcanny.TCanny(clip, sigma=sigma, mode=-1)


def freq_merge(src, flt, fun=None, **fun_args):
    """Replace high freq component in "src" with high freq component in "flt"
    Borrowed from https://github.com/IFeelBloated/Oyster

    Args:
        src, flt: Input.
        fun:(function) A low-pass filter. Default is gaussian.
    """

    core = vs.get_core()

    if fun is None or not callable(fun):
        fun = gauss

    low_src = func(src, **fun_args)
    low_flt = func(flt, **fun_args)
    return core.std.Expr([low_src, flt, low_flt], ['y z - x +'])


def band_merge(src, flt, fun=None, fun_args1=None, fun_args2=None, cascade=True):
    """Replace frequencies within a certain range in "src" with frequencies within a certain range in "flt"

    Args:
        src, flt: I nput.
        fun:(function) A low-pass filter. Default is gaussian.
        cascade:(bool) Whether to cascade functions. Default is True.

    """

    core = vs.get_core()

    if fun is None or not callable(fun):
        fun = gauss

    if fun_args1 == None:
        fun_args1 = {}

    if fun_args2 == None:
        fun_args2 = {}

    low_src1 = fun(src, **fun_args1)
    low_src2 = fun(low_src1 if cascade else src, **fun_args2)
    low_flt1 = fun(flt, **fun_args1)
    low_flt2 = fun(low_flt1 if cascade else flt, **fun_args2)
    return core.std.Expr([low_flt1, low_flt2, src, low_src1, low_src2], ['x y - b + a - z +'])


def detail_enhancement(clip, guidance=None, iter=3, radius=4, regulation=0.0005, fast=False, **args):
    """Novel detail enhancement filter using guided filter and defilter

    Args:
        clip: Gray scale.
        guidance: Guidance clip.

    """

    return muf.DeFilter(clip, muf.GuidedFilter, guidance=guidance, radius=radius, regulation=regulation, fast=fast, iter=iter, **args)


def SSR(clip, sigma=50, full=None, **args):
    """Single-scale Retinex

    Args:
        clip: Input. Only the first plane will be processed.
        sigma:(int) Standard deviation of gaussian blur. Default is 50.
        full:(bool) Whether input clip is of full range. Default is None.

    Ref:
        [1] Jobson, D. J., Rahman, Z. U., & Woodell, G. A. (1997). Properties and performance of a center/surround retinex. IEEE transactions on image processing, 6(3), 451-462.

    """

    core = vs.get_core()

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type
    isGray = clip.format.color_family == vs.GRAY

    if not isGray:
        clip_src = clip
        clip = mvf.GetPlane(clip)

    lowFre = gauss(clip, sigma=sigma, **args)

    clip = mvf.Depth(clip, 32, fulls=full)
    lowFre = mvf.Depth(lowFre, 32, fulls=full) # core.bilateral.Gaussian() doesn't support float input.

    expr = 'x 1 + log y 1 + log -'
    clip = core.std.Expr([clip, lowFre], [expr])

    stats = core.std.PlaneStats(clip, plane=[0])

    # Dynamic range stretching
    def Stretch(n, f, clip, core):
        alpha = f.props.PlaneStatsMax - f.props.PlaneStatsMin
        beta = f.props.PlaneStatsMin

        expr = 'x {beta} - {alpha} /'.format(beta=beta, alpha=alpha)
        return core.std.Expr([clip], [expr])

    clip = core.std.FrameEval(clip, functools.partial(Stretch, clip=clip, core=core), prop_src=stats)

    clip = mvf.Depth(clip, depth=bits, sample=sampleType, fulld=full)

    if not isGray:
        clip = core.std.ShufflePlanes([clip, clip_src], list(range(clip_src.format.num_planes)), clip_src.format.color_family)

    return clip
