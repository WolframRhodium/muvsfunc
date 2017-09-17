"""Miscellaneous functions
"""

import vapoursynth as vs
import muvsfunc as muf

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


def gauss(clip, sigma=None):
    """Gaussian filter using tcanny
    Borrowed from https://github.com/IFeelBloated/Oyster

    Args:
        sigma: Standard deviation of gaussian.

    """

    core = vs.get_core()

    return core.tcanny.TCanny(clip, sigma=sigma, mode=-1)


def freq_merge(src, flt, func=None, **func_args):
    """Replace high freq component in "src" with high freq component in "flt"
    Borrowed from https://github.com/IFeelBloated/Oyster

    Args:
        src, flt: Input.
        func:(function) A low-pass filter. Default is gaussian.
    """

    core = vs.get_core()

    if func is None or not callable(func):
        func = gauss

    low_src = func(src, **func_args)
    low_flt = func(flt, **func_args)
    return core.std.Expr([low_src, flt, low_flt], ['y z - x +'])


def band_merge(src, flt, func=None, func_args1=None, func_args2=None, cascade=True):
    """Replace frequencies within a certain range in "src" with frequencies within a certain range in "flt"

    Args:
        src, flt: Input.
        func:(function) A low-pass filter. Default is gaussian.
        cascade:(bool) Whether to cascade functions. Default is True.
    """

    core = vs.get_core()

    if func is None or not callable(func):
        func = gauss

    if func_args1 == None:
        func_args1 = {}

    if func_args2 == None:
        func_args2 = {}

    low_src1 = func(src, **func_args1)
    low_src2 = func(low_src1 if cascade else src, **func_args2)
    low_flt1 = func(flt, **func_args1)
    low_flt2 = func(low_flt1 if cascade else flt, **func_args2)
    return core.std.Expr([low_flt1, low_flt2, src, low_src1, low_src2], ['x y - b + a - z +'])


def detail_enhancement(clip, iter=3, radius=4, regulation=0.0005, fast=False, **args):
    """Novel detail enhancement filter using guided filter and defilter

    Args:
        clip: Gray scale.
    """
    
    return muf.DeFilter(clip, muf.GuidedFilter, guidance=clip, radius=radius, regulation=regulation, fast=False, iter=iter, **args)