"""
Miscellaneous functions:
    GPS
    gauss
    freq_merge
    band_merge
    detail_enhancement
    SSR
    Wiener2
    tv
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
        sigma: (int) Standard deviation of gaussian blur. Default is 50.
        full: (bool) Whether input clip is of full range. Default is None.

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


def Wiener2(input, radius_v=3, radius_h=None, noise=None, **depth_args):
    """2-D adaptive noise-removal filtering. (wiener2 from MATLAB)

    Wiener2 lowpass filters an intensity image that has been degraded by constant power additive noise.
    Wiener2 uses a pixel-wise adaptive Wiener method based on statistics estimated from a local neighborhood of each pixel.

    Estimate of the additive noise power will not be returned.

    Args:
        input: Input clip. Only the first plane will be processed.
        radius_v, radius_h: (int) Size of neighborhoods to estimate the local image mean and standard deviation. The size is (radius_v*2-1) * (radius_h*2-1).
            If "radius_h" is None, it will be set to "radius_v".
            Default is 3.
        noise: (float) Variance of addictive noise. If it is not given, average of all the local estimated variances will be used.
            Default is {}.
        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Lim, J. S. (1990). Two-dimensional signal and image processing. Englewood Cliffs, NJ, Prentice Hall, 1990, 710 p, p. 538, equations 9.26, 9.27, and 9.29.
        [2] 2-D adaptive noise-removal filtering - MATLAB wiener2 - MathWorks (https://www.mathworks.com/help/images/ref/wiener2.html)

    """

    core = vs.get_core()
    funcName = 'Wiener2'

    if not isinstance(input, vs.VideoNode) or input.format.num_planes > 1:
        raise TypeError(funcName + ': \"input\" must be a gray-scale/single channel clip!')

    bits = input.format.bits_per_sample
    sampleType = input.format.sample_type

    if radius_h is None:
        radius_h = radius_v

    input32 = mvf.Depth(input, depth=32, sample=vs.FLOAT, **depth_args)

    localMean = BoxFilter(input32, radius_h+1, radius_v+1)
    localVar = BoxFilter(core.std.Expr([input32], ['x dup *']), radius_h+1, radius_v+1)
    localVar = core.std.Expr([localVar, localMean], ['x y dup * -'])

    if noise is None:
        localVarStats = core.std.PlaneStats(localVar, plane=[0])

        def FLT(n, f, clip, core, localMean, localVar):
            noise = f.props.PlaneStatsAverage

            return core.std.Expr([clip, localMean, localVar], ['y z {noise} - 0 max z {noise} max / x y - * +'.format(noise=noise)])

        flt = core.std.FrameEval(input32, functools.partial(FLT, clip=input32, core=core, localMean=localMean, localVar=localVar), prop_src=[localVarStats])
    else:
        flt = core.std.Expr([input32, localMean, localVar], ['y z {noise} - 0 max z {noise} max / x y - * +'.format(noise=noise)])

    return mvf.Depth(flt, depth=bits, sample=sampleType, **depth_args)


def tv(I, iter=5, dt=None, ep=1, lam=0, I0=None, C=0):
    """Total Variation Denoising

    Args:
        I: Input. Recommended to input floating type clip.
        iter: (int) Num of iterations. Default is 5.
        dt: (float) Time step. Default is ep/5.
        ep: (float) Epsilon (of gradient regularization). Default is 1.
        lam: (float) Fidelity term lambda. Default is 0.
        I0: (clip) Input (noisy) image. Default is "I".
    
    Ref:
        [1] Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1-4), 259-268.
        [2] Total Variation Denoising : http://visl.technion.ac.il/~gilboa/PDE-filt/tv_denoising.html

    """

    core = vs.get_core()

    if dt is None:
        dt = ep / 5

    if I0 is None:
        I0 = I

    ep2 = ep * ep

    isFloat = I.format.sample_type == vs.FLOAT
    neutral = 0 if isFloat else muf.scale(128, I.format.bits_per_sample)

    for i in range(iter):
        I_x = core.std.Convolution(I, [-1, 0, 1], divisor=2, bias=neutral, mode='h') # correct
        I_y = core.std.Convolution(I, [-1, 0, 1], divisor=2, bias=neutral, mode='v') # correct
        I_xx = core.std.Convolution(I, [1, -2, 1], divisor=1 if isFloat else 4, bias=neutral, mode='h') # x4
        I_yy = core.std.Convolution(I, [1, -2, 1], divisor=1 if isFloat else 4, bias=neutral, mode='v') # x4
        Dp = core.std.Convolution(I, [1, 0, 0, 0, 0, 0, 0, 0, 1], divisor=2)
        Dm = core.std.Convolution(I, [0, 0, 1, 0, 0, 0, 1, 0, 0], divisor=2)
        I_xy = core.std.Expr([Dp, Dm], ['x y - 2 / {} +'.format(neutral)]) # correct

        if isFloat:
            expr = 'x {dt} a {ep2} z dup * + * 2 y * z * b * - c {ep2} y dup * + * + {ep2} y dup * + z dup * + 1.5 pow / {lam} d x - {C} + * + * +'.format(dt=dt, ep2=ep2, lam=lam, C=C)
        else: # isInteger
            expr = 'x {dt} a {neutral} - 4 * {ep2} z {neutral} - dup * + * 2 y {neutral} - * z {neutral} - * b {neutral} - * - c {neutral} - 4 * {ep2} y {neutral} - dup * + * + {ep2} y {neutral} - dup * + z {neutral} - dup * + 1.5 pow / {lam} d x - {C} + * + * +'.format(dt=dt, neutral=neutral, ep2=ep2, lam=lam, C=C)

        I = core.std.Expr([I, I_x, I_y, I_xx, I_xy, I_yy, I0], [expr])

    return I