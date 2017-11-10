import vapoursynth as vs
import mvsfunc as mvf
import functools
import numpy as np
import math

def numpy_process(clip, numpy_function, per_plane=True):
    """Helper function for filtering clip in numpy

    Args:
        clip: Input cilp.

        numpy_function: Spatial function to process numpy data. It should not change the dimensions or data type of its input.

        per_plane: (bool) Whether to process data by plane. If not, data would be processed by frame. Default is True.

    """

    core = vs.get_core()

    # The following code is modified from https://github.com/KotoriCANOE/MyTF/blob/master/utils/vshelper.py
    def FLT(n, f):
        fout = f.copy()
        planes = fout.format.num_planes
        if per_plane:
            for p in range(planes):
                s = np.array(fout.get_read_array(p), copy=False)
                s[:] = numpy_function(s)

                d = np.array(fout.get_write_array(p), copy=False)
                np.copyto(d, s)
                del d, s
        else:
            s_list = []
            for p in range(planes):
                arr = np.array(fout.get_read_array(p), copy=False)
                s_list.append(arr.reshape(list(arr.shape) + [1]))
            s = np.concatenate(s_list, axis=2)
            s[:] = numpy_function(s)

            for p in range(planes):
                d = np.array(fout.get_write_array(p), copy=False)
                np.copyto(d, s[:, :, p])
            del d, s

        return fout

    flt = core.std.ModifyFrame(clip, clip, FLT)

    return flt


def L0Smooth(clip, lamda=2e-2, kappa=2, color=True, **depth_args):
    """L0 Smooth in VapourSynth

    L0 smooth is a new image editing method, particularly effective for sharpening major edges 
    by increasing the steepness of transitions while eliminating a manageable degree of low-amplitude structures.
    It is achieved in an unconventional optimization framework making use of L0 gradient minimization,
    which can globally control how many non-zero gradients are resulted to approximate prominent structures in a structure-sparsity-management manner.
    Unlike other edge-preserving smoothing approaches, this method does not depend on local features and globally locates important edges.
    It, as a fundamental tool, finds many applications and is particularly beneficial to edge extraction, clip-art JPEG artifact removal, and non-photorealistic image generation.

    All the internal calculations are done at 32-bit float.

    Args:
        src: Input clip with no chroma subsampling.

        lamda: (float) Smoothing parameter controlling the degree of smooth.
            A large "lamda" makes the result have very few edges.
            Typically it is within the range [0.001, 0.1].
            This parameter is renamed from "lambda" for better compatibility with Python.
            Default is 0.02.

        kappa: (float) Parameter that controls the convergence rate of alternating minimization algorithm.
            Small "kappa" results in more iteratioins and with sharper edges.
            kappa = 2 is suggested for natural images, which is a good balance between efficiency and performance.
            Default is 2.

        color: (bool) Whether to process data collaboratively.
            If true, the gradient magnitude in the model is defined as the sum of gradient magnitudes in the original color space.
            If false, channels in "clip" will be processed separately.
            Default is True.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Xu, L., Lu, C., Xu, Y., & Jia, J. (2011, December). Image smoothing via L0 gradient minimization. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 174). ACM.
        [2] http://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/index.html

    TODO: Optimize FFT using pyfftw library.

    """

    funcName = 'L0Smooth'

    if not isinstance(clip, vs.VideoNode) or any((clip.format.subsampling_w, clip.format.subsampling_h)):
        raise TypeError(funcName + ': \"clip\" must be a clip with no chroma subsampling!')

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    clip = numpy_process(clip, functools.partial(L0Smooth_core, lamda=lamda, kappa=kappa), per_plane=(not color or clip.format.num_planes == 1))

    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def L0Smooth_core(src, lamda=2e-2, kappa=2):
    """L0 Smooth in NumPy.

    Args:
        src: 2-D or 3-D numpy array.
            3-D data will be processed collaboratively, which is the same as the official MATLAB version.

        lamda: (float) Smoothing parameter controlling the degree of smooth.
            Default is 2e-2.

        kappa: (float) Parameter that controls the rate of convergence.
            Default is 2.

        For detailed documentation, please refer to the docstring of "L0Smooth" funcion.

    TODO: Optimize FFT using pyfftw library.

    """

    funcName = 'L0_smooth_core'

    if not isinstance(src, np.ndarray) or src.ndim not in (2, 3):
        raise TypeError(funcName + ': \"src\" must be 2-D or 3-D numpy data!')

    imgSize = src.shape
    size2D = imgSize[:2]
    D = np.size(src, 2) if src.ndim == 3 else 1
    betamax = 1e5

    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    otfFx = psf2otf(fx, size2D)
    otfFy = psf2otf(fy, size2D)

    Normin1 = np.fft.fft2(src, axes=(0, 1))

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    beta = 2 * lamda

    if src.ndim == 3:
        Denormin2 = np.tile(Denormin2.reshape(list(Denormin2.shape) + [1]), (1, 1, D))

    while beta < betamax:
        Denormin = 1 + beta * Denormin2
        # h-v subproblem
        h = np.hstack((np.diff(src, 1, 1), src[:, 0:1] - src[:, -1:]))
        v = np.vstack((np.diff(src, 1, 0), src[0:1, :] - src[-1:, :]))
        if src.ndim == 3:
            t = np.sum(h ** 2 + v ** 2, 2) < lamda / beta
            t = np.tile(t.reshape(list(t.shape) + [1]), (1, 1, D))
        else: # src.ndim == 2
            t = h ** 2 + v ** 2 < lamda / beta
        h[t] = 0
        v[t] = 0
        # S subproblem
        Normin2 = np.hstack((h[:, -1:] - h[:, 0:1], -np.diff(h, 1, 1))) + np.vstack((v[-1:, :] - v[0:1, :], -np.diff(v, 1, 0)))
        FS = (Normin1 + beta * np.fft.fft2(Normin2, axes=(0, 1))) / Denormin
        src[:] = np.real(np.fft.ifft2(FS, axes=(0, 1)))
        beta *= kappa

    return src


def psf2otf(psf, outSize=None):
    """Function of convert point-spread function to optical transfer function

    Ported from MATLAB

    Args:
        psf: Point-spread function in numpy.ndarray.

        outSize: (tuple) The size of the OTF array. Default is the same as psf.

    """

    funcName = 'psf2otf'

    psfSize = np.array(np.shape(psf))

    if outSize is None:
        outSize = psfSize
    elif not isinstance(outSize, np.ndarray):
        outSize = np.array(outSize)
    else:
        raise TypeError("\'outSize\' must be a tuple!")

    # Pad the PSF to outSize
    padSize = tuple(outSize - psfSize)
    psf = np.lib.pad(psf, [(0, i) for i in padSize], 'constant')

    # Circularly shift otf so that the "center" of the PSF is at the (0, 0) element of the array.
    psf = np.roll(psf, shift=tuple(-np.floor_divide(psfSize, 2)), axis=tuple(range(np.size(psfSize))))

    # Compute the OTF
    otf = np.fft.fftn(psf)

    # Estimate the rough number of operations involved in the computation of the FFT.
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(np.ndim(psf)):
        nffts = nElem / psfSize[k]
        nOps += psfSize[k] * math.log2(psfSize[k]) * nffts

    # Discard the imaginary part of the psf if it's within roundoff error.
    eps = 2.220446049250313e-16
    if np.max(np.abs(np.imag(otf.flatten()))) / np.max(np.abs(otf.flatten())) <= nOps * eps:
        otf = np.real(otf)

    return otf