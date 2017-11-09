import vapoursynth as vs
import mvsfunc as mvf
import functools
import numpy as np
import math

def numpy_process(clip, numpy_function, per_plane=True):
    """Helper function for filtering clip in numpy

    Args:
        clip: Input cilp.
        numpy_function: Function processed on the numpy data. It should not change the dimensions or data type of its input.
        per_plane: (bool) Whether to pass the data by plane. If not, the data would be passed by frame. Default is True.

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


def L0Smooth(clip, lamda=2e-2, kappa=2, **depth_args):
    """L0 Smooth Filter

    L0 smooth is a new image editing method, particularly effective for sharpening major edges 
    by increasing the steepness of transitions while eliminating a manageable degree of low-amplitude structures.
    It is achieved in an unconventional optimization framework making use of L0 gradient minimization,
    which can globally control how many non-zero gradients are resulted to approximate prominent structures in a structure-sparsity-management manner.
    Unlike other edge-preserving smoothing approaches, this method does not depend on local features and globally locates important edges.
    It, as a fundamental tool, finds many applications and is particularly beneficial to edge extraction, clip-art JPEG artifact removal, and non-photorealistic image generation.

    All the internal calculations are done at 32-bit float.

    This function has not been completed yet,
    every planes are processed separately, and the performance is slow.

    Args:
        src: Input image in float type.

        lamda: (float) Smoothing parameter controlling the degree of smooth.
            Typically it is within the range [1e-3, 1e-1].
            Default is 2e-2.

        kappa: (float) Parameter that controls the rate.
            Small kappa results in more iteratioins and with sharper edges.
            kappa = 2 is suggested for natural images.
            Default is 2.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Xu, L., Lu, C., Xu, Y., & Jia, J. (2011, December). Image smoothing via L0 gradient minimization. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 174). ACM.
        [2] http://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/index.html

    """

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    clip = numpy_process(clip, functools.partial(L0Smooth_core, lamda=lamda, kappa=kappa), per_plane=True)

    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def L0Smooth_core(src, lamda=2e-2, kappa=2):
    """Ported of L0 Smooth in numpy

    This function has not been completed yet,
    every planes are processed separately, and the performance is slow.

    Args:
        src: Gray scale image in float type.

        lamda: (float) Smoothing parameter controlling the degree of smooth.
            Typically it is within the range [1e-3, 1e-1].
            Default is 2e-2.

        kappa: (float) Parameter that controls the rate.
            Small kappa results in more iteratioins and with sharper edges.
            kappa = 2 is suggested for natural images.
            Default is 2.

    Ref:
        [1] Xu, L., Lu, C., Xu, Y., & Jia, J. (2011, December). Image smoothing via L 0 gradient minimization. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 174). ACM.
        [2] http://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/index.html

    """

    imgSize = src.shape
    size2D = imgSize[:2]
    betamax = 1e5

    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    otfFx = psf2otf(fx, size2D)
    otfFy = psf2otf(fy, size2D)

    Normin1 = np.fft.fft2(src, axes=(0, 1))

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    beta = 2 * lamda

    while beta < betamax:
        Denormin = 1 + beta * Denormin2
        h = np.hstack((np.diff(src, 1, 1), src[:, 0:1] - src[:, -1:]))
        v = np.vstack((np.diff(src, 1, 0), src[0:1, :] - src[-1:, :]))
        t = h ** 2 + v ** 2 < lamda / beta
        h[t] = 0
        v[t] = 0
        Normin2 = np.hstack((h[:, -1:] - h[:, 0:1], -np.diff(h, 1, 1))) + np.vstack((v[-1:, :] - v[0:1, :], -np.diff(v, 1, 0)))
        FS = (Normin1 + beta * np.fft.fft2(Normin2)) / Denormin
        src[:] = np.real(np.fft.ifft2(FS))
        beta *= kappa

    return src


def psf2otf(psf, outSize=None):
    """Function on convert point-spread function to optical transfer function

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