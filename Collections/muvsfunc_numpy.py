"""
VapourSynth functions:
    numpy_process (helper function)
    L0Smooth
    L0GradientProjection

NumPy functions:
    L0Smooth_core
    psf2otf
    L0GradProj_core
"""

import functools
import math
import vapoursynth as vs
import mvsfunc as mvf
import numpy as np

def numpy_process(clip, numpy_function, per_plane=True, lock_source_array=True, **fun_args):
    """Helper function for filtering clip in NumPy

    Args:
        clip: Input cilp.

        numpy_function: Spatial function to process numpy data. It should not change the dimensions or data type of its input.

        per_plane: (bool) Whether to process data by plane. If not, data would be processed by frame.
            Default is True.

        lock_source_array: (bool) Whether to lock the source array to avoid unintentionally overwrite the data.
            Default is True.

        fun_args: (dict) Additional arguments passed to “numpy_function” in the form of keyword arguments.
            Default is {}.

    """

    core = vs.get_core()

    # The following code is modified from https://github.com/KotoriCANOE/MyTF/blob/master/utils/vshelper.py
    def FLT(n, f):
        fout = f.copy()
        planes = fout.format.num_planes
        if per_plane:
            for p in range(planes):
                s = np.asarray(f.get_read_array(p))
                if lock_source_array:
                    s.flags.writeable = False # Lock the source data, making it read-only

                fs = numpy_function(s, **fun_args)

                d = np.asarray(fout.get_write_array(p))
                np.copyto(d, fs)
                del d
        else:
            s_list = []
            for p in range(planes):
                arr = np.asarray(f.get_read_array(p)) # This is a 2-D array
                s_list.append(arr)
            s = np.stack(s_list, axis=2) # "s" is a 3-D array
            if lock_source_array:
                s.flags.writeable = False # Lock the source data, making it read-only

            fs = numpy_function(s, **fun_args)

            for p in range(planes):
                d = np.asarray(fout.get_write_array(p))
                np.copyto(d, fs[:, :, p])
                del d
        return fout

    flt = core.std.ModifyFrame(clip, clip, FLT)

    return flt


def L0Smooth(clip, lamda=2e-2, kappa=2, color=True, **depth_args):
    """L0 Smooth in VapourSynth

    It is also known as "L0 Gradient Minimization".

    L0 smooth is a new image editing method, particularly effective for sharpening major edges
    by increasing the steepness of transitions while eliminating a manageable degree of low-amplitude structures.
    It is achieved in an unconventional optimization framework making use of L0 gradient minimization,
    which can globally control how many non-zero gradients are resulted to approximate prominent structures in a structure-sparsity-management manner.
    Unlike other edge-preserving smoothing approaches, this method does not depend on local features and globally locates important edges.
    It, as a fundamental tool, finds many applications and is particularly beneficial to edge extraction, clip-art JPEG artifact removal, and non-photorealistic image generation.

    All the internal calculations are done at 32-bit float.

    Args:
        clip: Input clip with no chroma subsampling.

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

    core = vs.get_core()

    if not isinstance(clip, vs.VideoNode) or any((clip.format.subsampling_w, clip.format.subsampling_h)):
        raise TypeError(funcName + ': \"clip\" must be a clip with no chroma subsampling!')

    # Internal parameters
    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type
    per_plane = not color or clip.format.num_planes == 1

    # Convert to floating point
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    # Add padding for real Fast Fourier Transform
    if clip.width & 1:
        pad = True
        clip = core.std.AddBorders(clip, left=1)
    else:
        pad = False

    # Pre-calculate constant 2-D matrix
    size2D = (clip.height, clip.width)
    Denormin2 = L0Smooth_generate_denormin2(size2D)

    # Process
    clip = numpy_process(clip, functools.partial(L0Smooth_core, lamda=lamda, kappa=kappa, Denormin2=Denormin2, copy=True), per_plane=per_plane)

    # Crop the padding
    if pad:
        clip = core.std.CropRel(clip, left=1)

    # Convert the bit depth and sample type of output to the same as input
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def L0Smooth_generate_denormin2(size2D):
    """Helper function to generate constant "Denormin2"
    """
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    otfFx = psf2otf(fx, outSize=size2D)
    otfFy = psf2otf(fy, outSize=size2D)
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    Denormin2 = Denormin2[:, :size2D[1]//2+1]

    return Denormin2


def L0Smooth_core(src, lamda=2e-2, kappa=2, Denormin2=None, copy=False):
    """L0 Smooth in NumPy.

    It is also known as "L0 Gradient Minimization".

    Args:
        src: 2-D or 3-D numpy array. The length along the second dimension must be even.
            3-D data will be processed collaboratively, which is the same as the official MATLAB version.

        lamda: (float) Smoothing parameter controlling the degree of smooth.
            Default is 2e-2.

        kappa: (float) Parameter that controls the rate of convergence.
            Default is 2.

        Denormin2: (ndarray) Constant matrix. If it is None, it will be calculated automatically.
            If "src" is a 2-D array, "Denormin2" must also be 2-D array.
            Else, if "src" is a 2-D array, "Denormin2" can be either 2-D or 3-D array.

        copy: (bool) Whether to copy the data before processing. Default is False.

        For detailed documentation, please refer to the documentation of "L0Smooth" funcion in current library.

    TODO: Optimize FFT using pyfftw library.

    """

    funcName = 'L0Smooth_core'

    if not isinstance(src, np.ndarray) or src.ndim not in (2, 3):
        raise TypeError(funcName + ': \"src\" must be 2-D or 3-D numpy data!')

    if src.shape[1] & 1:
        raise TypeError(funcName + ': the length of \"src\" along the second dimension must be even!')

    if copy:
        src = src.copy()

    # Get size
    imgSize = src.shape
    size2D = imgSize[:2]
    r_size2D = (size2D[0], size2D[1] // 2 + 1)
    D = imgSize[2] if src.ndim == 3 else 1

    # Generate constant "Denormin2"
    if Denormin2 is None:
        Denormin2 = L0Smooth_generate_denormin2(size2D)

    if Denormin2.shape[:2] == size2D:
        Denormin2 = Denormin2[:, :size2D[1]//2+1]

    if src.ndim == 3 and Denormin2.shape == r_size2D:
        Denormin2 = Denormin2[:, :, np.newaxis]

    if (src.ndim == 2 and Denormin2.shape != r_size2D) or (src.ndim == 3 and Denormin2.shape not in ((*r_size2D, 1), (*r_size2D, D))):
        raise ValueError(funcName + ': the shape of \"Denormin2\" must be {}!'.format((*r_size2D, 1)))

    # Internal parameters
    beta = 2 * lamda
    betamax = 1e5

    # Pre-allocate memory
    Denormin = np.empty_like(Denormin2)
    h = np.empty_like(src)
    v = np.empty_like(src)
    t = np.empty(size2D, dtype='bool')
    FS = np.empty(r_size2D if src.ndim == 2 else (*r_size2D, D), dtype='complex')
    Normin2 = np.empty_like(src)

    # Start processing
    Normin1 = np.fft.rfft2(src, axes=(0, 1))

    while beta < betamax:
        Denormin = 1 + beta * Denormin2

        # h-v subproblem
        #h = np.hstack((np.diff(src, 1, 1), src[:, 0:1] - src[:, -1:]))
        #v = np.vstack((np.diff(src, 1, 0), src[0:1, :] - src[-1:, :]))
        h[:, :-1] = src[:, 1:] - src[:, :-1]
        h[:, -1:] = src[:, :1] - src[:, -1:]
        v[:-1, :] = src[1:, :] - src[:-1, :]
        v[-1:, :] = src[:1, :] - src[-1:, :]
        if src.ndim == 3:
            t[:] = np.sum(h ** 2 + v ** 2, 2) < lamda / beta
        else: # src.ndim == 2
            t[:] = (h ** 2 + v ** 2) < lamda / beta
        h[t] = 0
        v[t] = 0

        # S subproblem
        #Normin2 = np.hstack((h[:, -1:] - h[:, 0:1], -np.diff(h, 1, 1))) + np.vstack((v[-1:, :] - v[0:1, :], -np.diff(v, 1, 0)))
        Normin2[:, :1] = h[:, -1:] - h[:, :1]
        Normin2[:, 1:] = h[:, :-1] - h[:, 1:]
        Normin2[:1, :] += v[-1:, :] - v[:1, :]
        Normin2[1:, :] += v[:-1, :] - v[1:, :]
        FS[:] = (Normin1 + beta * np.fft.rfft2(Normin2, axes=(0, 1))) / Denormin
        src[:] = np.fft.irfft2(FS, axes=(0, 1))

        # Updata parameter
        beta *= kappa

    return src


def psf2otf(psf, outSize=None, fast=False):
    """Function of convert point-spread function to optical transfer function

    Ported from MATLAB

    Args:
        psf: Point-spread function in numpy.ndarray.

        outSize: (tuple) The size of the OTF array. Default is the same as psf.

        fast: (tuple) Whether to check the resulting values and discard the imaginary part if it's within roundoff error.
            Default is False.

    """

    funcName = 'psf2otf'

    psfSize = np.array(np.shape(psf))

    if outSize is None:
        outSize = psfSize
    elif not isinstance(outSize, np.ndarray):
        outSize = np.array(outSize)

    # Pad the PSF to outSize
    padSize = outSize - psfSize
    psf = np.lib.pad(psf, pad_width=[(0, i) for i in padSize], mode='constant', constant_values=0)

    # Circularly shift otf so that the "center" of the PSF is at the (0, 0) element of the array.
    psf = np.roll(psf, shift=tuple(-np.floor_divide(psfSize, 2)), axis=tuple(range(psf.ndim)))

    # Compute the OTF
    otf = np.fft.fftn(psf)

    if not fast:
        # Estimate the rough number of operations involved in the computation of the FFT.
        nElem = np.prod(psfSize)
        nOps = 0
        for k in range(np.ndim(psf)):
            nffts = nElem / psfSize[k]
            nOps += psfSize[k] * math.log2(psfSize[k]) * nffts

        # Discard the imaginary part of the psf if it's within roundoff error.
        eps = 2.220446049250313e-16
        if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * eps:
            otf = np.real(otf)

    return otf


def L0GradientProjection(clip, alpha=0.08, precision=255, epsilon=0.0002, maxiter=5000, gamma=3, eta=0.95, color=True, **depth_args):
    """L0 Gradient Projection in VapourSynth

    L0 gradient projection is a new edge-preserving filtering method based on the L0 gradient.
    In contrast to the L0 gradient minimization, L0 gradient projection framework is intuitive
    because one can directly impose a desired L0 gradient value on the output image.
    The constrained optimization problem is minimizing the quadratic data-fidelity subject to the hard constraint that
    the L0 gradient is less than a user-given parameter "alpha".
    The solution is based on alternating direction method of multipliers (ADMM), while the hard constraint is handled by
    projection onto the mixed L1,0 pseudo-norm ball with variable splitting, and the computational complexity is O(NlogN).

    However, current implementation here is extremely slow. In my experiment, the number of iteration of this algorithm is far more than L0Smooth.

    All the internal calculations are done at 32-bit float.

    Args:
        clip: Input clip with no chroma subsampling.

        alpha: (float) L0 gradient of output in percentage form, i.e. the range is [0, 1].
            It can be seen as the degree of flatness in the output.
            Default is 0.08.

        precision: (float) Precision of the calculation of L0 gradient. The larger the value, the more accurate the calculation.
            Default is 255.

        epsilon: (float) Stopping criterion in percentage form, i.e. the range is [0, 1].
            It determines the allowable error from alpha.
            Default is 0.0002.

        maxiter: (int) Maximum number of iterations.
            Default is 5000.

        gamma: (int) Step size of ADMM.
            Default is 3.

        eta: (int) Controling gamma for nonconvex optimization.
            It stabilizes ADMM for nonconvex optimization.
            According to recent convergence analyses of ADMM for nonconvex cases, under appropriate conditions,
            the sequence generated by ADMM converges to a stationary point with sufficiently small gamma.
            Default is 0.95.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Ono, S. (2017). $ L_ {0} $ Gradient Projection. IEEE Transactions on Image Processing, 26(4), 1554-1564.
        [2] Ono, S. (2017, March). Edge-preserving filtering by projection onto L 0 gradient constraint. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 1492-1496). IEEE.
        [3] https://sites.google.com/site/thunsukeono/publications

    TODO: Optimize FFT using pyfftw library.

    """

    funcName = 'L0GradientProjection'

    core = vs.get_core()

    if not isinstance(clip, vs.VideoNode) or any((clip.format.subsampling_w, clip.format.subsampling_h)):
        raise TypeError(funcName + ': \"clip\" must be a clip with no chroma subsampling!')

    # Internal parameters
    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type
    per_plane = not color or clip.format.num_planes == 1

    # Convert to floating point
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    # Add padding for real Fast Fourier Transform
    if clip.width & 1:
        pad = True
        clip = core.std.AddBorders(clip, left=1)
    else:
        pad = False

    # Pre-calculate constant 2-D matrix
    size2D = (clip.height, clip.width)
    Lap = L0GradProj_generate_lap(size2D)

    # Process
    clip = numpy_process(clip, functools.partial(L0GradProj_core, alpha=alpha, precision=precision, epsilon=epsilon, maxiter=maxiter,
        gamma=gamma, eta=eta, Lap=Lap, copy=True), per_plane=per_plane)

    # Crop the padding
    if pad:
        clip = core.std.CropRel(clip, left=1)

    # Convert the bit depth and sample type of output to the same as input
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def L0GradProj_core(src, alpha=0.08, precision=255, epsilon=0.0002, maxiter=5000, gamma=3, eta=0.95, Lap=None, copy=False):
    """L0 Gradient Projection in NumPy.

    Args:
        src: 2-D or 3-D numpy array. The length along the second dimension must be even.
            3-D data will be processed collaboratively, which is the same as the official MATLAB version.

        alpha: (float) L0 gradient of output in percentage form, i.e. the range is [0, 1]. Default is 0.08.

        precision: (float) Precision of the calculation of L0 gradient. Default is 255.

        epsilon: (float) Stopping criterion in percentage form, i.e. the range is [0, 1]. Default is 0.0002.

        maxiter: (int) Maximum number of iterations. Default is 5000.

        gamma: (int) Step size of ADMM. Default is 3.

        eta: (int) Controling gamma for nonconvex optimization. Default is 0.95.

        Lap: (ndarray) Constant matrix. If it is None, it will be calculated automatically.

        copy: (bool) Whether to copy the data before processing. Default is False.

        For detailed documentation, please refer to the documentation of "L0GradientProjection" funcion in current library.

    TODO: Optimize FFT using pyfftw library.

    """

    funcName = 'L0GradProj_core'

    if not isinstance(src, np.ndarray) or src.ndim not in (2, 3):
        raise TypeError(funcName + ': \"src\" must be 2-D or 3-D numpy data!')

    if src.shape[1] & 1:
        raise TypeError(funcName + ': the length of \"src\" along the second dimension must be even!')

    if copy:
        src = src.copy()

    src_ndim = src.ndim
    src_shape = src.shape


    N = src_shape[0] * src_shape[1]
    alpha = round(alpha * N)
    epsilon *= N

    if src_ndim == 2:
        src = src[:, :, np.newaxis, np.newaxis]
    else: # img.ndim == 3
        src = src[:, :, :, np.newaxis]

    # difference operators (periodic boundary)
    D = lambda z: np.concatenate([z[np.r_[1:z.shape[0], 0], :, :, :] - z, z[:, np.r_[1:z.shape[1], 0], :, :] - z], axis=3)
    Dt = lambda z: np.vstack([-z[:1, :, :, :1] + z[-1:, :, :, :1], -z[1:, :, :, :1] + z[:-1, :, :, :1]]) + np.hstack([-z[:, :1, :, 1:2] + z[:, -1:, :, 1:2], -z[:, 1:, :, 1:2] + z[:, :-1, :, 1:2]])

    # for fftbased diagonilization
    if Lap is None:
        Lap = L0GradProj_generate_lap(src_shape[:2])

    if Lap.shape == src_shape[:2]:
        Lap = Lap[:, :src_shape[1]//2+1, np.newaxis, np.newaxis]

    if Lap.shape != (src_shape[0], src_shape[1]//2+1, 1, 1):
        raise ValueError(funcName + ': the shape of \"Lap\" must be {}!'.format(src_shape[:2]))

    # calculating L0 gradient value
    # z: 3-D array
    #L0gradcalc = lambda z: L0GradValue(D((z[:, :, :, np.newaxis] * 255).astype(np.uint8).astype(np.float32)))
    L0gradcalc = lambda z: L0GradProj_L0GradValue(D(np.round(z[:, :, :, np.newaxis] * precision)))

    # variables
    u = np.empty_like(src)
    v = D(src)
    w = v.copy()

    for i in range(maxiter):
        rhs = src + Dt(v - w) / gamma

        u[:] = np.fft.irfft2(np.fft.rfft2(rhs, axes=(0, 1)) / (Lap / gamma + 1), axes=(0, 1))
        v[:] = L0GradProj_ProjL1ball(D(u) + w, alpha)
        w += D(u) - v

        gamma *= eta

        L0Grad = L0gradcalc(u)
        if abs(L0Grad - alpha) < epsilon:
            break

    u = u.reshape(src_shape)

    return u


def L0GradProj_ProjL1ball(Du, epsilon):
    """Internal function for L0GradProj_core()

    Projection onto mixed L1,0 pseudo-norm ball with binary mask

    Args:
        Du: 4-D array
        epsilon: (int) Threshold of the constraint.

    """

    sizeDu = Du.shape

    Du1 = Du[-1, :, :, 0].copy()
    Du2 = Du[:, -1, :, 1].copy()

    # masking differences between opposite boundaries
    Du[-1, :, :, 0] = 0
    Du[:, -1, :, 1] = 0

    sumDu = np.sum(Du ** 2, axis=(2, 3))
    # The worst-case complexity of Sort(modified quicksort actually) in MATLAB is O(n^2)
    # while it is O(n) for numpy.partition(introselect)
    I = np.argpartition(-sumDu.reshape(-1), epsilon)[:epsilon]
    threInd = np.zeros(sizeDu[:2])
    threInd.reshape(-1)[I] = 1; # set ones for values to be held

    threInd = np.tile(threInd[:, :, np.newaxis, np.newaxis], (1, 1, *sizeDu[2:]))
    Du *= threInd

    Du[-1, :, :, 0] = Du1
    Du[:, -1, :, 1] = Du2

    return Du


def L0GradProj_L0GradValue(Du):
    """Internal function for L0GradProj_core()

    Calculate L0 gradient

    Args:
        Du: 4-D array

    """

    Du[-1, :, :, 0] = 0
    Du[:, -1, :, 1] = 0

    return np.count_nonzero(np.abs(Du).sum(axis=(2, 3)).round())


def L0GradProj_generate_lap(size2D):
    """Helper function to generate constant "Denormin2"
    """
    Lap = np.zeros(size2D)
    Lap[0, 0] = 4
    Lap[0, 1] = -1
    Lap[1, 0] = -1
    Lap[-1, 0] = -1
    Lap[0, -1] = -1
    Lap = np.fft.fft2(Lap, axes=(0, 1))

    return Lap
