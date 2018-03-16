"""
VapourSynth functions:
    numpy_process (helper function)
    numpy_process_val (helper function)
    L0Smooth
    L0GradientProjection
    IEDD
    DNCNN
    BNNMDenoise
    FGS
    FDD
    SSFDeband
    SigmaFilter

NumPy functions:
    L0Smooth_core
    psf2otf
    L0GradProj_core
    IEDD_core
    get_blockwise_view
    BNNMDenoise_core
    FGS_2D_core
    FDD_2D_core
    SSFDeband_core
    SigmaFilter_core
"""

import functools
import math
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import numpy as np
from numpy.lib.stride_tricks import as_strided

def numpy_process(clips, numpy_function, per_plane=True, lock_source_array=True, **fun_args):
    """Helper function for filtering clip in NumPy

    Args:
        clips: Input cilp.
            It can also be a list of clips. If so, clips will be passed to "numpy_function" in order.
            The returned clip should has the same format as the first clip in the list.

        numpy_function: Spatial function to process numpy data. It should not change the dimensions or data type of its input.
            The format of the data provided to the function is "HWC", 
            i.e. number of pixels in vertical(height), horizontal(width) dimension and channels respectively.

        per_plane: (bool) Whether to process data by plane. If not, data would be processed by frame.
            Default is True.

        lock_source_array: (bool) Whether to lock the source array to avoid unintentionally overwrite the data.
            Default is True.

        fun_args: (dict) Additional arguments passed to “numpy_function” in the form of keyword arguments.
            Default is {}.

    """


    # The following code is modified from https://github.com/KotoriCANOE/MyTF/blob/master/utils/vshelper.py
    def FLT(n, f):
        if not isinstance(f, list): # single input
            fout = f.copy()
            planes = fout.format.num_planes
            if per_plane:
                for p in range(planes):
                    s = np.asarray(f.get_read_array(p))
                    s.flags.writeable = not lock_source_array # Lock the source data, making it read-only

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
                s.flags.writeable = not lock_source_array # Lock the source data, making it read-only

                fs = numpy_function(s, **fun_args)

                for p in range(planes):
                    d = np.asarray(fout.get_write_array(p))
                    np.copyto(d, fs[:, :, p])
                    del d
            return fout
        else: # multiple input
            fout = f[0].copy()
            planes = fout.format.num_planes
            if per_plane:
                for p in range(planes):
                    s_list = []
                    for frame in f:
                        s_list.append(np.asarray(frame.get_read_array(p)))
                        s_list[-1].flags.writeable = not lock_source_array # Lock the source data, making it read-only

                    fs = numpy_function(*s_list, **fun_args)

                    d = np.asarray(fout.get_write_array(p))
                    np.copyto(d, fs)
                    del d
            else:
                s_list = []
                for frame in f:
                    plane_list = []
                    for p in range(planes):
                        arr = np.asarray(frame.get_read_array(p))
                        plane_list.append(arr)
                    s_list.append(np.stack(plane_list, axis=2))
                    s_list[-1].flags.writeable = not lock_source_array # Lock the source data, making it read-only

                fs = numpy_function(*s_list, **fun_args)

                for p in range(planes):
                    d = np.asarray(fout.get_write_array(p))
                    np.copyto(d, fs[:, :, p])
                    del d
            return fout

    if not isinstance(clips, list):
        clips = [clips]

    flt = core.std.ModifyFrame(clips[0], clips, FLT)

    return flt


def numpy_process_val(clip, numpy_function, props_name, per_plane=True, lock_source_array=True, **fun_args):
    """Helper function for filtering clip in NumPy

    Args:
        clip: Input cilp.

        numpy_function: Spatial function to process numpy data. The output of the function should be single or multiple values.
            The format of the data provided to the function is "HWC", 
            i.e. number of pixels in vertical(height), horizontal(width) dimension and channels respectively.

        props_name: The name of property to be stored in each frame. It should be a list of strings.

        per_plane: (bool) Whether to process data by plane. If not, data would be processed by frame.
            Default is True.

        lock_source_array: (bool) Whether to lock the source array to avoid unintentionally overwrite the data.
            Default is True.

        fun_args: (dict) Additional arguments passed to “numpy_function” in the form of keyword arguments.
            Default is {}.

    """


    # The following code is modified from https://github.com/KotoriCANOE/MyTF/blob/master/utils/vshelper.py
    def FLT(n, f):
        fout = f.copy()
        planes = fout.format.num_planes

        val = []
        if per_plane:
            for p in range(planes):
                s = np.asarray(f.get_read_array(p))
                if lock_source_array:
                    s.flags.writeable = False # Lock the source data, making it read-only

                val.append(numpy_function(s, **fun_args))
        else:
            s_list = []
            for p in range(planes):
                arr = np.asarray(f.get_read_array(p)) # This is a 2-D array
                s_list.append(arr)
            s = np.stack(s_list, axis=2) # "s" is a 3-D array
            if lock_source_array:
                s.flags.writeable = False # Lock the source data, making it read-only

            val.append(numpy_function(s, **fun_args))

        for i, j in enumerate(val):
            fout.props[props_name[i]] = j

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
    clip = numpy_process(clip, functools.partial(L0Smooth_core, lamda=lamda, kappa=kappa, Denormin2=Denormin2), per_plane=per_plane, copy=True)

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
        src: 2-D or 3-D numpy array in the form of "HWC". The length along the second dimension must be even.
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
        gamma=gamma, eta=eta, Lap=Lap), per_plane=per_plane, copy=True)

    # Crop the padding
    if pad:
        clip = core.std.CropRel(clip, left=1)

    # Convert the bit depth and sample type of output to the same as input
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def L0GradProj_core(src, alpha=0.08, precision=255, epsilon=0.0002, maxiter=5000, gamma=3, eta=0.95, Lap=None, copy=False):
    """L0 Gradient Projection in NumPy.

    Args:
        src: 2-D or 3-D numpy array in the form of "HWC". The length along the second dimension must be even.
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
    I = np.argpartition(-sumDu.reshape(-1), epsilon-1)[:epsilon]
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


def IEDD(clip, blockSize=8, K=49, iteration=3):
    """IEDD in VapourSynth

    IEDD (Iterative Estimation in DCT Domain) is a method of blind estimation of white Gaussian noise variance in highly textured images.
    For a spatially correlated noise it is unusable.

    An input image is divided into 8x8 blocks and discrete cosine transform (DCT) is performed for each block.
    A part of 64 DCT coefficients with lowest energy calculated through all blocks is selected for further analysis.
    For the DCT coefficients, a robust estimate of noise variance is calculated.
    Corresponding to the obtained estimate, a part of blocks having very large values of local variance
    calculated only for the selected DCT coefficients are excluded from the further analysis.
    These two steps (estimation of noise variance and exclusion of blocks) are iteratively repeated three times.
    On the new noise-free test image database TAMPERE17,
    the method provides approximately two times lower estimation root mean square error than other methods.

    The result of each plane will be stored as frame property 'IEDD_AWGN_variance_{i}' in the output clip, where "i" stands for the index of plane.

    Args:
        clip: Input clip with no chroma subsampling.

        blockSize: (int) The side length of of block. Default is 8.

        K: (int) Number of DCT coefficients with lowest energy to be calculated.
            Lower value of K provides better robustness to a presence of textures.
            Higher value of K provides better accuracy of noise variance estimates.
            Default is 49.

        iteration: (int) Number of iterations. Default is 3.

    Ref:
        [1] Ponomarenko, M., Gapon, N., Voronin, V., & Egiazarian, K (2018). Blind estimation of white Gaussian noise variance in highly textured images. Image Processing: Algorithms and Systems (p. 5)
        [2] http://ponomarenko.info/iedd.html

    TODO: Optimize DCT using pyfftw library.

    """

    funcName = 'IEDD'

    if not isinstance(clip, vs.VideoNode) or any((clip.format.subsampling_w, clip.format.subsampling_h)):
        raise TypeError(funcName + ': \"clip\" must be a clip with no chroma subsampling!')

    props_name = ['IEDD_AWGN_variance_{}'.format(i) for i in range(clip.format.num_planes)]

    clip = numpy_process_val(clip, functools.partial(IEDD_core, blockSize=blockSize, K=K, iteration=iteration), props_name, per_plane=True)

    return clip


def IEDD_core(src, blockSize=8, K=49, iteration=3):
    """IEDD in NumPy

    IEDD is a method of blind estimation of white Gaussian noise variance in highly textured images.
    
    Args:
        src: 2-D numpy array in the form of "HW".

        blockSize: (int) The side length of of block. Default is 8.

        K: (int) Number of DCT coefficients with lowest energy to be calculated. Default is 49.

        iteration: (int) Number of iterations. Default is 3.

    TODO: Optimize DCT using pyfftw library.

    """

    from scipy.fftpack import dct

    funcName = 'IEDD_core'

    if not isinstance(src, np.ndarray) or src.ndim != 2:
        raise TypeError(funcName + ': \"src\" must be 2-D numpy data!')


    # copied from https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python/30110497#30110497
    def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
        # Parameters
        M,N = A.shape
        col_extent = N - BSZ[1] + 1
        row_extent = M - BSZ[0] + 1

        # Get Starting block indices
        start_idx = np.arange(BSZ[0])[:, np.newaxis] * N + np.arange(BSZ[1])

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, np.newaxis] * N + np.arange(col_extent)

        # Get all actual indices & index into input array for final output
        return np.take(A,start_idx.ravel()[:, np.newaxis] + offset_idx.ravel()[::stepsize])


    def mymad(d):
        d = d.flatten()
        m = np.median(d)
        return np.median(np.abs(d - m)) * 1.4826

    # function dctm
    blks = im2col_sliding_broadcasting(src.T.astype('float64', copy=False), [blockSize, blockSize])
    T = dct(np.eye(blockSize), axis=0, norm='ortho')
    blks = np.kron(T, T).dot(blks)

    ene = np.sum(blks ** 2, axis=1)
    m2 = np.argsort(ene)
    m1 = ene[m2]
    pz = np.nonzero(m2 == blockSize * blockSize - 1)[0]
    m2 = m2[:K]
    if pz < K and m1[pz] < m1[0] * 1.3:
        m2[pz] = m2[0]
        m2[0] = blockSize * blockSize - 1

    m = mymad(blks[m2[0]])

    for i in range(iteration):
        z = blks[m2]
        y = np.mean(z ** 2, axis=0)
        mp = y < (1 + np.sqrt(blockSize / K)) * m ** 2
        if np.count_nonzero(mp) > (blockSize * 4) ** 2:
            m = mymad(z[:1, mp])

    variance_estimate = m ** 2

    return variance_estimate


def DNCNN(clip, symbol_path, params_path, patch_size=[640, 640], device_id=0, **depth_args):
    """DnCNN in NumPy

    DnCNN is a deep convolutional neural network for image denoising.
    It can handel blind gaussian denoising or even general image denoising tasks.

    It's much slower than its C++ counterpart, and the GPU memory consumption is high. (See https://github.com/kice/vs_mxDnCNN)

    All the internal calculations are done at 32-bit float.

    Requirment: MXNet, pre-trained models.

    Args:
        clip: Input YUV clip with no chroma subsampling.

        symbol_path, params_path: Path to the model and params.

        patch_size: ([int, int]) The horizontal block size for dividing the image during processing.
            Smaller value results in lower VRAM usage or possibly border distortion, while larger value may not necessarily give faster speed.
            Default is [640, 640].

        device_id: (int) Which device to use. Starting with 0. If it is smaller than 0, CPU will be used.
            Default is 0.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    """

    import mxnet as mx
    from collections import namedtuple
    
    if any([clip.format.subsampling_w, clip.format.subsampling_h]):
        raise TypeError('Invalid type')

    # Load the model
    ctx = mx.gpu(device_id) if device_id >= 0 else mx.cpu()
    model = mx.mod.Module(mx.symbol.load(symbol_path), context=ctx, data_names=['data'])
    param = mx.nd.load(params_path)

    arg_param = {}
    aux_param = {}

    for k, v in param.items():
        if k.find("arg") != -1:
            arg_param[k.split(":")[1]] = v
        if k.find("aux") != -1:
            aux_param[k.split(":")[1]] = v

    model.bind(data_shapes=[('data', [1, 3, *patch_size])], for_training=False)
    model.set_params(arg_params=arg_param, aux_params=aux_param)


    # Execute
    def DNCNN_core(img, model):
        img = img.copy()
        img[:, :, 1:] += 0.5
        Batch = namedtuple('Batch', ['data'])

        data = mx.nd.expand_dims(mx.nd.array(img), axis=0)
        data = mx.nd.transpose(data, axes=(0, 3, 1, 2)).astype('float32')
        pred = mx.nd.empty(data.shape)

        for i in range(math.ceil(data.shape[2]/patch_size[0])):
            for j in range(math.ceil(data.shape[3]/patch_size[1])):
                model.forward(data_batch=Batch([data[:, :, i*patch_size[0]:min((i+1)*patch_size[0], img.shape[0]), j*patch_size[1]:min((j+1)*patch_size[1], img.shape[1])].copy()]), is_train=False)
                pred[:, :, i*patch_size[0]:min((i+1)*patch_size[0], img.shape[0]), j*patch_size[1]:min((j+1)*patch_size[1], img.shape[1])] = model.get_outputs()[0]

        pred = mx.nd.transpose(pred, axes=(0, 2, 3, 1)).reshape(img.shape).asnumpy()

        output = img - pred
        
        output[:, :, 1:] -= 0.5

        return output

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)
    clip = numpy_process(clip, DNCNN_core, per_plane=False, model=model) # Forward
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def get_blockwise_view(input_2D, block_size=8, strides=1, writeable=False):
    """Get block-wise view of an 2-D array.

    Args:
        input_2D: 2-D array.

        block_size: (int or [int, int]) The size of the block. It can be a single integer, which means the block is a square, 
            or a list of two integers specifying the height and width of the block respectively.
            Default is 8.

        strides: (int or [int, int]) The stride between the blocks. The format is similar to "patch_size".
            Default is 1.
        
        writeable: (bool) If set to False, the returned array will always be readonly.
            Otherwise it will be writable if the original array was. It is advisable to set this to False if possible.
            Default is False.

    """

    w, h = input_2D.shape

    if isinstance(block_size, int):
        block_size = [block_size]

    block_size_h = block_size[0]
    block_size_v = block_size[-1]

    if isinstance(strides, int):
        strides = [strides]

    strides_h = strides[0]
    strides_v = strides[-1]

    # assert(not any([(w-block_size_h) % strides_h, (h-block_size_v) % strides_v]))
    return as_strided(input_2D, shape=[(w-block_size_h)//strides_h+1, (h-block_size_v)//strides_v+1, block_size_h, block_size_v], 
                    strides=(input_2D.strides[0]*strides_h, input_2D.strides[1]*strides_v, *input_2D.strides), writeable=writeable)


def BNNMDenoise(clip, lamda=0.01, block_size=8, **depth_args):
    """Block-wise nuclear norm ninimization (NNM) based denoiser in VapourSynth

    Nuclear norm minimization methods is one category of low rank matrix approximation methods.

    This function minimize the following energy function given noisy patch Y:
        E(X) = ||Y - X||_{F}^{2} + λ||X||_{*}
    where F stands for Frobenius norm, * is the nuclear norm, i.e. sum of the singular values.

    It has been proved in [2] that such NNM based low rank matrix approximation problem with F-norm data fidelity
    can be solved by a soft-thresholding operation on the singular values of observation matrix.

    All the internal calculations are done at 32-bit float. Each plane is processed separately.

    Args:
        clip: Input clip.

        lamda: (float) The strength of the denoiser.
            Default is  0.01

        block_size: (int or [int, int]) The size of the block. It can be a single integer, which means the block is a square, 
            or a list of two integers specifying the height and width of the block respectively.
            Default is 8.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Gu, S., Xie, Q., Meng, D., Zuo, W., Feng, X., & Zhang, L. (2017). Weighted nuclear norm minimization and its applications to low level vision. International journal of computer vision, 121(2), 183-208.
        [2] Cai, J. F., Candès, E. J., & Shen, Z. (2010). A singular value thresholding algorithm for matrix completion. SIAM Journal on Optimization, 20(4), 1956-1982.

    """

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)
    clip = numpy_process(clip, functools.partial(BNNMDenoise_core, block_size=block_size, lamda=lamda), per_plane=True, copy=True)
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def BNNMDenoise_core(input_2D, block_size=8, lamda=0.01, copy=False):
    """Block-wise nuclear norm ninimization (NNM) based denoiser in NumPy

    This function minimize the following energy function given noisy patch Y:
        E(X) = ||Y - X||_{F}^{2} + λ||X||_{*}

    For detailed documentation, please refer to the documentation of "BNNMDenoise" funcion in current library.

    """

    if copy:
        output_2D = input_2D.copy()
    else:
        output_2D = input_2D

    block_view = get_blockwise_view(output_2D, block_size=block_size, strides=block_size, writeable=True) # No overlapping

    # Soft-thresholding
    u, s, v = np.linalg.svd(block_view, full_matrices=False, compute_uv=True)
    s[:] = np.maximum(s - lamda / 2, 0.)
    block_view[:] = u * s[:, :, np.newaxis, :] @ v

    return output_2D


def FGS(clip, ref=None, sigma=0.03, lamda=100, solver_iteration=3, solver_attenuation=4, **depth_args):
    """Fast Global Smoothing in VapourSynth

    Fast global smoother is a spatially inhomogeneous edge-preserving image smoothing technique, which has a comparable
    runtime to the fast edge-preserving filters, but its global optimization formulation overcomes many limitations of the
    local filtering approaches (halo, etc) and achieves high-quality results as the state-of-the-art optimization-based techniques.

    All the internal calculations are done at 32-bit float. Each plane is processed separately.

    The default parameters and the weight function is slightly dfferent from the original paper, in order to be similar to FDD().

    Args:
        clip: Input clip.

        ref: (clip) Reference clip used to compute the coefficients.
            It must has the same clip properties as 'input'.
            If it is None, it will be set to input.
            Default is None.

        sigma: (float or function) The standard deviation of the gaussian kernel defined on reference image.
            It can also be a function which takes two inputs and operates on vector in NumPy. The size of the output should be identical to the input.
            Default is 0.03.

        lamda: (float) The balance between the fidelity term and the regularization term.
            It can be treated as the strength of smoothing on the source image.
            Default is 100.

        solver_iterations: (int) Number of iterations to perform.
            Default is 3.

        solver_attenuation: (float) Attenuation factor for iteration.
            Default is 4.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Min, D., Choi, S., Lu, J., Ham, B., Sohn, K., & Do, M. N. (2014). Fast global image smoothing based on weighted least squares. IEEE Transactions on Image Processing, 23(12), 5638-5653.
        [2] https://sites.google.com/site/globalsmoothing/

    """

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    if ref is not None:
        ref = mvf.Depth(ref, depth=32, sample=vs.FLOAT, **depth_args)
        clip = [clip, ref]

    clip = numpy_process(clip, functools.partial(FGS_2D_core, sigma=sigma, lamda=lamda, 
        solver_iteration=solver_iteration, solver_attenuation=solver_attenuation), per_plane=True, copy=True)
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def FGS_2D_core(image_2D, joint_image_2D=None, sigma=0.03, lamda=100, solver_iteration=3, solver_attenuation=4, copy=False):
    """Fast Global Smoothing in NumPy

    Uncompleted. Only filtering on input image with one channel without guidance image is implemented.

    The default parameters and the weight function is slightly dfferent from the original paper, in order to be similar to FDD().

    For detailed documentation, please refer to the documentation of "FGS" funcion in current library.

    """

    from scipy.linalg import solve_banded

    if copy:
        image_2D = image_2D.copy()

    if joint_image_2D is None:
        joint_image_2D = image_2D

    h, w = image_2D.shape

    # variation of lamda
    lamda_in = 1.5 * lamda * 4**(solver_iteration - 1) / (4**solver_iteration - 1)

    # bilateral kernel weight
    if callable(sigma):
        BLF = sigma
    else:
        BLF = lambda x, y: np.exp(-(x - y) ** 2 / sigma)

    ab = np.empty((3, w * h), dtype=image_2D.dtype) # buffer

    for _ in range(solver_iteration):
        # for each row
        hflat = joint_image_2D.ravel(order='C')
        ab[0, 1:] = -lamda_in * BLF(hflat[1:], hflat[:-1])
        ab[0, ::w] = 0
        ab[2, :-1] = ab[0, 1:]
        ab[2, -1] = 0
        ab[1, :] = 1 - ab[0, :] - ab[2, :]

        tmp = image_2D.ravel(order='C')
        image_2D[:] = solve_banded((1, 1), ab, tmp, overwrite_ab=True, overwrite_b=True, check_finite=False).reshape((h, w), order='C')

        # for each column
        vflat = joint_image_2D.ravel(order='F')
        ab[0, 1:] = -lamda_in * BLF(vflat[1:], vflat[:-1])
        ab[0, ::h] = 0
        ab[2, :-1] = ab[0, 1:]
        ab[2, -1] = 0
        ab[1, :] = 1 - ab[0, :] - ab[2, :]

        tmp = image_2D.ravel(order='F')
        image_2D[:] = solve_banded((1, 1), ab, tmp, overwrite_ab=True, overwrite_b=True, check_finite=False).reshape((h, w), order='F')

        # variation of lamda
        lamda_in /= solver_attenuation

    return image_2D

    """
    # Old algorithms

    image_2D = image_2D.copy()
    joint_image_2D = image_2D#.copy()

    for _ in range(solver_iteration):
        # for each row
        for i in range(h):
            # a_vec
            ab_h[2, :-1] = -lamda_in * BLF(joint_image_2D[i, 1:] - joint_image_2D[i, :-1], sigma)
            ab_h[2, -1] = 0
            # c_vec
            ab_h[0, 0] = 0
            ab_h[0, 1:] = ab_h[2, :-1]
            # b_vec
            ab_h[1, :] = 1 - ab_h[0, :] - ab_h[2, :]

            tmp = image_2D[i, :].copy()
            # solve tridiagonal system
            image_2D[i, :] = solve_banded((1, 1), ab_h, tmp, overwrite_ab=True, overwrite_b=True, check_finite=False)

        # for each column
        for j in range(w):
            # a_vec
            ab_w[2, :-1] = -lamda_in * BLF(joint_image_2D[1:, j] - joint_image_2D[:-1, j], sigma)
            ab_w[2, -1] = 0
            # c_vec
            ab_w[0, 0] = 0
            ab_w[0, 1:] = ab_w[2, :-1]
            # b_vec
            ab_w[1, :] = 1 - ab_w[0, :] - ab_w[2, :]

            tmp = image_2D[:, j].copy()
            # solve tridiagonal system
            image_2D[:, j] = solve_banded((1, 1), ab_w, tmp, overwrite_ab=True, overwrite_b=True, check_finite=False)

        # variation of lamda
        lamda_in /= solver_attenuation

    return image_2D
    """


def FDD(clip, ref=None, sigma=0.03, lamda=100, beta=None, epsilon=1.2, solver_iteration=3, **depth_args):
    """Fast Domain Decomposition in VapourSynth

    Fast domain decomposition is a fast and linear time algorithm for global edge-preserving smoothing technique.
    It uses an efficient decomposition-based method to solve a sequence of 1-D sub-problems.
    An alternating direction method of multipliers algorithm is adopted to guarantee fast convergence.

    All the internal calculations are done at 32-bit float. Each plane is processed separately.

    The default parameters and the weight function is slightly dfferent from the original paper, in order to be similar to FGS().

    Args:
        clip: Input clip.

        ref: (clip) Reference clip used to compute the coefficients.
            It must has the same clip properties as 'input'.
            If it is None, it will be set to input.
            Default is None.

        sigma: (float or function) The standard deviation of the gaussian kernel defined on reference image.
            It can also be a function which takes two inputs and operates on vector in NumPy. The size of the output should be identical to the input.
            It is named as "kappa" in the paper.
            Default is 0.03.

        lamda: (float) The balance between the fidelity term and the regularization term.
            It can be treated as the strength of smoothing on the source image.
            Default is 100.

        beta: (float) Penalty parameter of augmented Lagrangian method. It will be increase during iteration by a factor of "epsilon".
            If it is None, it will be automatically set as sqrt(lamda)/2.
            Default is None.

        epsilon: (float) Multiplier of "beta" of each iteration.
            Default is 1.2.

        solver_iterations: (int) Number of iterations to perform.
            Default is 3.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Kim, Y., Min, D., Ham, B., & Sohn, K. (2017). Fast Domain Decomposition for Global Image Smoothing. IEEE Transactions on Image Processing.

    """

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    if beta is None:
        beta = math.sqrt(lamda) / 2

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    if ref is not None:
        ref = mvf.Depth(ref, depth=32, sample=vs.FLOAT, **depth_args)
        clip = [clip, ref]

    clip = numpy_process(clip, functools.partial(FDD_2D_core, sigma=sigma, lamda=lamda, beta=beta, 
        epsilon=epsilon, solver_iteration=solver_iteration), per_plane=True, copy=True)
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def FDD_2D_core(image_2D, joint_image_2D=None, sigma=0.03, lamda=100, beta=None, epsilon=1.2, solver_iteration=3, copy=False):
    """Fast Domain Decomposition in NumPy

    Uncompleted. Only filtering on input image with one channel without guidance image is implemented.

    The default parameters and the weight function is slightly dfferent from the original paper, in order to be similar to FGS().

    For detailed documentation, please refer to the documentation of "FDD" funcion in current library.

    """

    from scipy.linalg import solve_banded

    if copy:
        image_2D = image_2D.copy()

    if joint_image_2D is None:
        joint_image_2D = image_2D

    h, w = image_2D.shape

    if callable(sigma):
        BLF = sigma
    else:
        BLF = lambda x, y: np.exp(-(x - y) ** 2 / sigma)

    ab = np.empty((3, w * h), dtype=image_2D.dtype) # buffer

    alpha = 1
    v_pre = image_2D.copy()
    v_curr = np.empty_like(image_2D)
    v_hat_curr = image_2D.copy()
    u = np.empty_like(image_2D)
    gamma_pre = np.zeros_like(image_2D)
    gamma_curr = np.empty_like(image_2D)
    gamma_hat_curr = np.zeros_like(image_2D)
    f = np.empty_like(image_2D)

    for _ in range(solver_iteration):
        # for each row
        f[:] = (1 / (1 + beta)) * (image_2D + beta * (v_hat_curr + gamma_hat_curr))
        hflat = joint_image_2D.ravel(order='C')
        ab[0, 1:] = -2 * lamda / (1 + beta) * BLF(hflat[1:], hflat[:-1])
        ab[0, ::w] = 0
        ab[2, :-1] = ab[0, 1:]
        ab[2, -1] = 0
        ab[1, :] = 1 - ab[0, :] - ab[2, :]
        u[:] = solve_banded((1, 1), ab, f.ravel(order='C'), overwrite_ab=True, overwrite_b=True, check_finite=False).reshape((h, w), order='C')

        if _ == solver_iteration - 1:
            return u

        # for each column
        f[:] = (1 / (1 + beta)) * (image_2D + beta * (u - gamma_hat_curr))
        vflat = joint_image_2D.ravel(order='F')
        ab[0, 1:] = -2 * lamda / (1 + beta) * BLF(vflat[1:], vflat[:-1])
        ab[0, ::h] = 0
        ab[2, :-1] = ab[0, 1:]
        ab[2, -1] = 0
        ab[1, :] = 1 - ab[0, :] - ab[2, :]
        v_curr[:] = solve_banded((1, 1), ab, f.ravel(order='F'), overwrite_ab=True, overwrite_b=True, check_finite=False).reshape((h, w), order='F')

        # update parameters
        gamma_curr[:] = gamma_hat_curr - (u - v_curr)
        beta *= epsilon
        alpha = (1 + math.sqrt(1 + 4 * alpha ** 2)) / 2
        gamma_hat_curr[:] = gamma_curr + (alpha - 1) / alpha * (gamma_curr - gamma_pre)
        v_hat_curr[:] = v_curr + (alpha - 1) / alpha * (v_curr - v_pre)

        v_pre[:] = v_curr
        gamma_pre[:] = gamma_curr

    # return u


def SSFDeband(clip, thr=1, smooth_taps=2, edge_taps=3, strides=3, auto_scale_thr=True):
    """Selective sparse filter debanding in VapourSynth

    Deband using a selective sparse filter which combines smooth region detection and banding reduction.

    All the internal calculations are done at 32-bit float. Each plane is processed separately.

    Args:
        clip: Input clip.

        thr: (int) Threshold of banding detection.
            Default is 1.

        smooth_taps: (int) Taps of the sparse filter, the larger, the smoother.
            Default is 2.

        edge_taps: (int) Taps of the edge detector, the larger, smaller region will be smoothed.
            Default is 3.

        strides: (int) The stride of the sliding window.
            Default is 3.

        auto_scale_thr: (bool) Whether to automatically scale the "thr" according to the bit depth and sample type.
            Default is True.

    Ref:
        [1] Song, Q., Su, G. M., & Cosman, P. C. (2016, September). Hardware-efficient debanding and visual enhancement filter for inverse tone mapped high dynamic range images and videos. In Image Processing (ICIP), 2016 IEEE International Conference on (pp. 3299-3303). IEEE.

    """

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    if auto_scale_thr:
        if sampleType == vs.INTEGER:
            thr *= ((1 << bits) - 1) / 255
        else: # sampleType == vs.FLOAT
            thr /= (2 ** bits) - 1

    clip = numpy_process(clip, functools.partial(SSFDeband_core, thr=thr, smooth_taps=smooth_taps, 
        edge_taps=edge_taps, strides=strides), per_plane=True, copy=True)

    return clip


def SSFDeband_core(img_2D, thr=1, smooth_taps=2, edge_taps=3, strides=3, copy=False):
    """Selective sparse filter debanding in NumPy

    For detailed documentation, please refer to the documentation of "SSFDeband" funcion in current library.

    """

    img_2D_dtype = img_2D.dtype

    if copy or img_2D_dtype not in [np.float16, np.float32, np.float64]:
        img_2D = img_2D.astype('float32')

    isclose = lambda x, y, thr: np.abs(x - y) < thr

    h, w = img_2D.shape
    max_taps = max(smooth_taps, edge_taps)
    buff = strides * max_taps
    smooth_buff = strides * max(0, edge_taps - smooth_taps)
    edge_buff = strides * max(0, smooth_taps - edge_taps)

    cropped = img_2D[buff:-buff, :]
    # v_mask = isclose((img_2D[:-18, :], cropped) & isclose(img_2D[3:-15, :], cropped) & isclose(img_2D[6:-12, :], cropped) & isclose(img_2D[12:-6, :], cropped) & isclose(img_2D[15:-3, :], cropped) & isclose(img_2D[18:, :], cropped), atol=thr, rtol=0)
    upper_view = as_strided(img_2D[edge_buff:, :], shape=[h-2*buff, w, edge_taps], strides=(*img_2D.strides, strides*img_2D.strides[0]))
    upper_mask = np.logical_and.reduce(isclose(upper_view, cropped[..., np.newaxis], thr=thr), axis=2)
    lower_view = as_strided(img_2D[buff+strides:, :], shape=[h-2*buff, w, edge_taps], strides=(*img_2D.strides, strides*img_2D.strides[0]))
    lower_mask = np.logical_and.reduce(isclose(lower_view, cropped[..., np.newaxis], thr=thr), axis=2)
    v_mask = upper_mask & lower_mask
    # v_smooth = (img_2D[3:-15, :] + img_2D[6:-12, :] + img_2D[9:-9, :] + img_2D[12:-6, :] + img_2D[15:-3, :]) / 5
    v_smooth = as_strided(img_2D[smooth_buff:, :], shape=[h-2*buff, w, 2*smooth_taps+1], strides=(*img_2D.strides, strides*img_2D.strides[0])).mean(axis=2)
    cropped[:] = np.where(v_mask, v_smooth, cropped)

    cropped = img_2D[:, buff:-buff]
    # h_mask = isclose((img_2D[:, :-18], cropped) & isclose(img_2D[:, 3:-15], cropped) & isclose(img_2D[:, 6:-12], cropped) & isclose(img_2D[:, 12:-6], cropped) & isclose(img_2D[:, 15:-3], cropped) & isclose(img_2D[:, 18:], cropped, atol=thr, rtol=0)
    left_view = as_strided(img_2D[:, edge_buff:], shape=[h, w-2*buff, edge_taps], strides=(*img_2D.strides, strides*img_2D.strides[1]))
    left_mask = np.logical_and.reduce(isclose(left_view, cropped[..., np.newaxis], thr=thr), axis=2)
    right_view = as_strided(img_2D[:, buff+strides:], shape=[h, w-2*buff, edge_taps], strides=(*img_2D.strides, strides*img_2D.strides[1]))
    right_mask = np.logical_and.reduce(isclose(right_view, cropped[..., np.newaxis], thr=thr), axis=2)
    h_mask = left_mask & right_mask
    # h_smooth = (img_2D[:, 3:-15] + img_2D[:, 6:-12] + img_2D[:, 9:-9] + img_2D[:, 12:-6] + img_2D[:, 15:-3]) / 5
    h_smooth = as_strided(img_2D[:, smooth_buff:], shape=[h, w-2*buff, 2*smooth_taps+1], strides=(*img_2D.strides, strides*img_2D.strides[1])).mean(axis=2)
    cropped[:] = np.where(h_mask, h_smooth, cropped)

    return img_2D.astype(img_2D_dtype, copy=False)


def SigmaFilter(clip, radius=3, thr=0.01, **depth_args):
    """Sigma filter in VapourSynth

    Sigma filter is a local smoothing operator which replace the pixel to be processed 
    by the average of only those neighboring pixels having their intensity within a 
    fixed sigma range of the center pixel.

    The special step to handle sharp spot noise described in (4) in the paper is ignored
    for better detail preservation and lower computation.

    All the internal calculations are done at 32-bit float. Each plane is processed separately

    Args:
        clip: Input clip.

        radius: (int) Radius of the filtering window.
            Default is 3.

        thr: (int) Threshold of pixel selection.
            Default is 0.01.

    Ref:
        [1] Lee, J. S. (1983). Digital image smoothing and the sigma filter. Computer vision, graphics, and image processing, 24(2), 255-269.

    """

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)
    clip = numpy_process(clip, functools.partial(SigmaFilter_core, radius=radius, thr=thr), per_plane=True)
    clip = mvf.Depth(clip, depth=bits, sample=sampleType, **depth_args)

    return clip


def SigmaFilter_core(img_2D, radius=3, thr=0.01):
    """Sigma filter in NumPy

    The special step to handle sharp spot noise described in (4) in the paper is ignored 
    for better detail preservation and lower computation.

    For detailed documentation, please refer to the documentation of "SigmaFilter" funcion in current library.

    """

    pad_img = np.pad(img_2D, pad_width=radius, mode='constant')
    # img_view = as_strided(pad_img, shape=(*img_2D.shape, 2*radius+1, 2*radius+1), strides=pad_img.strides*2)
    img_view = get_blockwise_view(pad_img, block_size=2*radius+1, strides=1)
    select = np.where(np.absolute(img_view - img_2D[..., np.newaxis, np.newaxis]) < thr, img_view, 0) # Choose pixels inside the intensity range
    flt = np.sum(select, axis=(2, 3)) / (np.count_nonzero(select, axis=(2, 3)) + 1e-7) # Compute the average

    return flt
