"""Sigma Filter in Numba"""

from numba import jit, prange

import vapoursynth as vs
from vapoursynth import core
import numpy as np
from functools import partial


# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of mufnp.SigmaFilter()
radius = 3
thr = 0.01


# pre-processing
if src.format.id != vs.GRAYS:
    raise TypeError('Only 32-bit float grayscale input is supported!')


_is_api4: bool = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4


def _get_array(frame, plane, read=True):
    if not read and frame.readonly:
        raise ValueError("Frame is readonly")

    if _is_api4:
        return frame[plane]
    else:
        if read:
            return frame.get_read_array(plane)
        else:
            return frame.get_write_array(plane)


def erase_module(func):
    """Erase '__module__' attribute of a user-defined function that breaks numba"""

    if hasattr(func, '__module__') and func.__module__ == '__vapoursynth__':
        func.__module__ = None

    return func


@jit(nopython=True, nogil=True)
@erase_module
def clamp(val, low, high):
    return min(max(val, low), high)


@jit(nopython=True, nogil=True, fastmath=True, parallel=False)
@erase_module
def sigma_filter(src, dst, radius, threshold):
    height = src.shape[0]
    width = src.shape[1]

    for y in prange(height):
        for x in range(width):
            center = src[y, x]
            acc = 0.
            count = 0

            for j in range(-radius, radius + 1):
                for i in range(-radius, radius + 1):
                    val = src[clamp(y + j, 0, height - 1), clamp(x + i, 0, width - 1)]

                    if abs(center - val) < threshold:
                        acc += val
                        count += 1

            dst[y, x] = acc / count

    return


def executor(n, f, radius, thr):
    fout = f.copy()

    src = np.asarray(_get_array(f, 0, read=True))
    dst = np.asarray(_get_array(fout, 0, read=False))

    sigma_filter(src, dst, radius, thr)

    return fout


# process
res = core.std.ModifyFrame(src, src, partial(executor, radius=radius, thr=thr))

# res = mufnp.SigmaFilter(src, radius=radius, thr=thr)

res.set_output()
