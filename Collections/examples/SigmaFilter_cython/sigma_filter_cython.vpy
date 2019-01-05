"""Sigma Filter in Cython"""

import vapoursynth as vs
from functools import partial

# To build, run `python setup.py build_ext --inplace`
from sigma_filter import sigma_filter

core = vs.get_core()


# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of mufnp.SigmaFilter()
radius = 3
thr = 0.01


# pre-processing
if src.format.id != vs.GRAYS:
    raise TypeError('Only 32-bit float grayscale input is supported!')


def executor(n, f, radius, thr):
    fout = f.copy()

    src = f.get_read_array(0)
    dst = fout.get_write_array(0)

    sigma_filter(src, dst, radius, thr)

    return fout

# process
res = core.std.ModifyFrame(src, src, partial(executor, radius=radius, thr=thr))

# res = mufnp.SigmaFilter(src, radius=radius, thr=thr)

res.set_output()