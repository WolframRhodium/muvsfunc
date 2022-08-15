"""Sigma Filter in Cython"""

import vapoursynth as vs
from vapoursynth import core
from functools import partial

# To build, run `python setup.py build_ext --inplace`
from sigma_filter import sigma_filter

is_api4: bool = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4

def get_array(frame, plane, read=True):
    if not read and frame.readonly:
        raise ValueError("Frame is readonly")

    if is_api4:
        return frame[plane]
    else:
        if read:
            return frame.get_read_array(plane)
        else:
            return frame.get_write_array(plane)


# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of mufnp.SigmaFilter()
radius = 3
thr = 0.01

parallel = True


# pre-processing
if src.format.id != vs.GRAYS:
    raise TypeError('Only 32-bit float grayscale input is supported!')


def executor(n, f, radius, thr):
    fout = f.copy()

    src = get_array(f, 0)
    dst = get_array(fout, 0, read=False)

    sigma_filter(src, dst, radius, thr)

    return fout

selector = partial(executor, radius=radius, thr=thr)
# process
if parallel:
    res = core.std.FrameEval(src, lambda n: core.std.ModifyFrame(src, src, selector))
else:
    res = core.std.ModifyFrame(src, src, selector)

# res = mufnp.SigmaFilter(src, radius=radius, thr=thr)

res.set_output()
