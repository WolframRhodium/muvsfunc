"""Sigma Filter in CuPy"""

from string import Template

import cupy as cp
import vapoursynth as vs
import muvsfunc_numpy as mufnp

core = vs.get_core()


# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of mufnp.SigmaFilter()
radius = 3
thr = 0.01


# pre-processing
if src.format.id != vs.GRAYS:
    raise TypeError('Only 32-bit float grayscale input is supported!')

# CUDA kernel execution configuration
blksize = (16, 8, 1)

# load CUDA kernel
with open("sigma_filter.cu", 'r') as f:
    kernel_source_code = f.read()

kernel_source_code = Template(kernel_source_code)
kernel_source_code = kernel_source_code.substitute(
    width=src.width, height=src.height, radius=radius, threshold=thr)

kernel = cp.RawKernel(kernel_source_code, 'sigmaFilter')


# create NumPy function
def sigma_filter_core(h_img):
    d_img = cp.asarray(h_img)
    h, w = d_img.shape

    d_out = cp.empty_like(d_img)

    kernel(((w + blksize[0] - 1)//blksize[0], (h + blksize[1] - 1)//blksize[1]), blksize, (d_img, d_out))

    h_out = cp.asnumpy(d_out)

    return h_out


# process
res = mufnp.numpy_process(src, sigma_filter_core)

# res = mufnp.SigmaFilter(src, radius=radius, thr=thr)

res.set_output()