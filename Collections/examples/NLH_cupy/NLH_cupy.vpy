"""
Implementation of NLH (NLH: A Blind Pixel-level Non-local Method for Real-world Image Denoising) in CuPy

Ref:
    [1] Hou, Y., Xu, J., Liu, M., Liu, G., Liu, L., Zhu, F., & Shao, L. (2019). 
        NLH: A Blind Pixel-level Non-local Method for Real-world Image Denoising. 
        arXiv preprint arXiv:1906.06834.

"""

from string import Template

import cupy as cp
import vapoursynth as vs
from vapoursynth import core

import muvsfunc_numpy as mufnp


# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of NLH
# d = 0 # only spatial processing is implemented.
a = 2
s = 4
h = 1.6
h2 = 1.6

# whether to enable '--use_fast_math' in NVRTC runtime compilation
# to make use of fast math operations
fast = False

# CUDA kernel execution configuration
work_block = (16, 16, 1)


# pre-processing
if src.format.id != vs.GRAYS:
    raise TypeError('Only 32-bit float grayscale input is supported!')


# load CUDA kernel
with open('kernel.cu', 'r') as f:
    kernel_source_code = f.read()

kernel_source_code = Template(kernel_source_code)
kernel_source_code = kernel_source_code.substitute(
    width=src.width, height=src.height, a=a, s=s, h=h, h2=h2)

if fast:
    compute = cp.RawKernel(kernel_source_code, 'compute', options=('--use_fast_math', ))
else:
    compute = cp.RawKernel(kernel_source_code, 'compute')


# create NumPy function
def nlm_core(h_src, compute):
    d_src = cp.asarray(h_src)
    h, w = h_src.shape

    d_dst = cp.empty_like(d_src)

    compute(((w + work_block[0] - 1) // work_block[0], (h + work_block[1] - 1) // work_block[1]), work_block, (d_src, d_dst))

    h_out = cp.asnumpy(d_dst)

    return h_out

# process
res = mufnp.numpy_process(src, nlm_core, compute=compute)

# feisty2's CPU implementation (https://github.com/IFeelBloated/NLMeans-PM/tree/f2539968e3ded41588cc18b3c2a984f42b79e4a9)
# "ref" is not currently implemented on the CUDA version
# res = core.test.Test(src, a=a, s=s, h=h, h2=h2, sdev=1000000000)


res.set_output()
