"""Reimplementation of KNLMeansCL(Non-local Means) in CuPy"""

from string import Template

import cupy as cp
import vapoursynth as vs
from vapoursynth import core
import muvsfunc_numpy as mufnp


# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of KNLMeansCL. Documentation: https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description
# d = 0 # only spatial processing is implemented.
a = 2
s = 4
h = 1.2
channels = 'Y' # only grayscale filtering is implemented
wmode = 0
wref = 1.0
rclip = None # not implemented
ocl_x = 16 # local work group width of the separable convolution kernel
ocl_y = 8 # local work group height of the separable convolution kernel
ocl_r = 3 # number of processed pixel for work-item

# whether to enable '--use_fast_math' in NVRTC runtime compilation
# to make use of fast math operations
fast = False


# pre-processing
if src.format.id != vs.GRAYS:
    raise TypeError('Only 32-bit float grayscale input is supported!')

# CUDA kernel execution configuration
dst_block = (32, 32, 1) # used by 'nlmDistance'
hrz_block = (ocl_x, ocl_y, 1) # used by 'nlmHorizontal'
vrt_block = (ocl_x, ocl_y, 1) # used by 'nlmVertical'
work_block = (32, 32, 1) # used by 'nlmAccumulation' and 'nlmFinish'

# load CUDA kernel
with open('knlm.cu', 'r') as f:
    kernel_source_code = f.read()

kernel_source_code = Template(kernel_source_code)
kernel_source_code = kernel_source_code.substitute(
    width=src.width, height=src.height, s=s, h=h, wmode=wmode, wref=wref, 
    hrz_block_x=ocl_x, hrz_block_y=ocl_y, hrz_result=ocl_r, 
    vrt_block_x=ocl_x, vrt_block_y=ocl_y, vrt_result=ocl_r)

if fast:
    nlmDistance = cp.RawKernel(kernel_source_code, 'nlmDistance', 
        options=('--use_fast_math', ))
    nlmHorizontal = cp.RawKernel(kernel_source_code, 'nlmHorizontal', 
        options=('--use_fast_math', ))
    nlmVertical = cp.RawKernel(kernel_source_code, 'nlmVertical', 
        options=('--use_fast_math', ))
    nlmAccumulation = cp.RawKernel(kernel_source_code, 'nlmAccumulation', 
        options=('--use_fast_math', ))
    nlmFinish = cp.RawKernel(kernel_source_code, 'nlmFinish', 
        options=('--use_fast_math', ))
else:
    nlmDistance = cp.RawKernel(kernel_source_code, 'nlmDistance')
    nlmHorizontal = cp.RawKernel(kernel_source_code, 'nlmHorizontal')
    nlmVertical = cp.RawKernel(kernel_source_code, 'nlmVertical')
    nlmAccumulation = cp.RawKernel(kernel_source_code, 'nlmAccumulation')
    nlmFinish = cp.RawKernel(kernel_source_code, 'nlmFinish')


# create NumPy function
def nlm_core(h_img, a, nlmDistance, nlmHorizontal, nlmVertical, nlmAccumulation, nlmFinish):
    U1a = cp.asarray(h_img)
    h, w = U1a.shape

    U2a = cp.zeros_like(U1a)
    U2b = cp.zeros_like(U1a)
    U1z = cp.empty_like(U1a)
    U4a = cp.empty_like(U1a)
    U4b = cp.empty_like(U1a)
    U5 = cp.full_like(U1a, fill_value=1.1920928955078125e-7) # CL_FLT_EPSILON

    # Spatial processing
    for j in range(-a, a+1):
        for i in range(-a, a+1):
            if (j * (2 * a + 1) + i < 0):
                nlmDistance(((w + dst_block[0] - 1) // dst_block[0], (h + dst_block[1] - 1) // dst_block[1], 1), dst_block, (U1a, U4a, i, j))
                nlmHorizontal(((w + hrz_block[0] - 1) // hrz_block[0], (h + hrz_block[1] - 1) // hrz_block[1], 1), hrz_block, (U4a, U4b))
                nlmVertical(((w + vrt_block[0] - 1) // vrt_block[0], (h + vrt_block[1] - 1) // vrt_block[1]), vrt_block, (U4b, U4a))
                nlmAccumulation(((w + work_block[0] - 1) // work_block[0], (h + work_block[1] - 1) // work_block[1]), work_block, (U1a, U2a, U2b, U4a, U5, i, j))

    nlmFinish(((w + work_block[0] - 1) // work_block[0], (h + work_block[1] - 1) // work_block[1]), work_block, (U1a, U1z, U2a, U2b, U5))

    h_out = cp.asnumpy(U1z)

    return h_out


# process
res = mufnp.numpy_process(
    src, nlm_core, a=a, 
    nlmDistance=nlmDistance, nlmHorizontal=nlmHorizontal, 
    nlmVertical=nlmVertical, nlmAccumulation=nlmAccumulation, 
    nlmFinish=nlmFinish)

"""
res = core.knlm.KNLMeansCL(
    src, d=0, a=a, s=s, h=h, channels='Y', wmode=wmode, rclip=None, 
    device_type='GPU', ocl_x=ocl_x, ocl_y=ocl_y, ocl_r=ocl_r, info=False)
"""

res.set_output()
