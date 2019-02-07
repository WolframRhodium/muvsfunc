"""Bilateral-GPU in VapourSynth"""

from string import Template

import cupy as cp
import vapoursynth as vs
import muvsfunc_numpy as mufnp

core = vs.get_core()

# Load source clip. Only GRAYS is supported
src = core.std.BlankClip(format=vs.GRAYS)

# params of bilateral filter. See documentation at https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral
sigmaS = 3.0
sigmaR = 0.02
# params of SNN (Statistical Nearest Neighbors) sampling strategiy
# ref: I. Frosio, J. Kautz, Statistical Nearest Neighbors for Image Denoising, IEEE Trans. Image Processing, 2019.
sigma = 0 # 0.0003

# other params
half_kernel_size = round(sigmaS * 2)
blksize = (32, 8) # dimensions of the CUDA thread block

# whether to enable '--use_fast_math' in NVRTC runtime compilation
# to make use of fast math operations
fast = False

# pre-processing
snn = int(sigma > 0) # whether to use SNN sampling strategy

if src.format.id != vs.GRAYS:
    raise vs.Error("Bilateral: Only 32-bit float grayscale is supported!")

w, h = src.width, src.height

# source code of CUDA kernel
with open('bilateral.cu', 'r') as f:
    kernel_source_code = f.read()

kernel_source_code = Template(kernel_source_code)
kernel_source_code = kernel_source_code.substitute(
    width=w, height=h, sigma_s=-0.5/(sigmaS**2), sigma_r=-0.5/(sigmaR**2), 
    sigma=sigma, snn=snn, half_kernel_size=half_kernel_size)


if fast:
    kernel = cp.RawKernel(kernel_source_code, 'bilateral', 
        options=('--use_fast_math', ))
else:
    kernel = cp.RawKernel(kernel_source_code, 'bilateral')

# create NumPy function
def bilateral_core(h_img, kernel):
    # h_img must be a 2-D image

    d_img = cp.asarray(h_img)
    d_out = cp.empty_like(d_img)

    kernel(((w + blksize[0] - 1)//blksize[0], (h + blksize[1] - 1)//blksize[1]), blksize, (d_img, d_out))

    h_out = cp.asnumpy(d_out)

    return h_out

# process
res = mufnp.numpy_process(src, bilateral_core, kernel=kernel)

res.set_output()
