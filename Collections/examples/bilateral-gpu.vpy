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

# other params
kernel_size = round(sigmaS * 3) * 2 + 1
blksize = (32, 8) # dimensions of the CUDA thread block

# pre-processing
if src.format.id != vs.GRAYS:
    raise vs.Error("Bilateral: Only constant 32-bit float input is supported.")

w, h = src.width, src.height

# source code of CUDA kernel
_test_source = Template(r'''
// naive implementation of CUDA-accelerated Bilateral filter

// modified from
// https://github.com/opencv/opencv_contrib/blob/82733fe56b13401519ace101dc4d724f0a83f535/modules/cudaimgproc/perf/perf_bilateral_filter.cpp


#define WIDTH $width
#define HEIGHT $height
#define SIGMA_S (float) $sigma_s
#define SIGMA_R (float) $sigma_r
#define KERNEL_SIZE $kernel_size

extern "C"
__global__ void bilateral(const float * src, float * dst) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    float center = src[y * WIDTH + x];
    
    float sum1 = 0;
    float sum2 = 0;

    int r = KERNEL_SIZE / 2;

    int tx = x - r + KERNEL_SIZE;
    int ty = y - r + KERNEL_SIZE;

    for (int cy = max(y - r, 0); cy < min(ty, HEIGHT); ++cy)
        for (int cx = max(x - r, 0); cx < min(tx, WIDTH); ++cx) {
            float space = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            
            float value = src[cy * WIDTH + cx];

            float weight = __expf(space * SIGMA_S + (value - center) * (value - center) * SIGMA_R);

            sum1 += weight * value;
            sum2 += weight;
        }
    
    dst[y * WIDTH + x] = sum1 / sum2;
}
''')

_test_source = _test_source.substitute(width=w, height=h, sigma_s=-0.5/(sigmaS**2), sigma_r=-0.5/(sigmaR**2), kernel_size=kernel_size)


kernel = cp.RawKernel(_test_source, 'bilateral')

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
