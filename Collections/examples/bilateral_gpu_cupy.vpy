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
# ref: I. Frosio, J. Kautz, Statistical Neareast Neighbors for Image Denoising, IEEE Trans. Image Processing, 2018.
sigma = 0 # 0.0003

# other params
half_kernel_size = round(sigmaS * 2)
blksize = (32, 8) # dimensions of the CUDA thread block

# pre-processing
snn = int(sigma > 0) # whether to use SNN sampling strategy

if src.format.id != vs.GRAYS:
    raise vs.Error("Bilateral: Only constant 32-bit float input is supported.")

w, h = src.width, src.height

# source code of CUDA kernel
_test_source = Template(r'''
// naive implementation of CUDA-accelerated (NN/SNN) Bilateral filter

// modified from
// https://github.com/opencv/opencv_contrib/blob/82733fe56b13401519ace101dc4d724f0a83f535/modules/cudaimgproc/perf/perf_bilateral_filter.cpp


#define WIDTH $width
#define HEIGHT $height
#define SIGMA_S ${sigma_s}f
#define SIGMA_R ${sigma_r}f
#define SIGMA ${sigma}f
#define HALF_KERNEL_SIZE ${half_kernel_size}
#define SNN ${snn}

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

extern "C"
__global__ void bilateral(const float * __restrict__ src, float * __restrict__ dst) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    const float center = src[y * WIDTH + x];
    
    float sum1 = 0;
    float sum2 = 0;

    for (int cy = MAX(y - HALF_KERNEL_SIZE, 0); cy <= MIN(y + HALF_KERNEL_SIZE, HEIGHT); ++cy)
        for (int cx = MAX(x - HALF_KERNEL_SIZE, 0); cx <= MIN(x + HALF_KERNEL_SIZE, WIDTH); ++cx) {
            const float space = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            
            const float value = src[cy * WIDTH + cx];

#if SNN
            const float weight = __expf(space * SIGMA_S + 
                fabsf((value - center) * (value - center) - SIGMA) * SIGMA_R);
#else
            const float weight = __expf(space * SIGMA_S + (value - center) * (value - center) * SIGMA_R);
#endif

            sum1 += weight * value;
            sum2 += weight;
        }
    
    dst[y * WIDTH + x] = sum1 / sum2;
}
''')

_test_source = _test_source.substitute(width=w, height=h, sigma_s=-0.5/(sigmaS**2), sigma_r=-0.5/(sigmaR**2), 
    sigma=sigma, snn=snn, half_kernel_size=half_kernel_size)


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
