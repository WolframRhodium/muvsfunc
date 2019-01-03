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

    for (int cy = MAX(y - HALF_KERNEL_SIZE, 0); cy <= MIN(y + HALF_KERNEL_SIZE, HEIGHT - 1); ++cy)
        for (int cx = MAX(x - HALF_KERNEL_SIZE, 0); cx <= MIN(x + HALF_KERNEL_SIZE, WIDTH - 1); ++cx) {
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
