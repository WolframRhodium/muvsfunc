// original OpenCL implementation: https://github.com/Khanattila/KNLMeansCL/blob/27f95992e2344586b745d013eafa010764c78979/KNLMeansCL/NLMKernel.cpp#L67-L406

#define VI_DIM_X ${width}
#define VI_DIM_Y ${height}

#define NLM_S ${s}
#define NLM_H ((float) ${h})
#define NLM_WMODE ${wmode}
#define NLM_WREF ((float) ${wref})

#define NLM_NORM (255.0f * 255.0f)
#define NLM_LEGACY 3.0f
#define NLM_S_SIZE ((2 * NLM_S + 1) * (2 * NLM_S + 1))
#define NLM_H2_INV_NORM (NLM_NORM / (NLM_LEGACY * NLM_H * NLM_H * NLM_S_SIZE))

#define HRZ_BLOCK_X ${hrz_block_x}
#define HRZ_BLOCK_Y ${hrz_block_y}
#define HRZ_RESULT ${hrz_result}
#define VRT_BLOCK_X ${vrt_block_x}
#define VRT_BLOCK_Y ${vrt_block_y}
#define VRT_RESULT ${vrt_result}

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#define CLAMPX(x) (MIN(MAX(x, 0), VI_DIM_X - 1))
#define CLAMPY(y) (MIN(MAX(y, 0), VI_DIM_Y - 1))

extern "C" __global__
void nlmDistance(const float * __restrict__ U1, float * __restrict__ U4a, 
    const int qx, const int qy) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= VI_DIM_X || y >= VI_DIM_Y)
        return;

    const int gidx = y * VI_DIM_X + x;

    // #if defined(NLM_CLIP_REF_LUMA)
    const float u1 = U1[gidx];
    const float u1_pq = U1[CLAMPY(y + qy) * VI_DIM_X + CLAMPX(x + qx)];

    const float val = 3.0f * ((u1 - u1_pq) * (u1 - u1_pq));
    // #endif

    U4a[gidx] = val;
}

extern "C" __global__
void nlmHorizontal(const float * __restrict__ U4a, float * __restrict__ U4b) {

    __shared__ float buffer[HRZ_BLOCK_Y][(HRZ_RESULT + 2) * HRZ_BLOCK_X];

    const int x = (blockIdx.x * HRZ_RESULT - 1) * HRZ_BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i <= 1 + HRZ_RESULT; i++)
        buffer[threadIdx.y][threadIdx.x + i * HRZ_BLOCK_X] = 
            U4a[y * VI_DIM_X + CLAMPX(x + i * HRZ_BLOCK_X)];

    __syncthreads();

    for (int i = 1; i <= HRZ_RESULT; i++) {
        if ((x + i * HRZ_BLOCK_X >= VI_DIM_X) || y >= VI_DIM_Y) 
            return;

        float sum = 0.0f;

        for (int j = -NLM_S; j <= NLM_S; j++)
            sum += buffer[threadIdx.y][threadIdx.x + i * HRZ_BLOCK_X + j];

        U4b[y * VI_DIM_X + (x + i * HRZ_BLOCK_X)] = sum; // (x + i * HRZ_BLOCK_X) >= 0
    }
}

extern "C" __global__
void nlmVertical(const float * __restrict__ U4b, float * __restrict__ U4a) {

    __shared__ float buffer[VRT_BLOCK_X][(VRT_RESULT + 2) * VRT_BLOCK_Y + 1];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = (blockIdx.y * VRT_RESULT - 1) * VRT_BLOCK_Y + threadIdx.y;

    for (int i = 0; i <= 1 + VRT_RESULT; i++)
        buffer[threadIdx.x][threadIdx.y + i * VRT_BLOCK_Y] = 
            U4b[CLAMPY(y + i * VRT_BLOCK_Y) * VI_DIM_X + x];

   __syncthreads();

   for (int i = 1; i <= HRZ_RESULT; i++) {
       if (x >= VI_DIM_X || (y + i * VRT_BLOCK_Y) >= VI_DIM_Y) 
            return;

        float sum = 0.0f;

        for (int j = -NLM_S; j <= NLM_S; j++)
            sum += buffer[threadIdx.x][threadIdx.y + i * VRT_BLOCK_Y + j];

#if NLM_WMODE == 0
        // #if defined(NLM_WMODE_WELSCH)
        const float val = __expf(-sum * NLM_H2_INV_NORM);
#elif NLM_WMODE == 1
        // #if defined(NLM_WMODE_BISQUARE_A)
        const float val = fdimf(1.0f, sum * NLM_H2_INV_NORM);
#elif NLM_WMODE == 2
        // #if defined(NLM_WMODE_BISQUARE_B)
        const float val = __powf(fdimf(1.0f, sum * NLM_H2_INV_NORM), 2.0f);
#elif NLM_WMODE == 3
        // #if defined(NLM_WMODE_BISQUARE_C)
        const float val = __powf(fdimf(1.0f, sum * NLM_H2_INV_NORM), 8.0f);
#endif

        U4a[(y + i * VRT_BLOCK_Y) * VI_DIM_X + x] = val; // (y + i * VRT_BLOCK_Y) >= 0
    }
}

extern "C" __global__
void nlmAccumulation(const float * __restrict__ U1a, float * __restrict__ U2a, 
    float * __restrict__ U2b, const float * __restrict__ U4a, float * __restrict__ U5, 
    const int qx, const int qy) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= VI_DIM_X || y >= VI_DIM_Y)
        return;

    const int gidx = y * VI_DIM_X + x;

    const float u4 = U4a[gidx];
    const float u4_mq = U4a[CLAMPY(y - qy) * VI_DIM_X + CLAMPX(x - qx)];
    U5[gidx] = fmaxf(u4, fmaxf(u4_mq, U5[gidx]));

    // #if (NLM_CHANNELS == 1)
    const float u1_pq = U1a[CLAMPY(y + qy) * VI_DIM_X + CLAMPX(x + qx)];
    const float u1_mq = U1a[CLAMPY(y - qy) * VI_DIM_X + CLAMPX(x - qx)];

    U2a[gidx] += (u4 * u1_pq) + (u4_mq * u1_mq);
    U2b[gidx] += (u4 + u4_mq);
    // #endif
}

extern "C" __global__
void nlmFinish(const float * __restrict__ U1a, float * __restrict__ U1z, 
    const float * __restrict__ U2a, const float * __restrict__ U2b, 
    const float * __restrict__ U5) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= VI_DIM_X || y >= VI_DIM_Y)
        return;

    const int gidx = y * VI_DIM_X + x;
    const float m = NLM_WREF * U5[gidx];

    // #if (NLM_CHANNELS == 1)
    const float u1 = U1a[gidx];
    const float u2a = U2a[gidx];
    const float u2b = U2b[gidx];

    const float den = m + u2b;
    const float val = (u1 * m + u2a) / den;

    U1z[gidx] = val;
    // #endif
}
