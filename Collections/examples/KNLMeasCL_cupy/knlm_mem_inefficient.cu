// original OpenCL implementation: https://github.com/Khanattila/KNLMeansCL/blob/27f95992e2344586b745d013eafa010764c78979/KNLMeansCL/NLMKernel.cpp#L67-L406

#define WIDTH ${width}
#define HEIGHT ${height}

#define NLM_A ${a}
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

#define CLAMPX(x) (MIN(MAX(x, 0), WIDTH - 1))
#define CLAMPY(y) (MIN(MAX(y, 0), HEIGHT - 1))

#if __CUDACC_VER_MAJOR__ >= 9 // CUDA 9.0 or later
    #include <cooperative_groups.h>
    namespace cg = cooperative_groups;
#endif

extern "C" __global__
void nlmDistance(const float U1[HEIGHT][WIDTH], float U4a[NLM_A*2+1][NLM_A*2+1][HEIGHT][WIDTH]) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    for (int qy = -NLM_A; qy <= NLM_A; qy++)
        for (int qx = -NLM_A; qx <= NLM_A; qx++) 
            if (qy * (2 * NLM_A + 1) + qx < 0) {
                // #if defined(NLM_CLIP_REF_LUMA)
                float diff = U1[y][x] - U1[CLAMPY(y + qy)][CLAMPX(x + qx)];

                float val = 3.0f * diff * diff;
                // #endif

                U4a[qy+NLM_A][qx+NLM_A][y][x] = val;
            }
}

extern "C" __global__
void nlmHorizontal(const float U4a[NLM_A*2+1][NLM_A*2+1][HEIGHT][WIDTH], float U4b[NLM_A*2+1][NLM_A*2+1][HEIGHT][WIDTH]) {

    __shared__ float buffer[HRZ_BLOCK_Y][(HRZ_RESULT + 2) * HRZ_BLOCK_X];

    const int x = (blockIdx.x * HRZ_RESULT - 1) * HRZ_BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    cg::thread_block cta = cg::this_thread_block();

    for (int qy = -NLM_A; qy <= NLM_A; qy++)
        for (int qx = -NLM_A; qx <= NLM_A; qx++) {
            if (qy * (2 * NLM_A + 1) + qx < 0) {
                for (int i = 0; i <= 1 + HRZ_RESULT; i++)
                    buffer[threadIdx.y][threadIdx.x + i * HRZ_BLOCK_X] = 
                        U4a[qy+NLM_A][qx+NLM_A][y][CLAMPX(x + i * HRZ_BLOCK_X)];
            }

            cta.sync();

            if (qy * (2 * NLM_A + 1) + qx < 0) {
                for (int i = 1; i <= HRZ_RESULT; i++) {
                    if ((x + i * HRZ_BLOCK_X < WIDTH) && y < HEIGHT) {
                        float sum = 0.0f;

                        for (int j = -NLM_S; j <= NLM_S; j++)
                            sum += buffer[threadIdx.y][threadIdx.x + i * HRZ_BLOCK_X + j];

                        U4b[qy+NLM_A][qx+NLM_A][y][x + i * HRZ_BLOCK_X] = sum; // (x + i * HRZ_BLOCK_X) >= 0
                    }
                }
            }
        }
}

extern "C" __global__
void nlmVertical(const float U4b[NLM_A*2+1][NLM_A*2+1][HEIGHT][WIDTH], float U4a[NLM_A*2+1][NLM_A*2+1][HEIGHT][WIDTH]) {

    __shared__ float buffer[VRT_BLOCK_X][(VRT_RESULT + 2) * VRT_BLOCK_Y + 1];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = (blockIdx.y * VRT_RESULT - 1) * VRT_BLOCK_Y + threadIdx.y;


    cg::thread_block cta = cg::this_thread_block();

    for (int qy = -NLM_A; qy <= NLM_A; qy++)
        for (int qx = -NLM_A; qx <= NLM_A; qx++) {
            if (qy * (2 * NLM_A + 1) + qx < 0) {
                for (int i = 0; i <= 1 + VRT_RESULT; i++)
                    buffer[threadIdx.x][threadIdx.y + i * VRT_BLOCK_Y] = 
                        U4b[qy+NLM_A][qx+NLM_A][CLAMPY(y + i * VRT_BLOCK_Y)][x];
            }

            cta.sync();

            if (qy * (2 * NLM_A + 1) + qx < 0) {
                for (int i = 1; i <= HRZ_RESULT; i++) {
                   if (x < WIDTH && (y + i * VRT_BLOCK_Y) < HEIGHT) {
                        float sum = 0.0f;

                        for (int j = -NLM_S; j <= NLM_S; j++)
                            sum += buffer[threadIdx.x][threadIdx.y + i * VRT_BLOCK_Y + j];

#if NLM_WMODE == 0
                        // #if defined(NLM_WMODE_WELSCH)
                        const float val = expf(-sum * NLM_H2_INV_NORM);
#elif NLM_WMODE == 1
                        // #if defined(NLM_WMODE_BISQUARE_A)
                        const float val = fdimf(1.0f, sum * NLM_H2_INV_NORM);
#elif NLM_WMODE == 2
                        // #if defined(NLM_WMODE_BISQUARE_B)
                        const float val = powf(fdimf(1.0f, sum * NLM_H2_INV_NORM), 2.0f);
#elif NLM_WMODE == 3
                        // #if defined(NLM_WMODE_BISQUARE_C)
                        const float val = powf(fdimf(1.0f, sum * NLM_H2_INV_NORM), 8.0f);
#endif

                        U4a[qy+NLM_A][qx+NLM_A][y + i * VRT_BLOCK_Y][x] = val; // (y + i * VRT_BLOCK_Y) >= 0
                   }
                }
            }
        }
}

extern "C" __global__
void nlmAccumulation_Finish(const float U1a[HEIGHT][WIDTH], float U1z[HEIGHT][WIDTH], 
    const float U4a[NLM_A*2+1][NLM_A*2+1][HEIGHT][WIDTH]) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    float u5 = 1.1920928955078125e-7; // CL_FLT_EPSILON
    float u2a = 0.f;
    float u2b = 0.f;

    for (int qy = -NLM_A; qy <= NLM_A; qy++)
        for (int qx = -NLM_A; qx <= NLM_A; qx++) 
            if (qy * (2 * NLM_A + 1) + qx < 0) {
                float u4 = U4a[qy+NLM_A][qx+NLM_A][y][x];
                float u4_mq = U4a[qy+NLM_A][qx+NLM_A][CLAMPY(y - qy)][CLAMPX(x - qx)];
                u5 = fmaxf(u4, fmaxf(u4_mq, u5));

                // #if (NLM_CHANNELS == 1)
                float u1_pq = U1a[CLAMPY(y + qy)][CLAMPX(x + qx)];
                float u1_mq = U1a[CLAMPY(y - qy)][CLAMPX(x - qx)];

                u2a += (u4 * u1_pq) + (u4_mq * u1_mq);
                u2b += (u4 + u4_mq);
                // #endif
            }

    float m = NLM_WREF * u5;
    float den = m + u2b;

    U1z[y][x] = (U1a[y][x] * m + u2a) / (m + u2b);
}
