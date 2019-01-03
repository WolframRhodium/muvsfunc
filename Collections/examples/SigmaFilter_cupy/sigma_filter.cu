#define WIDTH ${width}
#define HEIGHT ${height}

#define RADIUS ${radius}
#define THRESHOLD ((float) ${threshold})

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#define CLAMPX(x) (MIN(MAX(x, 0), WIDTH - 1))
#define CLAMPY(y) (MIN(MAX(y, 0), HEIGHT - 1))

extern "C" __global__
void sigmaFilter(const float * __restrict__ src, float * __restrict__ dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    const float center = src[y * WIDTH + x];

    float sum = 0.0f;
    int count = 0;

    for (int j = -RADIUS; j <= RADIUS; j++)
        for (int i = -RADIUS; i <= RADIUS; i++) {
            const float val = src[CLAMPY(y + j) * WIDTH + CLAMPX(x + i)];

            if (fabsf(val - center) < THRESHOLD) {
                sum += val;
                count += 1;
            }
        }

    dst[y * WIDTH + x] = sum / count;
}
