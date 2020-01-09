#define WIDTH ${width}
#define HEIGHT ${height}
#define NLM_A ${a}
#define NLM_S ${s}
#define NLM_H ((float) (${h} / 79.636080791869483631941455867052))
#define NLM_H2 ((float) (${h2} / 79.636080791869483631941455867052))


#define GET(pointer, y0, x0) pointer[max(min((y0), HEIGHT-1), 0) * WIDTH + max(min((x0), WIDTH-1), 0)]
#define PatchMatrix(y0, x0) GET(srcp, y-NLM_A-NLM_S + (y0) / (2*NLM_A+1) + (x0) / (2*NLM_S+1), x-NLM_A-NLM_S + (y0) % (2*NLM_A+1) + (x0) % (2*NLM_S+1))
#define Square(x) ((x) * (x))

#define PatchSize Square(2 * NLM_S + 1)
#define SearchSize Square(2 * NLM_A + 1)

extern "C" __global__ 
void compute(const float * __restrict__ srcp, float * __restrict__ dstp) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    float PatchWeights[SearchSize];

    // CalculatePatchWeights
    float NormalizingConstant = 0.f;
    for (int i = 0; i < SearchSize; i++) {
        float SSE = 0.f;
        for (int j = 0; j < PatchSize; j++)
            SSE += Square(PatchMatrix(i, j) - PatchMatrix(SearchSize / 2, j));
        float Weight = expf(-SSE / Square(NLM_H));
        PatchWeights[i] = Weight;
        NormalizingConstant += Weight;
    }

    for (int i = 0; i < SearchSize; i++) {
        PatchWeights[i] /= NormalizingConstant;
    }

    // CalculatePositionWeights & Aggregate
    float Result = 0.f;
    NormalizingConstant = 0.f;
    for (int j = 0; j < PatchSize; j++) {
        float SSE = 0.f;
        for (int i = 0; i < SearchSize; i++)
            SSE += PatchWeights[i] * Square(PatchMatrix(i, j) - PatchMatrix(i, PatchSize / 2));
        float Weight = expf(-SSE / Square(NLM_H2));
        Result += Weight * PatchMatrix(j, PatchSize / 2);
        NormalizingConstant += Weight;
    }
    
    GET(dstp, y, x) = Result / NormalizingConstant;
}
