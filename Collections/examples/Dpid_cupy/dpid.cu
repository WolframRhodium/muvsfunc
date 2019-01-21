// Copyright (c) 2016 Nicolas Weber and Sandra C. Amend / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.
// modified by WolframRhodium

#define THREADS 128
#define WSIZE 32
#define TSIZE (THREADS / WSIZE)

#define TX threadIdx.x
#define PX (blockIdx.x * TSIZE + (TX / WSIZE))
#define PY blockIdx.y

#define WTHREAD (TX % WSIZE)
#define WARP (TX / WSIZE)

#define LAMBDA ((float) (${lamda}))
#define IWIDTH (${iwidth})
#define IHEIGHT (${iheight})
#define OWIDTH (${owidth})
#define OHEIGHT (${oheight})
#define PWIDTH ((float) (${pwidth}))
#define PHEIGHT ((float) (${pheight}))
#define PIXEL_MAX ((float) ${pixel_max})

#define SX (fmaxf(PX * PWIDTH, 0.0f))
#define EX (fminf((PX + 1) * PWIDTH, IWIDTH))
#define SY (fmaxf(PY * PHEIGHT, 0.0f))
#define EY (fminf((PY + 1) * PHEIGHT, IHEIGHT))
#define SXR (__float2uint_rd(SX))
#define SYR (__float2uint_rd(SY))
#define EXR (__float2uint_ru(EX))
#define EYR (__float2uint_ru(EY))
#define XCOUNT (EXR - SXR)
#define YCOUNT (EYR - SYR)
#define PIXELCOUNT (XCOUNT * YCOUNT)

//-------------------------------------------------------------------
// DEVICE
//-------------------------------------------------------------------
__device__ __forceinline__ void normalize(float4& var)
{
    var.x /= var.w;
    var.y /= var.w;
    var.z /= var.w;
    var.w = 1.0f;
}

//-------------------------------------------------------------------
__device__ __forceinline__ void add(float4& output, const ${dtype}3& color, const float factor)
{
    output.x += color.x * factor;
    output.y += color.y * factor;
    output.z += color.z * factor;
    output.w += factor;
}

//-------------------------------------------------------------------
__device__ __forceinline__ void add(float4& output, const float4& color)
{
    output.x += color.x;
    output.y += color.y;
    output.z += color.z;
    output.w += color.w;
}

//-------------------------------------------------------------------
__device__ __forceinline__ float lambda(const float dist)
{
    if (LAMBDA == 0.0f)
        return 1.0f;
    else if (LAMBDA == 1.0f)
        return dist;

    return __powf(dist, LAMBDA);
}

//-------------------------------------------------------------------
__device__ __forceinline__ void operator+=(float4& output, const float4 value)
{
    output.x += value.x;
    output.y += value.y;
    output.z += value.z;
    output.w += value.w;
}

//-------------------------------------------------------------------
__device__ __forceinline__ float contribution(float f, const unsigned int x, const unsigned int y)
{
    if (x < SX)
        f *= 1.0f - (SX - x);

    if ((x + 1.0f) > EX)
        f *= 1.0f - ((x + 1.0f) - EX);

    if (y < SY)
        f *= 1.0f - (SY - y);

    if ((y + 1.0f) > EY)
        f *= 1.0f - ((y + 1.0f) - EY);

    return f;
}

//-------------------------------------------------------------------
// taken from: https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__device__ __forceinline__ float4 __shfl_down(const float4 var, const unsigned int srcLane, const unsigned int width = 32)
{
    float4 output;

#if __CUDA_ARCH__ >= 900
    output.x = __shfl_down_sync(0xFFFFFFFF, var.x, srcLane, width);
    output.y = __shfl_down_sync(0xFFFFFFFF, var.y, srcLane, width);
    output.z = __shfl_down_sync(0xFFFFFFFF, var.z, srcLane, width);
    output.w = __shfl_down_sync(0xFFFFFFFF, var.w, srcLane, width);
#else
    output.x = __shfl_down(var.x, srcLane, width);
    output.y = __shfl_down(var.y, srcLane, width);
    output.z = __shfl_down(var.z, srcLane, width);
    output.w = __shfl_down(var.w, srcLane, width);
#endif

    return output;
}

//-------------------------------------------------------------------
__device__ __forceinline__ void reduce(float4& value)
{
    value += __shfl_down(value, 16);
    value += __shfl_down(value, 8);
    value += __shfl_down(value, 4);
    value += __shfl_down(value, 2);
    value += __shfl_down(value, 1);
}

//-------------------------------------------------------------------
__device__ __forceinline__ float distance(const float4& avg, const ${dtype}3& color)
{
    const float x = avg.x - color.x;
    const float y = avg.y - color.y;
    const float z = avg.z - color.z;

    return sqrtf((x * x + y * y + z * z) / 3.0f) / PIXEL_MAX; // L2-Norm / sqrt(255^2 * 3)
}

//-------------------------------------------------------------------
extern "C"
__global__ void kernelGuidance(const ${dtype}3* __restrict__ input, ${dtype}3* __restrict__ patches)
{
    if (PX >= OWIDTH || PY >= OHEIGHT)
        return;

    // init
    float4 color = { 0 };

    // iterate pixels
    for (unsigned int i = WTHREAD; i < PIXELCOUNT; i += WSIZE)
    {
        const unsigned int x = SXR + (i % XCOUNT);
        const unsigned int y = SYR + (i / XCOUNT);

        const float f = contribution(1.0f, x, y);

        const ${dtype}3& pixel = input[x + y * IWIDTH];

        add(color, make_float4(pixel.x * f, pixel.y * f, pixel.z * f, f));
    }

    // reduce warps
    reduce(color);

    // store results
    if ((TX % 32) == 0)
    {
        normalize(color);
        patches[PX + PY * OWIDTH] = make_${dtype}3(color.x, color.y, color.z);
    }
}

//-------------------------------------------------------------------
__device__ __forceinline__ float4 calcAverage(const ${dtype}3* __restrict__ patches)
{
    const float corner = 1.0f;
    const float edge = 2.0f;
    const float center = 4.0f;

    // calculate average color
    float4 avg = { 0.f };

    // TOP
    if (PY > 0)
    {
        if (PX > 0)
            add(avg, patches[(PX - 1) + (PY - 1) * OWIDTH], corner);

        add(avg, patches[(PX)+(PY - 1) * OWIDTH], edge);

        if ((PX + 1) < OWIDTH)
            add(avg, patches[(PX + 1) + (PY - 1) * OWIDTH], corner);
    }

    // LEFT
    if (PX > 0)
        add(avg, patches[(PX - 1) + (PY)* OWIDTH], edge);

    // CENTER
    add(avg, patches[(PX)+(PY)* OWIDTH], center);

    // RIGHT
    if ((PX + 1) < OWIDTH)
        add(avg, patches[(PX + 1) + (PY)* OWIDTH], edge);

    // BOTTOM
    if ((PY + 1) < OHEIGHT)
    {
        if (PX > 0)
            add(avg, patches[(PX - 1) + (PY + 1) * OWIDTH], corner);

        add(avg, patches[(PX)+(PY + 1) * OWIDTH], edge);

        if ((PX + 1) < OWIDTH)
            add(avg, patches[(PX + 1) + (PY + 1) * OWIDTH], corner);
    }

    normalize(avg);

    return avg;
}

//-------------------------------------------------------------------
extern "C"
__global__ void kernelDownsampling(const ${dtype}3* __restrict__ input, const ${dtype}3* __restrict__ patches, ${dtype}3* __restrict__ output)
{
    if (PX >= OWIDTH || PY >= OHEIGHT) return;

    // init
    const float4 avg = calcAverage(patches);

    float4 color = { 0.f };

    // iterate pixels
    for (unsigned int i = WTHREAD; i < PIXELCOUNT; i += WSIZE)
    {
        const unsigned int x = SXR + (i % XCOUNT);
        const unsigned int y = SYR + (i / XCOUNT);

        const ${dtype}3& pixel = input[x + y * IWIDTH];
        float f = distance(avg, pixel);

        f = lambda(f);
        f = contribution(f, x, y);

        add(color, pixel, f);
    }

    // reduce warp
    reduce(color);

    if (WTHREAD == 0)
    {
        ${dtype}3& ref = output[PX + PY * OWIDTH];

        if (color.w == 0.0f)
            ref = make_${dtype}3(avg.x, avg.y, avg.z);
        else
        {
            normalize(color);
            ref = make_${dtype}3(color.x, color.y, color.z);
        }
    }
}
