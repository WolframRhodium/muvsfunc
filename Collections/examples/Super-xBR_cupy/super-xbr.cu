/*
CUDA port of Super-xBR image upscaling algorithm by WolframRhodium

The algorithm is modified for data parallelism

Source: https://pastebin.com/cbH8ZQQT

*******  Super XBR Scaler  *******
 
Copyright (c) 2016 Hyllian - sergiogdb@gmail.com
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#define IWIDTH (${width})
#define IHEIGHT (${height})
#define OWIDTH (IWIDTH * 2)
#define OHEIGHT (IHEIGHT * 2)

#define WGT1 ((float) ${wgt1}) // 0.129633f
#define WGT2 ((float) ${wgt2}) // 0.175068f
#define W1 (-WGT1)
#define W2 (WGT1 + 0.5f)
#define W3 (-WGT2)
#define W4 (WGT2 + 0.5f)

__device__ __forceinline__ int clamp(int x, int floor, int ceil) {
    return max(floor, min(x, ceil));
}

__device__ __forceinline__ float3 min4(float3 a, float3 b, float3 c, float3 d) {
    return make_float3(
        fminf(fminf(a.x, b.x), fminf(c.x, d.x)), 
        fminf(fminf(a.y, b.y), fminf(c.y, d.y)), 
        fminf(fminf(a.z, b.z), fminf(c.z, d.z)));
}

__device__ __forceinline__ float3 max4(float3 a, float3 b, float3 c, float3 d) {
    return make_float3(
        fmaxf(fmaxf(a.x, b.x), fmaxf(c.x, d.x)), 
        fmaxf(fmaxf(a.y, b.y), fmaxf(c.y, d.y)), 
        fmaxf(fmaxf(a.z, b.z), fmaxf(c.z, d.z)));
}

__device__ __forceinline__ float3 clamp(float3 x, float3 floor, float3 ceil) {
    return make_float3(
        fmaxf(floor.x, fminf(x.x, ceil.x)), 
        fmaxf(floor.y, fminf(x.y, ceil.y)), 
        fmaxf(floor.z, fminf(x.z, ceil.z)));
}

__device__ __forceinline__ float df(float a, float b) {
    return fabsf(a - b);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(
        a.x + b.x, 
        a.y + b.y, 
        a.z + b.z);
}

__device__ __forceinline__ float3 operator*(float a, float3 b)
{
    return make_float3(
        a * b.x, 
        a * b.y,
        a * b.z);
}

__device__ __forceinline__ float diagonal_edge(const float mat[][4], const float *wp) {
    float dw1 = wp[0]*(df(mat[0][2], mat[1][1]) + df(mat[1][1], mat[2][0]) + df(mat[1][3], mat[2][2]) + df(mat[2][2], mat[3][1])) + \
                wp[1]*(df(mat[0][3], mat[1][2]) + df(mat[2][1], mat[3][0])) + \
                wp[2]*(df(mat[0][3], mat[2][1]) + df(mat[1][2], mat[3][0])) + \
                wp[3]*(df(mat[1][2], mat[2][1])) + \
                wp[4]*(df(mat[0][2], mat[2][0]) + df(mat[1][3], mat[3][1])) + \
                wp[5]*(df(mat[0][1], mat[1][0]) + df(mat[2][3], mat[3][2]));
 
    float dw2 = wp[0]*(df(mat[0][1], mat[1][2]) + df(mat[1][2], mat[2][3]) + df(mat[1][0], mat[2][1]) + df(mat[2][1], mat[3][2])) + \
                wp[1]*(df(mat[0][0], mat[1][1]) + df(mat[2][2], mat[3][3])) + \
                wp[2]*(df(mat[0][0], mat[2][2]) + df(mat[1][1], mat[3][3])) + \
                wp[3]*df(mat[1][1], mat[2][2]) + \
                wp[4]*(df(mat[1][0], mat[3][2]) + df(mat[0][1], mat[2][3])) + \
                wp[5]*(df(mat[0][2], mat[1][3]) + df(mat[2][0], mat[3][1]));
 
    return (dw1 - dw2);
}

extern "C"
__global__ void super_xbr_pass1(const float3 * __restrict__ src, float3 * __restrict__ dst) {
    // src: W x H, dst: 2W x 2H
    // x: 0:W:1, y: 0:H:1

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= IWIDTH || y >= IHEIGHT)
        return;

    // copy pixels to output image
    dst[((y * 2) * OWIDTH) + (x * 2)] = src[y * IWIDTH + x];
    dst[((y * 2) * OWIDTH) + (x * 2 + 1)] = src[y * IWIDTH + x];
    dst[((y * 2 + 1) * OWIDTH) + (x * 2)] = src[y * IWIDTH + x];

    // init
    const float wp[6] = { 2.0f, 1.0f, -1.0f, 4.0f, -1.0f, 1.0f };

    float3 rgb_data[4][4];
    float y_data[4][4];

    // sample supporting pixels in original image
    for (int sy = -1; sy <= 2; ++sy) {
        const int csy = clamp(y + sy, 0, IHEIGHT - 1);

        for (int sx = -1; sx <= 2; ++sx) {
            // clamp pixel locations
            const int csx = clamp(x + sx, 0, IWIDTH - 1);

            // sample & add weighted components
            rgb_data[sy + 1][sx + 1] = src[csy * IWIDTH + csx];

            y_data[sy + 1][sx + 1] = 0.2126f * rgb_data[sy + 1][sx + 1].x + 0.7152f * rgb_data[sy + 1][sx + 1].y + \
                0.0722f * rgb_data[sy + 1][sx + 1].z;
        }
    }

    const float3 min_sample = min4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);
    const float3 max_sample = max4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);

    const float d_edge = diagonal_edge(y_data, wp);

    const float3 rgb1 = W1 * (rgb_data[0][3] + rgb_data[3][0]) + W2 * (rgb_data[1][2] + rgb_data[2][1]);
    const float3 rgb2 = W1 * (rgb_data[0][0] + rgb_data[3][3]) + W2 * (rgb_data[1][1] + rgb_data[2][2]);

    // generate and write result
    float3 rgbf = (d_edge <= 0.0f) ? rgb1 : rgb2;

    // anti-ringing, clamp
    rgbf = clamp(rgbf, min_sample, max_sample);

    // output
    dst[((y * 2 + 1) * OWIDTH) + (x * 2 + 1)] = rgbf;
}

extern "C"
__global__ void super_xbr_pass2(const float3 * __restrict__ src, float3 * __restrict__ dst) {
    // src: 2W x 2H, dst: 2W x 2H
    // x: 0:W:1, y: 0:H:1

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= IWIDTH || y >= IHEIGHT)
        return;

    // copy pixels to output image
    dst[((y * 2) * OWIDTH) + (x * 2)] = src[((y * 2) * OWIDTH) + (x * 2)];
    dst[((y * 2 + 1) * OWIDTH) + (x * 2 + 1)] = src[((y * 2 + 1) * OWIDTH) + (x * 2 + 1)];

    // init
    const float wp[6] = { 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    float3 rgb_data[4][4];
    float y_data[4][4];

    // output: dst[((y * 2) * OWIDTH) + (x * 2 + 1)]
    {
        // sample supporting pixels in original image
        for (int sy = -1; sy <= 2; ++sy) {
            for (int sx = -1; sx <= 2; ++sx) {
                // clamp pixel locations
                const int csy = clamp((y * 2) + sx - sy, 0, OHEIGHT - 1);
                const int csx = clamp((x * 2) + sx + sy, 0, OWIDTH - 1);

                // sample & add weighted components
                rgb_data[sy + 1][sx + 1] = src[(csy * OWIDTH + csx)];

                y_data[sy + 1][sx + 1] = 0.2126f * rgb_data[sy + 1][sx + 1].x + 0.7152f * rgb_data[sy + 1][sx + 1].y + \
                    0.0722f * rgb_data[sy + 1][sx + 1].z;
            }
        }

        const float3 min_sample = min4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);
        const float3 max_sample = max4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);

        const float d_edge = diagonal_edge(y_data, wp);

        const float3 rgb1 = W3 * (rgb_data[0][3] + rgb_data[3][0]) + W4 * (rgb_data[1][2] + rgb_data[2][1]);
        const float3 rgb2 = W3 * (rgb_data[0][0] + rgb_data[3][3]) + W4 * (rgb_data[1][1] + rgb_data[2][2]);

        // generate and write result
        float3 rgbf = (d_edge <= 0.0f) ? rgb1 : rgb2;

        // anti-ringing, clamp
        rgbf = clamp(rgbf, min_sample, max_sample);

        // output
        dst[((y * 2) * OWIDTH) + (x * 2 + 1)] = rgbf;
    }

    // output: dst[((y * 2 + 1) * OWIDTH) + (x * 2)]
    {
        // sample supporting pixels in original image
        for (int sy = -1; sy <= 2; ++sy) {
            for (int sx = -1; sx <= 2; ++sx) {
                // clamp pixel locations
                const int csy = clamp((y * 2) + sx - sy + 1, 0, OHEIGHT - 1);
                const int csx = clamp((x * 2) + sx + sy - 1, 0, OWIDTH - 1);

                // sample & add weighted components
                rgb_data[sy + 1][sx + 1] = src[csy * OWIDTH + csx];

                y_data[sy + 1][sx + 1] = 0.2126f * rgb_data[sy + 1][sx + 1].x + 0.7152f * rgb_data[sy + 1][sx + 1].y + \
                    0.0722f * rgb_data[sy + 1][sx + 1].z;
            }
        }

        const float3 min_sample = min4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);
        const float3 max_sample = max4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);

        const float d_edge = diagonal_edge(y_data, wp);

        const float3 rgb1 = W3 * (rgb_data[0][3] + rgb_data[3][0]) + W4 * (rgb_data[1][2] + rgb_data[2][1]);
        const float3 rgb2 = W3 * (rgb_data[0][0] + rgb_data[3][3]) + W4 * (rgb_data[1][1] + rgb_data[2][2]);

        // generate and write result
        float3 rgbf = (d_edge <= 0.0f) ? rgb1 : rgb2;

        // anti-ringing, clamp
        rgbf = clamp(rgbf, min_sample, max_sample);

        // output
        dst[((y * 2 + 1) * OWIDTH) + (x * 2)] = rgbf;
    }
}

extern "C"
__global__ void super_xbr_pass3(const float3 * __restrict__ src, float3 * __restrict__ dst) {
    // src: 2W x 2H, dst: 2W x 2H
    // x: 0:2W:1, y: 0:2H:1

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= OWIDTH || y >= OHEIGHT)
        return;

    // init
    const float wp[6] = { 2.0f, 1.0f, -1.0f, 4.0f, -1.0f, 1.0f };

    float3 rgb_data[4][4];
    float y_data[4][4];

    // sample supporting pixels in original image
    for (int sy = -2; sy <= 1; ++sy) {
        const int csy = clamp(y + sy, 0, OHEIGHT - 1);

        for (int sx = -2; sx <= 1; ++sx) {
            // clamp pixel locations
            const int csx = clamp(x + sx, 0, OWIDTH - 1);

            // sample & add weighted components
            rgb_data[sy + 2][sx + 2] = src[csy * OWIDTH + csx];

            y_data[sy + 2][sx + 2] = 0.2126f * rgb_data[sy + 2][sx + 2].x + 0.7152f * rgb_data[sy + 2][sx + 2].y + \
                0.0722f * rgb_data[sy + 2][sx + 2].z;
        }
    }

    const float3 min_sample = min4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);
    const float3 max_sample = max4(rgb_data[1][1], rgb_data[2][1], rgb_data[1][2], rgb_data[2][2]);

    const float d_edge = diagonal_edge(y_data, wp);

    const float3 rgb1 = W1 * (rgb_data[0][3] + rgb_data[3][0]) + W2 * (rgb_data[1][2] + rgb_data[2][1]);
    const float3 rgb2 = W1 * (rgb_data[0][0] + rgb_data[3][3]) + W2 * (rgb_data[1][1] + rgb_data[2][2]);

    // generate and write result
    float3 rgbf = (d_edge <= 0.0f) ? rgb1 : rgb2;

    // anti-ringing, clamp
    rgbf = clamp(rgbf, min_sample, max_sample);

    // output
    dst[y * OWIDTH + x] = rgbf;
}