#include <cstdint>
#include <VapourSynth.h>
#include <VSHelper.h>

#define kNumInputs ${num_inputs}
#define T ${t}
#define kInputs ${inputs}
#define kFunction ${func_name}
const int kProcess[3] = ${planes};

${func_impl}

typedef struct {
    VSNodeRef *node[kNumInputs];
    const VSVideoInfo *vi;
} ExprData;

static void VS_CC ExprInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    ExprData *d = (ExprData *) * instanceData;
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC ExprGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    ExprData *d = (ExprData *) * instanceData;

    if (activationReason == arInitial) {
        for (int i = 0; i < kNumInputs; ++i)
            vsapi->requestFrameFilter(n, d->node[i], frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src[kNumInputs] = {};
        for (int i = 0; i < kNumInputs; ++i)
            src[i] = vsapi->getFrameFilter(n, d->node[i], frameCtx);

        const VSFormat *fi = d->vi->format;
        int height = vsapi->getFrameHeight(src[0], 0);
        int width = vsapi->getFrameWidth(src[0], 0);

        int planes[3] = { 0, 1, 2 };
        const VSFrameRef *srcf[3] = { kProcess[0] ? nullptr : src[0], kProcess[1] ? nullptr : src[0], kProcess[2] ? nullptr : src[0] };
        VSFrameRef *dst = vsapi->newVideoFrame2(fi, width, height, srcf, planes, src[0], core);

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!kProcess[plane])
                continue;

            const T *srcp[kNumInputs] = {};
            for (int i = 0; i < kNumInputs; ++i)
                srcp[i] = (const T*) vsapi->getReadPtr(src[i], plane);

            int src_stride = vsapi->getStride(src[0], plane);
            T *dstp = (T*) vsapi->getWritePtr(dst, plane);
            int dst_stride = vsapi->getStride(dst, plane);
            int h = vsapi->getFrameHeight(src[0], plane);
            int w = vsapi->getFrameWidth(src[0], plane);

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    dstp[x] = kFunction(kInputs);
                }

                dstp += dst_stride;
                for (int i = 0; i < kNumInputs; ++i)
                    srcp[i] += src_stride;
            }
        }

        for (int i = 0; i < kNumInputs; ++i)
            vsapi->freeFrame(src[i]);

        return dst;
    }

    return 0;
}

static void VS_CC ExprFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ExprData *d = (ExprData *)instanceData;
    for (int i = 0; i < kNumInputs; ++i)
        vsapi->freeNode(d->node[i]);
    free(d);
}

static void VS_CC ExprCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    ExprData d;
    ExprData *data;

    for (int i = 0; i < kNumInputs; ++i) {
        auto node = vsapi->propGetNode(in, "clips", i, 0);
        auto vi = vsapi->getVideoInfo(node);
        if (!isConstantFormat(vi)) {
            vsapi->setError(out, "Expr: only constant format input supported");
            for (int j = 0; j < i; ++j)
                vsapi->freeNode(d.node[j]);
            return;
        }
        d.node[i] = node;
    }

    d.vi = vsapi->getVideoInfo(d.node[0]);

    data = (ExprData *) malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "Expr", ExprInit, ExprGetFrame, ExprFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("exprtest", "expr", "Expr test", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Expr", "clips:clip[];", ExprCreate, 0, plugin);
}
