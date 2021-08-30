from typing import List, Optional

import numpy as np
from cuda import cuda
import tensorrt as trt


def checkError(args):
    err, *ret = args

    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")

    if len(ret) == 0:
        return
    elif len(ret) == 1:
        return ret[0]
    else:
        return ret


class UniqueResource:
    def __init__(self, obj, func, *args, **kwargs):
        self.obj = obj
        self._func = [func]
        self._args = [args]
        self._kwargs = [kwargs]

    def register(self, func, *args, **kwargs):
        """ register a finalizer """

        self._func.append(func)
        self._args.append(args)
        self._kwargs.append(kwargs)

    def __del__(self):
        # calls finalizers in reversed order
        it = zip(reversed(self._func), reversed(self._args), reversed(self._kwargs))

        for func, args, kwargs in it:
            func(*args, **kwargs)


def init_cuda():
    checkError(cuda.cuInit(0))
    device = checkError(cuda.cuDeviceGet(0))

    context = checkError(cuda.cuDevicePrimaryCtxRetain(device))
    context = UniqueResource(context, cuda.cuDevicePrimaryCtxRelease, device)

    checkError(cuda.cuCtxPushCurrent(context.obj))
    context.register(cuda.cuCtxPopCurrent)

    return device, context


def convolution(
    network: trt.INetworkDefinition,
    input: trt.ITensor,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    kernel: Optional[np.array] = None,
    bias: Optional[np.array] = None
) -> trt.ITensor:

    if kernel is None:
        kernel = np.empty(
            (out_channels, in_channels, kernel_size, kernel_size),
            dtype=np.float32)

    if bias is None:
        bias = np.zeros(out_channels, dtype=np.float32)

    output = network.add_convolution_nd(
        input=input, num_output_maps=out_channels,
        kernel_shape=(kernel_size, kernel_size),
        kernel=kernel, bias=bias)

    output.padding_nd = (kernel_size // 2, kernel_size // 2)
    output.stride_nd = (1, 1)

    return output.get_output(0)


def pixel_unshuffle(
    network: trt.INetworkDefinition,
    input: trt.ITensor,
    downscale_factor: int
) -> trt.ITensor:

    n, ic, ih, iw = input.shape
    assert ih % downscale_factor == 0 and ih % downscale_factor == 0
    oc = ic * (downscale_factor ** 2)
    oh = ih // downscale_factor
    ow = iw // downscale_factor

    reshape = network.add_shuffle(input)
    reshape.reshape_dims = trt.Dims([n, ic, oh, downscale_factor, ow, downscale_factor])
    reshape.second_transpose = trt.Permutation([0, 1, 3, 5, 2, 4])

    reshape = network.add_shuffle(reshape.get_output(0))
    reshape.reshape_dims = trt.Dims([n, oc, oh, ow])

    return reshape.get_output(0)


def pixel_shuffle(
    network: trt.INetworkDefinition,
    input: trt.ITensor,
    upscale_factor: int
) -> trt.ITensor:

    n, ic, ih, iw = input.shape
    assert ic % (upscale_factor ** 2) == 0
    oc = ic // (upscale_factor ** 2)
    oh = ih * upscale_factor
    ow = iw * upscale_factor

    reshape = network.add_shuffle(input)
    reshape.reshape_dims = trt.Dims([n, oc, upscale_factor, upscale_factor, ih, iw])
    reshape.second_transpose = trt.Permutation([0, 1, 4, 2, 5, 3])

    reshape = network.add_shuffle(reshape.get_output(0))
    reshape.reshape_dims = trt.Dims([n, oc, oh, ow])

    return reshape.get_output(0)

