import ctypes

import vapoursynth as vs
from vapoursynth import core

import numpy as np
import tensorrt as trt

from utils import *


_cuda_context = init_cuda()


def FFDNet(
    clip: vs.VideoNode,
    sigma: float = 5.0,
    logger: trt.Logger = trt.Logger(trt.Logger.WARNING)
) -> vs.VideoNode:

    assert clip.format.id == vs.RGBS
    width, height = clip.width, clip.height

    sigma /= 255

    runtime = trt.Runtime(logger)

    with open(f"ffdnet_{width}_{height}.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    execution_context = engine.create_execution_context()
    input_size = execution_context.get_strides(0)[0] * 4
    input_shape = execution_context.get_binding_shape(0)
    sigma_size = execution_context.get_strides(1)[0] * 4
    sigma_shape = execution_context.get_binding_shape(1)
    output_size = execution_context.get_strides(2)[0] * 4
    output_shape = execution_context.get_binding_shape(2)

    h_sigma = checkError(cuda.cuMemHostAlloc(
        sigma_size, cuda.CU_MEMHOSTALLOC_WRITECOMBINED))
    h_sigma = UniqueResource(h_sigma, cuda.cuMemFreeHost, h_sigma)
    h_sigma_pointer = ctypes.cast(
        ctypes.c_void_p(h_sigma.obj), ctypes.POINTER(ctypes.c_float))
    h_sigma_array = np.ctypeslib.as_array(
        h_sigma_pointer, shape=(sigma_size // 4,)).reshape(sigma_shape)

    d_sigma = checkError(cuda.cuMemAlloc(sigma_size))
    d_sigma = UniqueResource(d_sigma, cuda.cuMemFree, d_sigma)

    h_input = checkError(cuda.cuMemHostAlloc(
        input_size, cuda.CU_MEMHOSTALLOC_WRITECOMBINED))
    h_input = UniqueResource(h_input, cuda.cuMemFreeHost, h_input)
    h_input_pointer = ctypes.cast(
        ctypes.c_void_p(h_input.obj), ctypes.POINTER(ctypes.c_float))
    h_input_array = np.ctypeslib.as_array(
        h_input_pointer, shape=(input_size // 4,)).reshape(input_shape)

    d_input = checkError(cuda.cuMemAlloc(input_size))
    d_input = UniqueResource(d_input, cuda.cuMemFree, d_input)

    d_output = checkError(cuda.cuMemAlloc(output_size))
    d_output = UniqueResource(d_output, cuda.cuMemFree, d_output)

    h_output = checkError(cuda.cuMemAllocHost(output_size))
    h_output = UniqueResource(h_output, cuda.cuMemFreeHost, h_output)
    h_output_pointer = ctypes.cast(
        ctypes.c_void_p(h_output.obj), ctypes.POINTER(ctypes.c_float))
    h_output_array = np.ctypeslib.as_array(
        h_output_pointer, shape=(output_size // 4,)).reshape(output_shape)

    stream = checkError(cuda.cuStreamCreate(
        cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value))
    stream = UniqueResource(stream, cuda.cuStreamDestroy, stream)

    h_sigma_array[...] = sigma
    checkError(cuda.cuMemcpyHtoDAsync(
        d_sigma.obj, h_sigma.obj, sigma_size, stream.obj))

    def inference_core(n, f):
        for i in range(3):
            h_input_array[0, i, :, :] = np.asarray(f.get_read_array(i))

        checkError(cuda.cuMemcpyHtoDAsync(
            d_input.obj, h_input.obj, input_size, stream.obj))

        execution_context.execute_async_v2(
            [d_input.obj, d_sigma.obj, d_output.obj],
            stream_handle=stream.obj)

        checkError(cuda.cuMemcpyDtoHAsync(
            h_output.obj, d_output.obj, output_size, stream.obj))

        fout = f.copy()
        fout.get_write_array(0) # triggers COW
        checkError(cuda.cuStreamSynchronize(stream.obj))

        for i in range(3):
            np.asarray(fout.get_write_array(i))[...] = h_output_array[0, i, :, :]

        return fout

    return core.std.ModifyFrame(clip, clips=[clip], selector=inference_core)

