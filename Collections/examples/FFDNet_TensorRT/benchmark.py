from cuda import cuda
import tensorrt as trt
from utils import *


def _get_bindings(
    context: trt.IExecutionContext,
    num_bindings: int
) -> List[UniqueResource]:

    bindings = []
    for i in range(num_bindings):
        binding = checkError(cuda.cuMemAlloc(context.get_strides(i)[0] * 4))
        binding = UniqueResource(binding, cuda.cuMemFree, binding)
        bindings.append(binding)
    return bindings


def benchmark(
    width: int,
    height: int,
    iter: int = 5,
    use_cuda_graph: bool = False,
    logger: trt.Logger = trt.Logger(trt.Logger.VERBOSE)
) -> None:

    cuda_context = init_cuda()

    runtime = trt.Runtime(logger)

    with open(f"ffdnet_{width}_{height}.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    execution_context = engine.create_execution_context()

    _bindings = _get_bindings(execution_context, engine.num_bindings)
    bindings = [binding.obj for binding in _bindings]

    stream = checkError(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value))
    stream = UniqueResource(stream, cuda.cuStreamDestroy, stream)

    start = checkError(cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT.value))
    start = UniqueResource(start, cuda.cuEventDestroy, start)

    end = checkError(cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT.value))
    end = UniqueResource(end, cuda.cuEventDestroy, end)

    def execute():
        execution_context.execute_async_v2(bindings, stream_handle=stream.obj)

    if use_cuda_graph:
        checkError(cuda.cuStreamBeginCapture(
            stream.obj, cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED))

        execute()

        graph = checkError(cuda.cuStreamEndCapture(stream.obj))
        graphexec, error_node = checkError(cuda.cuGraphInstantiate(
            graph, logBuffer=b"", bufferSize=0))
        graphexec = UniqueResource(graphexec, cuda.cuGraphExecDestroy, graphexec)
        checkError(cuda.cuGraphDebugDotPrint(
            graph, b"ffdnet.dot",
            cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE.value))
        checkError(cuda.cuGraphDestroy(graph))

    for _ in range(iter):
        checkError(cuda.cuEventRecord(start.obj, stream.obj))

        if use_cuda_graph:
            checkError(cuda.cuGraphLaunch(graphexec.obj, stream.obj))
        else:
            execute()

        checkError(cuda.cuEventRecord(end.obj, stream.obj))
        checkError(cuda.cuEventSynchronize(end.obj))

        duration = checkError(cuda.cuEventElapsedTime(start.obj, end.obj))

        print(f"duration: {duration} ms")


if __name__ == "__main__":
    benchmark(width=1920, height=1080, iter=10, use_cuda_graph=False)
