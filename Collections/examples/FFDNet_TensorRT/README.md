# Instructions
1. Install [CUDA-Python](https://github.com/NVIDIA/cuda-python).

2. Install TensorRT Python API. [Install-guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). PyCUDA is not required.

3. Download [FFDNet model](https://github.com/HolyWu/vs-ffdnet/blob/master/vsffdnet/ffdnet_color.pth) from HolyWu/vs-ffdnet.

4. Run `build_engine.py` to create serialized TensorRT engine.

    "The generated plan files are **not portable** across platforms or TensorRT versions and are specific to the exact GPU model they were built on", according to [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work). This sample further assumes that the dimensions of the video are fixed and known before engine creation.

5. (Optionally) Run `benchmark.py` or `trtexec --loadEngine="ffdnet.engine" --useCudaGraph` to test the engine's raw performance.

    `benchmark.py` writes a DOT file "ffdnet.dot" describing inference graph structure when `use_cuda_graph=True`. The DOT file can be visualized by running `dot -Tsvg ffdnet.dot > ffdnet.svg`.

6. Run `ffdnet_test.vpy` to test in VapourSynth.

