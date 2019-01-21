"""core.dpid.Dpid() in CuPy"""

from string import Template

import cupy as cp
import vapoursynth as vs
import muvsfunc_numpy as mufnp

core = vs.get_core()


# Load source clip. Only RGB24/RGB48/RGBS is supported
src = core.std.BlankClip(format=vs.RGB24)


# params of core.dpid.Dpid()
width = src.width // 2
height = src.height // 2
_lambda = 1.0

# whether to enable '--use_fast_math' in NVRTC runtime compilation
# to make use of fast math operations
fast = False


# pre-processing
if src.format.color_family != vs.RGB:
    raise TypeError("'src' must be a RGB clip.")


if src.format.sample_type == vs.FLOAT:
    dtype = 'float'
    pixel_max = 1

elif src.format.bits_per_sample == 8:
    dtype = 'uchar'
    pixel_max = 255 # (1 << 8) - 1

else:
    dtype = 'ushort'
    pixel_max = (1 << src.format.bits_per_sample) - 1


# load CUDA kernel
with open('dpid.cu', 'r') as f:
    kernel_source_code = f.read()

kernel_source_code = Template(kernel_source_code)
kernel_source_code = kernel_source_code.substitute(
    iwidth=src.width, iheight=src.height, owidth=width, oheight=height,
    pwidth=src.width / width, pheight=src.height / height, lamda=_lambda, 
    pixel_max=pixel_max, dtype=dtype)


if fast:
    kernelGuidance = cp.RawKernel(code=kernel_source_code, name='kernelGuidance', 
        options=('--use_fast_math', ))
    kernelDownsampling = cp.RawKernel(code=kernel_source_code, name='kernelDownsampling', 
        options=('--use_fast_math', ))
else:
    kernelGuidance = cp.RawKernel(code=kernel_source_code, name='kernelGuidance')
    kernelDownsampling = cp.RawKernel(code=kernel_source_code, name='kernelDownsampling')


# create NumPy function
def dpid_core(h_input, width, height, kernelGuidance, kernelDownsampling):
    d_input = cp.asarray(h_input)
    d_output = cp.zeros((height, width, 3), dtype=h_input.dtype)
    d_guidance = cp.zeros((height, width, 3), dtype=h_input.dtype)

    kernelGuidance((width // 4, height, 1), (128, 1, 1), (d_input, d_guidance))
    kernelDownsampling((width // 4, height, 1), (128, 1, 1), (d_input, d_guidance, d_output))

    h_out = cp.asnumpy(d_output)

    return h_out


# process
res = mufnp.numpy_process(
        [core.std.BlankClip(src, width=width, height=height), src], 
        dpid_core, width=width, height=height, 
        kernelGuidance=kernelGuidance, kernelDownsampling=kernelDownsampling, 
        input_per_plane=False, output_per_plane=False, omit_first_clip=True)

"""
if src.format.sample_type == vs.INTEGER:
    res = core.dpid.Dpid(src, width=width, height=height, _lambda=_lambda)

else: # src.format.sample_type == vs.FLOAT
    res = core.dpid.Dpid(src.fmtc.bitdepth(bits=16), width=width, height=height, _lambda=_lambda)
    res = core.fmtc.bitdepth(res, bits=src.format.bits_per_sample)
"""


res.set_output()
