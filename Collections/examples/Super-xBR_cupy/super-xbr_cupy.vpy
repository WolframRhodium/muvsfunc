"""Super-xBR in CuPy"""

from string import Template

import cupy as cp
import vapoursynth as vs
import muvsfunc_numpy as mufnp

core = vs.get_core()


# Load source clip. Only RGBS is supported
src = core.std.BlankClip(format=vs.RGBS)


# params of Super-xBR
# Super-xBR upscale an image by a factor of 2
wgt1 = 0.129633
wgt2 = 0.175068

# whether to enable '--use_fast_math' in NVRTC runtime compilation
# to make use of fast math operations
fast = False

# CUDA kernel execution configuration
blk_size = (8, 8)

# pre-processing
if src.format.id != vs.RGBS:
    raise vs.Error("Super-xBR: Only 32-bit float RGB is supported!")


# load CUDA kernel
with open('super-xbr.cu', 'r') as f:
    kernel_source_code = f.read()

kernel_source_code = Template(kernel_source_code)
kernel_source_code = kernel_source_code.substitute(
    width=src.width, height=src.height, 
    wgt1=wgt1, wgt2=wgt2)


if fast:
    pass1 = cp.RawKernel(code=kernel_source_code, name='super_xbr_pass1', 
        options=('--use_fast_math', ))
    pass2 = cp.RawKernel(code=kernel_source_code, name='super_xbr_pass2', 
        options=('--use_fast_math', ))
    pass3 = cp.RawKernel(code=kernel_source_code, name='super_xbr_pass3', 
        options=('--use_fast_math', ))
else:
    pass1 = cp.RawKernel(code=kernel_source_code, name='super_xbr_pass1')
    pass2 = cp.RawKernel(code=kernel_source_code, name='super_xbr_pass2')
    pass3 = cp.RawKernel(code=kernel_source_code, name='super_xbr_pass3')


# create NumPy function
def superxbr_core(h_input, pass1, pass2, pass3):
    h, w, _ = h_input.shape
    d_input = cp.asarray(h_input)

    d_output = cp.zeros((h * 2, w * 2, 3), dtype=h_input.dtype)
    d_tmp = cp.zeros((h * 2, w * 2, 3), dtype=h_input.dtype)

    pass1(((w + blk_size[0] - 1) // blk_size[0], (h + blk_size[1] - 1) // blk_size[1]), blk_size, (d_input, d_output))
    pass2(((w + blk_size[0] - 1) // blk_size[0], (h + blk_size[1] - 1) // blk_size[1]), blk_size, (d_output, d_tmp))
    pass3(((w * 2 + blk_size[0] - 1) // blk_size[0], (h * 2 + blk_size[1] - 1 )// blk_size[1]), blk_size, (d_tmp, d_output))

    h_out = cp.asnumpy(d_output)

    return h_out


# process
res = mufnp.numpy_process(
        [core.std.BlankClip(src, width=src.width*2, height=src.height*2), src], 
        superxbr_core, pass1=pass1, pass2=pass2, pass3=pass3, 
        input_per_plane=False, output_per_plane=False, channels_last=True, 
        omit_first_clip=True)


res.set_output()