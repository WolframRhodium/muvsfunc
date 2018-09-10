import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0' # 0: False 1: Default 2: Full

import mxnet as mx

import vapoursynth as vs
import muvsfunc as muf
import muvsfunc_numpy as mufnp

core = vs.get_core()

# super resolution using MXNet


# global params
src = core.std.BlankClip(width=720, height=480, length=1000, format=vs.RGBS) # can be RGB/YUV/GRAY
sr_algorithm = 0 # 0: waifu2x, 1: VDSR (faster to slower)
device_id = 0 # -1: CPU, 0, 1, ...: GPU


# params of the algos
# (download link for models: https://github.com/WolframRhodium/Super-Resolution-Zoo )
# use the information provided in "info.md" in the model's folder to set the paramter "sr_args"
if sr_algorithm == 0:
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', device_id=device_id, block_w=128, block_h=128, pad=(7,7,7,7))

elif sr_algorithm == 1:
    sr_args = dict(model_filename=r'VDSR\pytorch-vdsr@twtygqyy\VDSR', device_id=device_id, block_w=128, block_h=128, is_rgb_model=False, pre_upscale=True)


# sr = mufnp.super_resolution(src, **sr_args) # using NumPy
sr = muf.super_resolution(src, **sr_args) # using C++ plugin

sr.set_output()