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
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', 
        device_id=device_id, block_w=128, block_h=128, 
        up_scale=2)


    # advanced I: padded upsampling to reduce blocking artifacts when small patch size is used

    r"""
    # both NumPy and C++ versions are available
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', 
        device_id=device_id, block_w=128, block_h=128, 
        up_scale=2, pre_upscale=False, pad=(0,5,0,5), crop=(0,10,0,10))
    """

    r"""
    # only NumPy version is available
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', 
        device_id=device_id, block_w=128, block_h=128, 
        up_scale=2, pre_upscale=False, pad=(5,5,5,5), crop=(10,10,10,10))
    """

    """ explanation:
    Suppose we want to upsample a 48x48 patch in an image. Let's denote such patch as X.
    The size of patch X is too small so that it's very likely to introduce blocking artifacts in the upsampled output.
    Thus we may want to feed the network with a 58x58 patch Y, where X is located at the center of Y.
    After processing by the network, we can then crop the output, to obtain an upsampled version of X with less blocking artifacts.

    Such procedure can be denoted as "pad=(5, 5, 5, 5), crop=(10, 10, 10, 10)".
    The value 5 is obtained by (58-48)/2=5, and the value 10 is obtained by 5*2=10, where 2 is the upsampling factor.
    Note that if pre_upscale is "True", the upsampling factor is always 1, regardless the value of "up_scale".
    """


    # advanced II: multi-GPU data parallelism

    r"""
    # only C++ version is available

    # 2 GPUS
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', 
        device_id=[0, 1], block_w=128, block_h=128, 
        up_scale=2)

    # 2 Queues
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', 
        device_id=[0, 0], block_w=128, block_h=128, 
        up_scale=2)

    # 4 Queues 2 GPUs
    sr_args = dict(model_filename=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', 
        device_id=[0, 1, 0, 1], block_w=128, block_h=128, 
        up_scale=2)
    """

# VDSR
elif sr_algorithm == 1:
    sr_args = dict(model_filename=r'VDSR\pytorch-vdsr@twtygqyy\VDSR', 
        device_id=device_id, block_w=128, block_h=128, 
        up_scale=2, is_rgb_model=False, pre_upscale=True)


# sr = mufnp.super_resolution(src, **sr_args) # using NumPy
sr = muf.super_resolution(src, **sr_args) # using C++ plugin

sr.set_output()