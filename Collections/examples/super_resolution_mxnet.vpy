import vapoursynth as vs
import muvsfunc_numpy as mufnp

core = vs.get_core()

# super resolution using MXNet


# global params
src = core.std.BlankClip(width=720, height=480, length=1000, format=vs.RGBS) # can be RGB/YUV/GRAY
sr_algorithm = 0 # 0: waifu2x, 1: VDSR (faster to slower)
device_id = -1 # -1: CPU, 0, 1, ...: GPU


# params of the algos
# (download link for models: https://github.com/WolframRhodium/Super-Resolution-Zoo )
# use the information provided in "info.md" in the model's folder to set the paramter "sr_args"
if sr_algorithm == 0:
    sr_args = dict(model_prefix=r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model', device_id=device_id, block_w=128, block_h=128, pad=(7,7,7,7), is_caffe_model=True)

elif sr_algorithm == 3:
    sr_args = dict(model_prefix=r'VDSR (unofficial)\VDSR', device_id=device_id, block_w=128, block_h=128, is_rgb_model=False, pre_upscale=True, is_caffe_model=True)

sr = mufnp.super_resolution(src, **sr_args)

sr.set_output()