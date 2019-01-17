# super resolution using OpenCV
# note: the input image to the network is not cropped, which might triggers out-of-memory error.

import os

# Set OpenCL device in format `<Platform>:<CPU|GPU|ACCELERATOR|nothing=GPU/CPU>:<DeviceName or ID>`
# examples: 'AMD:GPU:', ':GPU:1', 'Intel:CPU:', 
# https://github.com/opencv/opencv/wiki/OpenCL-optimizations#opencv-opencl-configuration-options
os.environ['OPENCV_OPENCL_DEVICE'] = 'NVIDIA:GPU:' # use GPU to accelerate processing


import vapoursynth as vs
import cv2
import mvsfunc as mvf
import muvsfunc_numpy as mufnp

core = vs.get_core()


# global params
src = core.std.BlankClip(width=640, height=360, length=1000, format=vs.RGBS) # can be RGB/YUV/GRAY
sr_algorithm = 0 # 0: waifu2x, 1: IDN, 2: TSCN, 3: VDSR, 4: DBPN (faster to slower)

if 'GPU' in os.environ['OPENCV_OPENCL_DEVICE']:
    if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_OPENCL # available on NVIDIA GPU since OpenCV 4.0.1, but only works on Intel GPU before OpenCV 3.4.2
    else:
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv2.dnn.DNN_TARGET_CPU


# params of the algos
if sr_algorithm == 0:
    # https://github.com/php-opencv/php-opencv-examples/tree/master/models/waifu2x
    # other models can be found at
    # https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Waifu2x-caffe/tree/master/Waifu2x-caffe/models
    sr_args = dict(prototxt=r'scale2.0x_model.prototxt', 
        caffe_model=r'scale2.0x_model.caffemodel', up_scale=2, is_rgb_model=True, pad=(7,7,7,7))

elif sr_algorithm == 1:
    # https://github.com/Zheng222/IDN-Caffe/tree/master/test/caffemodel
    sr_args = dict(prototxt=r'IDN_x2_deploy.prototxt', 
        caffe_model=r'IDN_x2.caffemodel', up_scale=2, is_rgb_model=False, pad=(1,1,1,1), crop=(1,2,1,2),
        upscale_uv=False, merge_residual=True)

elif sr_algorithm == 2:
    # https://github.com/Zheng222/TSCN/tree/master/test
    sr_args = dict(prototxt=r'TSCN_x2_deploy.prototxt',
        caffe_model=r'TSCN_x2.caffemodel', up_scale=2, is_rgb_model=False)

elif sr_algorithm == 3:
    # https://github.com/huangzehao/caffe-vdsr/tree/master/Train
    sr_args = dict(prototxt=r'VDSR_net_deploy.prototxt', 
        caffe_model=r'VDSR_Adam.caffemodel', up_scale=2, is_rgb_model=False, pre_upscale=True, upscale_uv=False)

elif sr_algorithm == 4:
    # https://github.com/alterzero/DBPN-caffe
    # https://drive.google.com/drive/folders/1ahbeoEHkjxoo4NV1wReOmpoRWbl448z-?usp=sharing
    sr_args = dict(prototxt=r'DBPN_mat_2x.prototxt',
        caffe_model=r'DBPN_2x.caffemodel', up_scale=2, is_rgb_model=True)


# internel functions
def channel_last(arr):
    """Convert a CHW array to HWC."""
    ndim = arr.ndim
    return arr.swapaxes(ndim - 3, ndim - 2).swapaxes(ndim - 2, ndim - 1)


def super_resolution_core(img, net, pad=None, crop=None):
    if pad is not None:
        img = cv2.copyMakeBorder(img, *pad, 1)

    blob = cv2.dnn.blobFromImage(img)

    net.setInput(blob, '')

    super_res = net.forward()

    if img.ndim == 2:
        if crop is not None:
            return super_res[0, 0, crop[0]:-crop[1], crop[2]:-crop[3]]
        else:
            return super_res[0, 0, :, :]
    else:
        # the output is BGR rather than RGB so channel reversal is needed
        if crop is not None:
            return channel_last(super_res[0, ::-1, crop[0]:-crop[1], crop[2]:-crop[3]])
        else:
            return channel_last(super_res[0, ::-1, :, :])


def run_super_resolution(clip, prototxt, caffe_model, up_scale=2, is_rgb_model=True, pad=None, crop=None, backend=None, target=None):
    """ Super-Resolution without color family hadling
    """

    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

    if backend is not None:
        net.setPreferableBackend(backend)

    if target is not None:
        net.setPreferableTarget(target)

    if up_scale != 1:
        blank = core.std.BlankClip(clip, width=clip.width*up_scale, height=clip.height*up_scale)
        super_res = mufnp.numpy_process([blank, clip], super_resolution_core, net=net, 
            input_per_plane=(not is_rgb_model), output_per_plane=(not is_rgb_model), pad=pad, crop=crop, 
            omit_first_clip=True)
    else:
        super_res = mufnp.numpy_process(clip, super_resolution_core, net=net, 
            input_per_plane=(not is_rgb_model), output_per_plane=(not is_rgb_model), pad=pad, crop=crop)

    return super_res


def super_resolution(clip, prototxt, caffe_model, up_scale=2, is_rgb_model=True, pad=None, crop=None, backend=None, target=None, pre_upscale=False, upscale_uv=False, merge_residual=False):
    """ Super-Resolution with color family hadling
    
    The color space of the output depends on the algorithm
    """

    isGray = clip.format.color_family == vs.GRAY
    isRGB = clip.format.color_family == vs.RGB

    if is_rgb_model and not isRGB:
        clip = mvf.ToRGB(clip, depth=32)

    elif not is_rgb_model:
        if isRGB:
            clip = mvf.ToYUV(clip, depth=32)

        if not isGray and not upscale_uv: # isYUV/RGB and only upscale Y
            clip = mvf.GetPlane(clip)

    clip = mvf.Depth(clip, depth=32)
    
    if pre_upscale:
        clip = core.resize.Bicubic(clip, clip.width*up_scale, clip.height*up_scale, filter_param_a=0, filter_param_b=0.5)
        up_scale = 1

    super_res = run_super_resolution(clip, prototxt=prototxt, caffe_model=caffe_model, 
        up_scale=up_scale, is_rgb_model=is_rgb_model, pad=pad, crop=crop, backend=backend, target=target)

    if merge_residual:
        low_res = core.resize.Bicubic(clip, super_res.width, super_res.height, filter_param_a=0, filter_param_b=0.5)
        super_res = core.std.Expr([super_res, low_res], ['x y +'])

    return super_res

sr = super_resolution(src, **sr_args, backend=backend, target=target)

# sr = core.caffe.Waifu2x(src, noise=-1, scale=2, cudnn=True, model=3)

sr.set_output()