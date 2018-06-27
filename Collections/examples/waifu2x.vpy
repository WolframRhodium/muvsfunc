import vapoursynth as vs
import cv2
import mvsfunc as mvf
import muvsfunc_numpy as mufnp

core = vs.get_core()

# waifu2x using OpenCV

src = core.std.BlankClip(width=720, height=480, length=1000, format=vs.RGBS) 

def channel_last(arr):
    """Convert a CHW array to HWC."""
    ndim = arr.ndim
    return arr.swapaxes(ndim - 3, ndim - 2).swapaxes(ndim - 2, ndim - 1)


def waifu2x_core(img, net):
    img = cv2.copyMakeBorder(img, 7, 7, 7, 7, 1)
    
    blob = cv2.dnn.blobFromImage(img)

    net.setInput(blob, '')

    r = net.forward()

    return channel_last(r[0, ::-1])


def waifu2x(clip, prototxt, caffe_model, backend=None, target=None):
    bits = clip.format.bits_per_sample
    
    rgb = mvf.ToRGB(clip, depth=32)
    blank = core.std.BlankClip(rgb, width=rgb.width*2, height=rgb.height*2)

    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

    if backend is not None:
        net.setPreferableBackend(backend)
    
    if target is not None:
        net.setPreferableTarget(target)

    sr = mufnp.numpy_process([blank, rgb], waifu2x_core, net=net, input_per_plane=False, output_per_plane=False, omit_first_clip=True)

    return mvf.ToRGB(sr, depth=bits)


# download link
# https://github.com/php-opencv/php-opencv-examples/tree/master/models/waifu2x
sr = waifu2x(src, r'scale2.0x_model.prototxt', r'scale2.0x_model.caffemodel', backend=cv2.dnn.DNN_BACKEND_DEFAULT, target=cv2.dnn.DNN_TARGET_CPU)

#sr = core.caffe.Waifu2x(down, noise=-1, scale=2, cudnn=True, model=3)

sr.set_output()