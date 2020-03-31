# Modified from https://github.com/himesaka-noa/descale-verifier/blob/master/descale_verify.py

import vapoursynth as vs
import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
except BaseException:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import sys


core = vs.core

# parameters
src = core.lsmas.LWLibavSource(r"00004.m2ts")
src = core.std.SelectEvery(src, 100, 0)
src = core.resize.Point(src, format=vs.GRAYS)

descaled_width = 1280
descaled_height = 720
kernel = "bicubic"
a, b = 0, 0.5


if sys.platform != "win32":
    save_filename = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
else:
    save_filename = datetime.now().strftime("%y-%m-%d %H-%M-%S")


# functions
def calc_statistics(original, w, h, dw, dh, kernel, a, b):
    if kernel == 'bicubic':
        descaled = core.descale.Debicubic(original, dw, dh, b=a, c=b)
        rescaled = core.resize.Bicubic(descaled, w, h, filter_param_a=a, filter_param_b=b)
    elif kernel == 'bilinear':
        descaled = core.descale.Debilinear(original, dw, dh)
        rescaled = core.resize.Bilinear(descaled, w, h)
    elif kernel == 'lanczos':
        descaled = core.descale.Delanczos(original, dw, dh, taps=int(a))
        rescaled = core.resize.Lanczos(descaled, w, h, filter_param_a=int(a))
    elif kernel == 'spline16':
        descaled = core.descale.Despline16(original, dw, dh)
        rescaled = core.resize.Spline16(descaled, w, h)
    elif kernel == 'spline36':
        descaled = core.descale.Despline36(original, dw, dh)
        rescaled = core.resize.Spline36(descaled, w, h)
    else:
        raise NotImplementedError(f"Kernel {kernel} is not implemented.")

    return core.std.PlaneStats(original, rescaled)


def create_plot(data, save_filename):
    fig, ax = plt.subplots()
    t = np.arange(data.shape[0])
    ax.plot(t, data)
    ax.set(xlabel='frames', ylabel='difference', title='Descale Error')
    ax.grid()
    fig.savefig(f"{save_filename}.png")


def output_statistics(clip, save_filename):
    values = np.zeros(clip.num_frames)
    remaining_frames = clip.num_frames

    def filter_core(n, f, clip):
        values[n] = f.props["PlaneStatsDiff"]

        nonlocal remaining_frames
        remaining_frames -= 1

        if remaining_frames == 0:
            create_plot(values, save_filename)

        return clip

    clip = core.std.FrameEval(clip, partial(filter_core, clip=clip), clip)
    return clip


# process
clip = calc_statistics(src, src.width, src.height, descaled_width, descaled_height, kernel, a, b)
clip = output_statistics(clip, save_filename)
clip.set_output()
