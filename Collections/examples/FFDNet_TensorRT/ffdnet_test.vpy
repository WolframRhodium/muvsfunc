import vapoursynth as vs
from vapoursynth import core

import sys
sys.path.append(".")
import vs_ffdnet

src = core.lsmas.LWLibavSource(r'PV02.mkv')
src = core.resize.Bicubic(src, 1920, 1080, format=vs.RGBS, matrix_in_s="709")
res = vs_ffdnet.FFDNet(src, sigma=5.0, use_cuda_graph=False)

res.set_output()

