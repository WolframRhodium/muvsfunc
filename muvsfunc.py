'''
Functions:
    LDMerge
    Compare (2)
    ExInpand
    InDeflate
    MultiRemoveGrain
    GradFun3
    AnimeMask (2)
    PolygonExInpand
    Luma
    ediaa
    nnedi3aa
    maa
    SharpAAMcmod
    TEdge
    Sort
    Soothe_mod
    TemporalSoften
    FixTelecinedFades
    TCannyHelper
    MergeChroma
    firniture
    BoxFilter
    SmoothGrad
    DeFilter
    scale
    ColorBarsHD
    SeeSaw
    abcxyz
    Sharpen
    Blur
    BlindDeHalo3
    dfttestMC
    TurnLeft
    TurnRight
    BalanceBorders
    DisplayHistogram
    GuidedFilter (Color)
    GMSD
    SSIM
    SSIM_downsample
    LocalStatisticsMatching
    LocalStatistics
    TextSub16
    TMinBlur
    mdering
    BMAFilter
    LLSURE
    YAHRmod
    RandomInterleave
    super_resolution
    MDSI
    MaskedLimitFilter
    @multi_scale
    avg_decimate
    YAHRmask
    Cdeblend
    S_BoxFilter
    VFRSplice
    MSR
    getnative
    downsample
    SSFDeband
    pyramid_texture_filter
    flip
    temporal_dft
    temporal_idft
    srestore
'''

from collections import abc
import functools
import fractions
import itertools
import math
import numbers
import operator
import os
import typing
from typing import Any, Callable, Dict, Iterable, List, Optional
from typing import Sequence, Tuple, TypedDict, TypeVar, Union

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf


_is_api4: bool = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4

_has_lexpr: bool = (
    hasattr(core, "akarin") and
    b'x.property' in core.akarin.Version()["expr_features"]
)
_has_lexpr_lutspa: bool = (
    hasattr(core, "akarin") and
    b'X' in core.akarin.Version()["expr_features"]
)

# Type aliases
PlanesType = Optional[Union[int, Sequence[int]]]
VSFuncType = Union[vs.Func, Callable[..., vs.VideoNode]]

# Function alias
nnedi3: Optional[Callable[..., vs.VideoNode]] = core.nnedi3.nnedi3 if hasattr(core, "nnedi3") else None


def LDMerge(flt_h: vs.VideoNode, flt_v: vs.VideoNode, src: vs.VideoNode, mrad: int = 0,
            show: bool = False, planes: PlanesType = None,
            convknl: int = 1, conv_div: Optional[int] = None, calc_mode: int = 0,
            power: float = 1.0
            ) -> vs.VideoNode:
    """Merges two filtered clips based on the gradient direction map from a source clip.

    Args:
        flt_h, flt_v: Two filtered clips.

        src: Source clip. Must be the same format as the filtered clips.

        mrad: (int) Expanding of gradient direction map. Default is 0.

        show: (bool) Whether to output gradient direction map. Default is False.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the first clip, "flt_h".

        convknl: (0 or 1) Convolution kernel used to generate gradient direction map.
            0: Seconde order center difference in one direction and average in perpendicular direction
            1: First order center difference in one direction and weighted average in perpendicular direction.
            Default is 1.

        conv_div: (int) Divisor in convolution filter. Default is the max value in convolution kernel.

        calc_mode: (0 or 1) Method used to calculate the gradient direction map. Default is 0.

        power: (float) Power coefficient in "calc_mode=0".

    Example:
        # Fast anti-aliasing
        horizontal = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='h')
        vertical = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='v')
        blur_src = core.tcanny.TCanny(clip, mode=-1, planes=[0]) # Eliminate noise
        antialiasing = muf.LDMerge(horizontal, vertical, blur_src, mrad=1, planes=[0])

    """

    funcName = 'LDMerge'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')

    if not isinstance(flt_h, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_h\" must be a clip!')
    if src.format.id != flt_h.format.id:
        raise TypeError(funcName + ': \"flt_h\" must be of the same format as \"src\"!')
    if src.width != flt_h.width or src.height != flt_h.height:
        raise TypeError(funcName + ': \"flt_h\" must be of the same size as \"src\"!')

    if not isinstance(flt_v, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_v\" must be a clip!')
    if src.format.id != flt_v.format.id:
        raise TypeError(funcName + ': \"flt_v\" must be of the same format as \"src\"!')
    if src.width != flt_v.width or src.height != flt_v.height:
        raise TypeError(funcName + ': \"flt_v\" must be of the same size as \"src\"!')

    if not isinstance(mrad, int):
        raise TypeError(funcName + '\"mrad\" must be an int!')

    if not isinstance(show, int):
        raise TypeError(funcName + '\"show\" must be an int!')
    if show not in list(range(0, 4)):
        raise ValueError(funcName + '\"show\" must be in [0, 1, 2, 3]!')

    if planes is None:
        planes = list(range(flt_h.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    bits = flt_h.format.bits_per_sample

    if convknl == 0:
        convknl_h = [-1, -1, -1, 2, 2, 2, -1, -1, -1]
        convknl_v = [-1, 2, -1, -1, 2, -1, -1, 2, -1]
    else: # convknl == 1
        convknl_h = [-17, -61, -17, 0, 0, 0, 17, 61, 17]
        convknl_v = [-17, 0, 17, -61, 0, 61, -17, 0, 17]

    if conv_div is None:
        conv_div = max(convknl_h)

    hmap = core.std.Convolution(src, matrix=convknl_h, saturate=False, planes=planes, divisor=conv_div)
    vmap = core.std.Convolution(src, matrix=convknl_v, saturate=False, planes=planes, divisor=conv_div)

    if mrad > 0:
        hmap = haf_mt_expand_multi(hmap, sw=0, sh=mrad, planes=planes)
        vmap = haf_mt_expand_multi(vmap, sw=mrad, sh=0, planes=planes)
    elif mrad < 0:
        hmap = haf_mt_inpand_multi(hmap, sw=0, sh=-mrad, planes=planes)
        vmap = haf_mt_inpand_multi(vmap, sw=-mrad, sh=0, planes=planes)

    if calc_mode == 0:
        ldexpr = '{peak} 1 x 0.0001 + y 0.0001 + / {power} pow + /'.format(peak=(1 << bits) - 1, power=power)
    else:
        ldexpr = 'y 0.0001 + x 0.0001 + dup * y 0.0001 + dup * + sqrt / {peak} *'.format(peak=(1 << bits) - 1)
    ldmap = core.std.Expr([hmap, vmap], [(ldexpr if i in planes else '') for i in range(src.format.num_planes)])

    if show == 0:
        return core.std.MaskedMerge(flt_h, flt_v, ldmap, planes=planes)
    elif show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap
    else:
        raise ValueError


def Compare(src: vs.VideoNode, flt: vs.VideoNode, power: float = 1.5,
            chroma: bool = False, mode: int = 2
            ) -> vs.VideoNode:
    """Visualizes the difference between the source clip and filtered clip.

    Args:
        src: Source clip.

        flt: Filtered clip.

        power: (float) The variable in the processing function which controls the "strength" to increase difference. Default is 1.5.

        chroma: (bool) Whether to process chroma. Default is False.

        mode: (1 or 2) Different processing function. 1: non-linear; 2: linear.

    """

    funcName = 'Compare'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"src\" must be a YUV clip!')
    if not isinstance(flt, vs.VideoNode):
        raise TypeError(funcName + ': \"flt\" must be a clip!')
    if mode not in [1, 2]:
        raise TypeError(funcName + ': \"mode\" must be in [1, 2]!')

    Compare2(src, flt, props_list=['width', 'height', 'format.name'])

    isGray = src.format.color_family == vs.GRAY
    bits = src.format.bits_per_sample
    sample = src.format.sample_type

    expr = {}
    expr[1] = 'y x - abs 1 + {power} pow 1 -'.format(power=power)
    expr[2] = 'y x - {scale} * {neutral} +'.format(scale=32768 / (65536 ** (1 / power) - 1), neutral=32768)

    chroma = chroma or isGray

    if bits != 16:
        src = mvf.Depth(src, 16, sample=vs.INTEGER)
        flt = mvf.Depth(flt, 16, sample=vs.INTEGER)
        diff = core.std.Expr([src, flt], [expr[mode]] if chroma else [expr[mode], '{neutral}'.format(neutral=32768)])
        diff = mvf.Depth(diff, depth=bits, sample=sample, fulls=True, fulld=True, dither="none", ampo=0, ampn=0)
    else:
        diff = core.std.Expr([src, flt], [expr[mode]] if chroma else [expr[mode], '{neutral}'.format(neutral=32768)])

    return diff


def Compare2(clip1: vs.VideoNode, clip2: vs.VideoNode,
             props_list: Optional[Sequence[str]] = None
             ) -> None:
    """Compares the formats of two clips.

    TypeError will be raised when one of the format of two clips are not identical.
    Otherwise, None is returned.

    Args:
        clip1, clip2: Input.

        props_list: (list of strings) A list containing the format to be compared.
            If it is none, all the formats will be compared.
            Default is None.

    """

    funcName = 'Compare2'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')

    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if props_list is None:
        props_list = ['width', 'height', 'num_frames', 'fps', 'format.name']

    info = ''

    for prop in props_list:
        clip1_prop = eval('clip1.{prop}'.format(prop=prop))
        clip2_prop = eval('clip2.{prop}'.format(prop=prop))

        if clip1_prop != clip2_prop:
            info += '{prop}: {clip1_prop} != {clip2_prop}\n'.format(prop=prop, clip1_prop=clip1_prop, clip2_prop=clip2_prop)

    if info != '':
        info = '\n\n{}'.format(info)

        raise TypeError(info)


def ExInpand(input: vs.VideoNode, mrad: Union[int, Sequence[int], Sequence[Sequence[int]]] = 0,
             mode: Union[int, str, Sequence[Union[int, str]]] = 'rectangle',
             planes: PlanesType = None
             ) -> vs.VideoNode:
    """A filter to simplify the calls of std.Maximum()/std.Minimum() and their concatenation.

    Args:
        input: Source clip.

        mrad: (int []) How many times to use std.Maximum()/std.Minimum(). Default is 0.
            Positive value indicates to use std.Maximum().
            Negative value indicates to use std.Minimum().
            Values can be put into a list, or a list of lists.

            Example:
                mrad=[2, -1] is equvalant to clip.std.Maximum().std.Maximum().std.Minimum()
                mrad=[[2, 1], [2, -1]] is equivalant to
                    haf_mt_expand_multi(clip, sw=2, sh=1).std.Maximum().std.Maximum().std.Minimum()

        mode: (0:"rectangle", 1:"losange" or 2:"ellipse", int [] or str []). Default is "rectangle"
            The shape of the kernel.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    funcName = 'ExInpand'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if isinstance(mrad, int):
        mrad = [mrad]
    elif not isinstance(mrad, abc.Sequence):
        raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or lists of two ints!')

    if isinstance(mode, (str, int)):
        mode = [mode]
    elif not isinstance(mode, abc.Sequence):
        raise TypeError(funcName + ': \"mode\" must be an int, a string, a list of ints, strings or unions of ints and strings!')

    # internel function
    def ExInpand_process(input: vs.VideoNode, mode: Union[int, str],
                         planes: Union[int, Sequence[int]],
                         mrad: Union[int, Tuple[int, int]]) -> vs.VideoNode:
        if isinstance(mode, int):
            mode = ['rectangle', 'losange', 'ellipse'][mode]
        elif isinstance(mode, str):
            mode = mode.lower()
            if mode not in ['rectangle', 'losange', 'ellipse']:
                raise ValueError(funcName + ': \"mode\" must be an int in [0, 2] or a specific string in [\"rectangle\", \"losange\", \"ellipse\"]!')
        else:
            raise TypeError(funcName + ': \"mode\" must be an int in [0, 2] or a specific string in [\"rectangle\", \"losange\", \"ellipse\"]!')

        if isinstance(mrad, int):
            sw = sh = mrad
        else:
            sw, sh = mrad

        if sw * sh < 0:
            raise TypeError(funcName + ': \"mrad\" at a time must be both positive or negative!')

        if sw > 0 or sh > 0:
            return haf_mt_expand_multi(input, mode=mode, planes=planes, sw=sw, sh=sh)
        else:
            return haf_mt_inpand_multi(input, mode=mode, planes=planes, sw=-sw, sh=-sh)

    # process
    for i in range(len(mrad)):
        if isinstance(mrad[i], abc.Sequence):
            for n in mrad[i]: # type: ignore
                if not isinstance(n, int):
                    raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or lists of two ints!')

            if len(mrad[i]) == 1: # type: ignore
                _mrad = (mrad[i][0], mrad[i][0]) # type: ignore
            elif len(mrad[i]) == 2: # type: ignore
                _mrad = tuple(mrad[i]) # type: ignore
            else:
                raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or lists of two ints!')
        elif isinstance(mrad[i], int):
            _mrad = mrad[i] # type: ignore
        else:
            raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or lists of two ints!')

        clip = ExInpand_process(input, mode=mode[min(i, len(mode) - 1)], mrad=_mrad, planes=planes)

    return clip


def InDeflate(input: vs.VideoNode, msmooth: Union[int, Sequence[int]] = 0,
              planes: PlanesType = None
              ) -> vs.VideoNode:
    """A filter to simplify the calls of std.Inflate()/std.Deflate() and their concatenation.

    Args:
        input: Source clip.

        msmooth: (int []) How many times to use std.Inflate()/std.Deflate(). Default is 0.
            The behaviour is the same as "mode" in ExInpand().

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    funcName = 'InDeFlate'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if isinstance(msmooth, int):
        msmoooth = [msmooth]

    # internel function
    def InDeflate_process(input: vs.VideoNode, radius: int,
                          planes: PlanesType = None
                          ) -> vs.VideoNode:
        if radius > 0:
            return haf_mt_inflate_multi(input, planes=planes, radius=radius)
        else:
            return haf_mt_deflate_multi(input, planes=planes, radius=-radius)

    # process
    if isinstance(msmooth, list):
        for m in msmooth:
            if not isinstance(m, int):
                raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')
            else:
                clip = InDeflate_process(input, radius=m, planes=planes)
    else:
        raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')

    return clip


def MultiRemoveGrain(input: vs.VideoNode, mode: Union[int, Sequence[int]] = 0,
                     loop: int = 1
                     ) -> vs.VideoNode:
    """A filter to simplify the calls of rgvs.RemoveGrain().

    Args:
        input: Source clip.

        mode: (int []) "mode" in rgvs.RemoveGrain().
            Can be a list, the logic is similar to "mode" in ExInpand().

            Example: mode=[4, 11, 11] is equivalant to clip.rgvs.RemoveGrain(4).rgvs.RemoveGrain(11).rgvs.RemoveGrain(11)
            Default is 0.

        loop: (int) How many times the "mode" loops.

    """

    funcName = 'MultiRemoveGrain'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if isinstance(mode, int):
        mode = [mode]

    if not isinstance(loop, int):
        raise TypeError(funcName + ': \"loop\" must be an int!')
    if loop < 0:
        raise ValueError(funcName + ': \"loop\" must be positive value!')

    if isinstance(mode, list):
        for i in range(loop):
            for m in mode:
                clip = core.rgvs.RemoveGrain(input, mode=m)
    else:
        raise TypeError(funcName + ': \"mode\" must be an int, a list of ints or a list of a list of ints!')

    return clip


def GradFun3(src: vs.VideoNode, thr: float = 0.35, radius: Optional[int] = None,
             elast: float = 3.0, mask: int = 2, mode: Optional[int] = None,
             ampo: Optional[float] = None, ampn: Optional[float] = None,
             pat: Optional[int] = None, dyn: Optional[int] = None, lsb: bool = False,
             staticnoise: Optional[int] = None, smode: int = 1,
             thr_det: Optional[float] = None, debug: bool = False,
             thrc: Optional[float] = None, radiusc: Optional[int] = None,
             elastc: Optional[float] = None, planes: PlanesType = None,
             ref: Optional[vs.VideoNode] = None
             ) -> vs.VideoNode:
    """GradFun3 by Firesledge v0.1.1

    Port by Muonium  2016/6/18
    Port from Dither_tools v1.27.2 (http://avisynth.nl/index.php/Dither_tools)
    Internal precision is always 16 bits.

    Read the document of Avisynth version for more details.

    Notes:
        1. In this function I try to keep the original look of GradFun3 in Avisynth.
            It should be better to use Frechdachs's GradFun3 in his fvsfunc.py
            (https://github.com/Irrational-Encoding-Wizardry/fvsfunc) which is more novel and powerful.

    Removed parameters:
        "dthr", "wmin", "thr_edg", "subspl", "lsb_in"

    Parameters "y", "u", "v" are changed into "planes"

    """

    funcName = 'GradFun3'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.YUV, vs.GRAY]:
        raise TypeError(funcName + ': \"src\" must be YUV or GRAY color family!')

    if not isinstance(thr, (float, int)):
        raise TypeError(funcName + ': \"thr\" must be an int or a float!')

    if smode not in [0, 1, 2, 3]:
        raise ValueError(funcName + ': \"smode\" must be in [0, 1, 2, 3]!')

    if radius is None:
        radius = (16 if src.width > 1024 or src.height > 576 else 12) if (smode == 1 or smode == 2) else 9
    elif isinstance(radius, int):
        if radius <= 0:
            raise ValueError(funcName + ': \"radius\" must be strictly positive.')
    else:
        raise TypeError(funcName + ': \"radius\" must be an int!')

    if isinstance(elast, (int, float)):
        if elast < 1:
            raise ValueError(funcName + ': Valid range of \"elast\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elast\" must be an int or a float!')

    if not isinstance(mask, int):
        raise TypeError(funcName + ': \"mask\" must be an int!')

    if thr_det is None:
        thr_det = 2 + round(max(thr - 0.35, 0) / 0.3)
    elif isinstance(thr_det, (int, float)):
        if thr_det <= 0.0:
            raise ValueError(funcName + '" \"thr_det\" must be strictly positive!')
    else:
        raise TypeError(funcName + ': \"mask\" must be an int or a float!')

    if not isinstance(debug, bool) and debug not in [0, 1]:
        raise TypeError(funcName + ': \"debug\" must be a bool!')

    if thrc is None:
        thrc = thr

    if radiusc is None:
        radiusc = radius
    elif isinstance(radiusc, int):
        if radiusc <= 0:
            raise ValueError(funcName + '\"radiusc\" must be strictly positive.')
    else:
        raise TypeError(funcName + '\"radiusc\" must be an int!')

    if elastc is None:
        elastc = elast
    elif isinstance(elastc, (int, float)):
        if elastc < 1:
            raise ValueError(funcName + ':valid range of \"elastc\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elastc\" must be an int or a float!')

    if planes is None:
        planes = list(range(src.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if ref is None:
        ref = src
    elif not isinstance(ref, vs.VideoNode):
        raise TypeError(funcName + ': \"ref\" must be a clip!')
    elif ref.format.color_family not in [vs.YUV, vs.GRAY]:
        raise TypeError(funcName + ': \"ref\" must be YUV or GRAY color family!')
    elif src.width != ref.width or src.height != ref.height:
        raise TypeError(funcName + ': \"ref\" must be of the same size as \"src\"!')

    bits = src.format.bits_per_sample
    src_16 = core.fmtc.bitdepth(src, bits=16, planes=planes) if bits < 16 else src
    src_8 = core.fmtc.bitdepth(src, bits=8, dmode=1, planes=[0]) if bits != 8 else src
    if src is ref:
        ref_16 = src_16
    else:
        ref_16 = core.fmtc.bitdepth(ref, bits=16, planes=planes) if ref.format.bits_per_sample < 16 else ref

    # Main debanding
    """
    chroma_flag: Whether we need to process Y and UV separately. It's True when:
        Y is processed;
        at least one from UV is processed;
        Y and UV use different parameters.
    """
    chroma_flag = (thrc != thr or radiusc != radius or
                   elastc != elast) and 0 in planes and (1 in planes or 2 in planes)

    if chroma_flag:
        planes2 = [0]
    else:
        planes2 = list(planes)

    if not planes2:
        raise ValueError(funcName + ': no plane is processed!')

    flt_y = _GF3_smooth(src_16, ref_16, smode, radius, thr, elast, planes2)
    if chroma_flag:
        planes2 = [i for i in planes if i > 0]
        flt_c = _GF3_smooth(src_16, ref_16, smode, radiusc, thrc, elastc, planes2)
        flt = core.std.ShufflePlanes([flt_y, flt_c], list(range(src.format.num_planes)), src.format.color_family)
    else:
        flt = flt_y

    # Edge/detail mask
    td_lo = max(thr_det * 0.75, 1.0)
    td_hi = max(thr_det, 1.0)
    mexpr = 'x {tl} - {th} {tl} - / 255 *'.format(tl=td_lo - 0.0001, th=td_hi + 0.0001)

    if mask > 0:
        dmask = mvf.GetPlane(src_8, 0)
        dmask = _Build_gf3_range_mask(dmask, mask)
        dmask = core.std.Expr([dmask], [mexpr])
        dmask = core.rgvs.RemoveGrain(dmask, [22])
        if mask > 1:
            dmask = core.std.Convolution(dmask, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
            if mask > 2:
                dmask = core.std.Convolution(dmask, matrix=[1]*9)
        dmask = core.resize.Point(dmask, format=vs.GRAY16)
        res_16 = core.std.MaskedMerge(flt, src_16, dmask, planes=planes, first_plane=True)
    else:
        res_16 = flt

    # Dithering
    result = res_16 if lsb or bits >= 16 else core.fmtc.bitdepth(res_16, bits=bits, planes=planes, dmode=mode,
                                                                 ampo=ampo, ampn=ampn, dyn=dyn,
                                                                 staticnoise=staticnoise, patsize=pat)

    if debug:
        last = dmask
        if not lsb:
            last = core.fmtc.bitdepth(last, bits=8, fulls=True, fulld=True)
    else:
        last = result

    return last


def _GF3_smooth(src_16: vs.VideoNode, ref_16: vs.VideoNode, smode: int, radius: int,
                thr: float, elast: float, planes: PlanesType
                ) -> vs.VideoNode:
    funcName = "_GF3_smooth"

    if smode == 0:
        return _GF3_smoothgrad_multistage(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 1:
        return _GF3_dfttest(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 2:
        return _GF3_bilateral_multistage(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 3:
        return _GF3_smoothgrad_multistage_3(src_16, radius, thr, elast, planes)
    else:
        raise ValueError(funcName + ': wrong smode value!')


def _GF3_smoothgrad_multistage(src: vs.VideoNode, ref: vs.VideoNode, radius: int,
                               thr: float, elast: float, planes: PlanesType
                               ) -> vs.VideoNode:
    ela_2 = max(elast * 0.83, 1.0)
    ela_3 = max(elast * 0.67, 1.0)
    r2 = radius * 2 // 3
    r3 = radius * 3 // 3
    r4 = radius * 4 // 4
    last = src
    last = SmoothGrad(last, radius=r2, thr=thr, elast=elast, ref=ref, planes=planes) if r2 >= 1 else last
    last = SmoothGrad(last, radius=r3, thr=thr * 0.7, elast=ela_2, ref=ref, planes=planes) if r3 >= 1 else last
    last = SmoothGrad(last, radius=r4, thr=thr * 0.46, elast=ela_3, ref=ref, planes=planes) if r4 >= 1 else last
    return last


def _GF3_smoothgrad_multistage_3(src: vs.VideoNode, radius: int, thr: float,
                                 elast: float, planes: PlanesType
                                 ) -> vs.VideoNode:
    ref = SmoothGrad(src, radius=radius // 3, thr=thr * 0.8, elast=elast)
    last = BoxFilter(src, radius=radius, planes=planes)
    last = BoxFilter(last, radius=radius, planes=planes)
    last = mvf.LimitFilter(last, src, thr=thr * 0.6, elast=elast, ref=ref, planes=planes)
    return last


def _GF3_dfttest(src: vs.VideoNode, ref: vs.VideoNode, radius: int,
                 thr: float, elast: float, planes: PlanesType
                 ) -> vs.VideoNode:
    hrad = max(radius * 3 // 4, 1)
    last = core.dfttest.DFTTest(src, sigma=hrad * thr * thr * 32, sbsize=hrad * 4,
                                sosize=hrad * 3, tbsize=1, planes=planes)
    last = mvf.LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)

    return last


def _GF3_bilateral_multistage(src: vs.VideoNode, ref: vs.VideoNode, radius: int,
                              thr: float, elast: float, planes: PlanesType
                              ) -> vs.VideoNode:
    last = core.bilateral.Bilateral(src, ref=ref, sigmaS=radius / 2, sigmaR=thr / 255, planes=planes, algorithm=0)

    last = mvf.LimitFilter(last, src, thr=thr, elast=elast, planes=planes)

    return last


def _Build_gf3_range_mask(src: vs.VideoNode, radius: int = 1) -> vs.VideoNode:
    last = src

    if radius > 1:
        ma = haf_mt_expand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        mi = haf_mt_inpand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        last = core.std.Expr([ma, mi], ['x y -'])
    else:
        bits = src.format.bits_per_sample
        black = 0
        white = (1 << bits) - 1
        maxi = core.std.Maximum(last, [0])
        mini = core.std.Minimum(last, [0])
        exp = "x y -"
        exp2 = "x {thY1} < {black} x ? {thY2} > {white} x ?".format(thY1=0, thY2=255, black=black, white=white)
        last = core.std.Expr([maxi,mini], [exp])
        last = core.std.Expr([last], [exp2])

    return last


def AnimeMask(input: vs.VideoNode, shift: float = 0, expr: Optional[str] = None,
              mode: int = 1, **resample_args: Any
              ) -> vs.VideoNode:
    """Generates edge/ringing mask for anime based on gradient operator.

    For Anime's ringing mask, it's recommended to set "shift" between 0.5 and 1.0.

    Args:
        input: Source clip. Only the First plane will be processed.

        shift: (float, -1.5 ~ 1.5) The distance of translation. Default is 0.

        expr: (string) Subsequent processing in std.Expr(). Default is "".

        mode: (-1 or 1) Type of the kernel, which simply inverts the pixel values and "shift".
            Typically, -1 is for edge, 1 is for ringing. Default is 1.

        resample_args: (dict) Additional parameters passed to core.resize in the form of dict.

    """

    funcName = 'AnimeMask'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    if mode not in [-1, 1]:
        raise ValueError(funcName + ': \'mode\' have not a correct value! [-1 or 1]')

    if mode == -1:
        input = core.std.Invert(input)
        shift = -shift

    full_args = dict(range_s="full", range_in_s="full")
    mask1 = core.std.Convolution(input, [0, 0, 0, 0, 2, -1, 0, -1, 0], saturate=True).resize.Bicubic(src_left=shift,
        src_top=shift, **full_args, **resample_args) # type: ignore
    mask2 = core.std.Convolution(input, [0, -1, 0, -1, 2, 0, 0, 0, 0], saturate=True).resize.Bicubic(src_left=-shift,
        src_top=-shift, **full_args, **resample_args) # type: ignore
    mask3 = core.std.Convolution(input, [0, -1, 0, 0, 2, -1, 0, 0, 0], saturate=True).resize.Bicubic(src_left=shift,
        src_top=-shift, **full_args, **resample_args) # type: ignore
    mask4 = core.std.Convolution(input, [0, 0, 0, -1, 2, 0, 0, -1, 0], saturate=True).resize.Bicubic(src_left=-shift,
        src_top=shift, **full_args, **resample_args) # type: ignore

    calc_expr = 'x x * y y * + z z * + a a * + sqrt '

    if isinstance(expr, str):
        calc_expr += expr

    mask = core.std.Expr([mask1, mask2, mask3, mask4], [calc_expr])

    return mask


def AnimeMask2(input: vs.VideoNode, r: float = 1.2, expr: Optional[str] = None,
               mode: int = 1
               ) -> vs.VideoNode:
    """Yet another filter to generate edge/ringing mask for anime.

    More specifically, it's an approximatation of the difference of gaussians filter based on resampling.

    Args:
        input: Source clip. Only the First plane will be processed.

        r: (float, positive) Radius of resampling coefficient. Default is 1.2.

        expr: (string) Subsequent processing in std.Expr(). Default is "".

        mode: (-1 or 1) Type of the kernel. Typically, -1 is for edge, 1 is for ringing. Default is 1.

    """

    funcName = 'AnimeMask2'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    w = input.width
    h = input.height

    if mode not in [-1, 1]:
        raise ValueError(funcName + ': \'mode\' have not a correct value! [-1 or 1]')

    smooth = core.resize.Bicubic(input, haf_m4(w / r), haf_m4(h / r), filter_param_a=1/3, filter_param_b=1/3).resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    smoother = core.resize.Bicubic(input, haf_m4(w / r), haf_m4(h / r), filter_param_a=1/3, filter_param_b=1/3).resize.Bicubic(w, h, filter_param_a=1.5, filter_param_b=-0.25)

    calc_expr = 'x y - ' if mode == 1 else 'y x - '

    if isinstance(expr, str):
        calc_expr += expr

    mask = core.std.Expr([smooth, smoother], [calc_expr])

    return mask


def PolygonExInpand(input: vs.VideoNode, shift: float = 0, shape: int = 0, mixmode: int = 0,
                    noncentral: bool = False, step: float = 1, amp: float = 1,
                    **resample_args: Any
                    ) -> vs.VideoNode:
    """Processes mask based on resampling.

    Args:
        input: Source clip. Only the First plane will be processed.

        shift: (float) Distance of expanding/inpanding. Default is 0.

        shape: (int, 0:losange, 1:square, 2:octagon) The shape of expand/inpand kernel. Default is 0.

        mixmode: (int, 0:max, 1:arithmetic mean, 2:quadratic mean)
            Method used to calculate the mix of different mask. Default is 0.

        noncentral: (bool) Whether to calculate the center pixel in mix process.

        step: (float) Step of expanding/inpanding. Default is 1.

        amp: (float) Linear multiple to strengthen the final mask. Default is 1.

        resample_args: (dict) Additional parameters passed to core.resize in the form of dict.

    """

    funcName = 'PolygonExInpand'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if shape not in list(range(3)):
        raise ValueError(funcName + ': \'shape\' have not a correct value! [0, 1 or 2]')

    if mixmode not in list(range(3)):
        raise ValueError(funcName + ': \'mixmode\' have not a correct value! [0, 1 or 2]')

    if step <= 0:
        raise ValueError(funcName + ': \'step\' must be positive!')

    invert = False
    if shift < 0:
        invert = True
        input = core.std.Invert(input)
        shift = -shift
    elif shift == 0.:
        return input

    mask5 = input

    while shift > 0:
        step = min(step, shift)
        shift = shift - step

        ortho = step
        inv_ortho = -step
        dia = math.sqrt(step / 2)
        inv_dia = -math.sqrt(step / 2)

        # shift
        if shape == 0 or shape == 2:
            mask2 = core.resize.Bilinear(mask5, src_left=0, src_top=ortho, **resample_args)
            mask4 = core.resize.Bilinear(mask5, src_left=ortho, src_top=0, **resample_args)
            mask6 = core.resize.Bilinear(mask5, src_left=inv_ortho, src_top=0, **resample_args)
            mask8 = core.resize.Bilinear(mask5, src_left=0, src_top=inv_ortho, **resample_args)

        if shape == 1 or shape == 2:
            mask1 = core.resize.Bilinear(mask5, src_left=dia, src_top=dia, **resample_args)
            mask3 = core.resize.Bilinear(mask5, src_left=inv_dia, src_top=dia, **resample_args)
            mask7 = core.resize.Bilinear(mask5, src_left=dia, src_top=inv_dia, **resample_args)
            mask9 = core.resize.Bilinear(mask5, src_left=inv_dia, src_top=inv_dia, **resample_args)

        # mix
        if noncentral:
            expr_list = [
                'x y max z max a max',
                'x y + z + a + 4 /',
                'x x * y y * + z z * + a a * + 4 / sqrt',
                'x y max z max a max b max c max d max e max',
                'x y + z + a + b + c + d + e + 8 /',
                'x x * y y * + z z * + a a * + b b * + c c * + d d * + e e * + 8 / sqrt',
                ]

            if shape == 0 or shape == 1:
                expr = expr_list[mixmode] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask2, mask4, mask6, mask8] if shape == 0 else [mask1, mask3, mask7, mask9], [expr])
            else: # shape == 2
                expr = expr_list[mixmode + 3] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask1, mask2, mask3, mask4, mask6, mask7, mask8, mask9], [expr])
        else: # noncentral == False
            expr_list = [
                'x y max z max a max b max',
                'x y + z + a + b + 5 /',
                'x x * y y * + z z * + a a * + b b * + 5 / sqrt',
                'x y max z max a max b max c max d max e max f max',
                'x y + z + a + b + c + d + e + f + 9 /',
                'x x * y y * + z z * + a a * + b b * + c c * + d d * + e e * + f f * + 9 / sqrt',
                ]

            if (shape == 0) or (shape == 1):
                expr = expr_list[mixmode] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask2, mask4, mask5, mask6, mask8] if shape == 0 else
                    [mask1, mask3, mask5, mask7, mask9], [expr])
            else: # shape == 2
                expr = expr_list[mixmode + 3] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9], [expr])

    return core.std.Invert(mask5) if invert else mask5


def Luma(input: vs.VideoNode, plane: int = 0, power: int = 4) -> vs.VideoNode:
    """std.Lut() implementation of Luma() in Histogram() filter.

    Args:
        input: Source clip. Only one plane will be processed.

        plane: (int) Which plane to be processed. Default is 0.

        power: (int) Coefficient in processing. Default is 4.

    """

    funcName = 'Luma'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if (input.format.sample_type != vs.INTEGER):
        raise TypeError(funcName + ': \"input\" must be of integer format!')

    bits = input.format.bits_per_sample
    peak = (1 << bits) - 1

    clip = mvf.GetPlane(input, plane)


    def calc_luma(x: int) -> int:
        p = x << power
        return (peak - (p & peak)) if (p & (peak + 1)) else (p & peak)

    return core.std.Lut(clip, function=calc_luma)


def ediaa(a: vs.VideoNode) -> vs.VideoNode:
    """Suggested by Mystery Keeper in "Denoise of tv-anime" thread

    Read the document of Avisynth version for more details.

    """

    funcName = 'ediaa'

    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    last = core.eedi2.EEDI2(a, field=1).std.Transpose()
    last = core.eedi2.EEDI2(last, field=1).std.Transpose()
    last = core.resize.Spline36(last, a.width, a.height, src_left=-0.5, src_top=-0.5)

    return last


def nnedi3aa(a: vs.VideoNode) -> vs.VideoNode:
    """Using nnedi3 (Emulgator):

    Read the document of Avisynth version for more details.

    """

    funcName = 'nnedi3aa'

    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    if nnedi3 and callable(nnedi3):
        last = nnedi3(a, field=1, dh=True).std.Transpose()
        last = nnedi3(last, field=1, dh=True).std.Transpose()
    else:
        raise RuntimeError("nnedi3 not found")
    last = core.resize.Spline36(last, a.width, a.height, src_left=-0.5, src_top=-0.5)

    return last


def maa(input: vs.VideoNode) -> vs.VideoNode:
    """Anti-aliasing with edge masking by martino,
    mask using "sobel" taken from Kintaro's useless filterscripts and modded by thetoof for spline36

    Read the document of Avisynth version for more details.

    """

    funcName = 'maa'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    w = input.width
    h = input.height
    bits = input.format.bits_per_sample

    if input.format.color_family != vs.GRAY:
        input_src = input # type: Optional[vs.VideoNode]
        input = mvf.GetPlane(input, 0)
    else:
        input_src = None

    mask = core.std.Convolution(input, [0, -1, 0, -1, 0, 1, 0, 1, 0], divisor=2, saturate=False).std.Binarize(scale(7, bits) + 1)
    aa_clip = core.resize.Spline36(input, w * 2, h * 2)
    aa_clip = core.sangnom.SangNom(aa_clip).std.Transpose()
    aa_clip = core.sangnom.SangNom(aa_clip).std.Transpose()
    aa_clip = core.resize.Spline36(aa_clip, w, h)
    last = core.std.MaskedMerge(input, aa_clip, mask)

    if input_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, input_src], planes=list(range(input_src.format.num_planes)),
            colorfamily=input_src.format.color_family)


def SharpAAMcmod(orig: vs.VideoNode, dark: float = 0.2, thin: int = 10, sharp: int = 150,
                 smooth: int = -1, stabilize: bool = False, tradius: int = 2, aapel: int = 1,
                 aaov: Optional[int] = None, aablk: Optional[int] = None, aatype: str = 'nnedi3'
                 ) -> vs.VideoNode:
    """High quality MoComped AntiAliasing script.

    Also a line darkener since it uses edge masking to apply tweakable warp-sharpening,
    "normal" sharpening and line darkening with optional temporal stabilization of these edges.
    Part of AnimeIVTC.

    Author: thetoof. Developed in the "fine anime antialiasing thread".

    Only the first plane (luma) will be processed.

    Args:
        orig: Source clip. Only the first plane will be processed.

        dark: (float) Strokes darkening strength. Default is 0.2.

        thin: (int) Presharpening. Default is 10.

        sharp: (int) Postsharpening. Default is 150.

        smooth: (int) Postsmoothing. Default is -1.

        stabilize: (bool) Use post stabilization with Motion Compensation. Default is False.

        tradius: (1~3) 1 = Degrain1 / 2 = Degrain2 / 3 = Degrain3. Default is 2.

        aapel: (int) Accuracy of the motion estimation. Default is 1
            (Value can only be 1, 2 or 4.
            1 means a precision to the pixel.
            2 means a precision to half a pixel,
            4 means a precision to quarter a pixel,
            produced by spatial interpolation (better but slower).)

        aaov: (int) Block overlap value (horizontal). Default is None.
            Must be even and less than block size.(Higher = more precise & slower)

        aablk: (4, 8, 16, 32, 64, 128) Size of a block (horizontal). Default is 8.
            Larger blocks are less sensitive to noise, are faster, but also less accurate.

        aatype: ("sangnom", "eedi2" or "nnedi3"). Default is "nnedi3".
            Use Sangnom() or EEDI2() or NNEDI3() for anti-aliasing.

    """

    funcName = 'SharpAAMcmod'

    if not isinstance(orig, vs.VideoNode):
        raise TypeError(funcName + ': \"orig\" must be a clip!')

    w = orig.width
    h = orig.height
    bits = orig.format.bits_per_sample

    if orig.format.color_family != vs.GRAY:
        orig_src = orig # type: Optional[vs.VideoNode]
        orig = mvf.GetPlane(orig, 0)
    else:
        orig_src = None

    if aaov is None:
        aaov = 8 if w > 1100 else 4

    if aablk is None:
        aablk = 16 if w > 1100 else 8

    m = core.std.Expr([core.std.Convolution(orig, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
        core.std.Convolution(orig, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
        ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (bits-1), peak=(1 << bits)-1)])

    if thin == 0 and dark == 0.:
        preaa = orig
    elif thin == 0:
        preaa = haf_Toon(orig, str=dark)
    elif dark == 0.:
        preaa = core.warp.AWarpSharp2(orig, depth=thin)
    else:
        preaa = haf_Toon(orig, str=dark).warp.AWarpSharp2(depth=thin)

    aatype = aatype.lower()
    if aatype == 'sangnom':
        aa = core.resize.Spline36(preaa, w * 2, h * 2)
        aa = core.std.Transpose(aa).sangnom.SangNom()
        aa = core.std.Transpose(aa).sangnom.SangNom()
        aa = core.resize.Spline36(aa, w, h)
    elif aatype == 'eedi2':
        aa = ediaa(preaa)
    elif aatype == 'nnedi3':
        aa = nnedi3aa(preaa)
    else:
        raise ValueError(funcName + ': valid values of \"aatype\" are \"sangnom\", \"eedi2\" and \"nnedi3\"!')

    if sharp == 0 and smooth == 0:
        postsh = aa
    else:
        postsh = haf_LSFmod(aa, strength=sharp, overshoot=1, soft=smooth, edgemode=1)

    merged = core.std.MaskedMerge(orig, postsh, m)

    if stabilize:
        sD = core.std.MakeDiff(orig, merged)

        origsuper = haf_DitherLumaRebuild(orig, s0=1).mv.Super(pel=aapel)
        sDsuper = core.mv.Super(sD, pel=aapel)

        if tradius >= 1:
            fv1 = core.mv.Analyse(origsuper, isb=False, delta=1, overlap=aaov, blksize=aablk)
            bv1 = core.mv.Analyse(origsuper, isb=True, delta=1, overlap=aaov, blksize=aablk)
            sDD = core.mv.Degrain1(sD, sDsuper, bv1, fv1)
            if tradius >= 2:
                fv2 = core.mv.Analyse(origsuper, isb=False, delta=2, overlap=aaov, blksize=aablk)
                bv2 = core.mv.Analyse(origsuper, isb=True, delta=2, overlap=aaov, blksize=aablk)
                sDD = core.mv.Degrain2(sD, sDsuper, bv1, fv1, bv2, fv2)
                if tradius == 3:
                    fv3 = core.mv.Analyse(origsuper, isb=False, delta=3, overlap=aaov, blksize=aablk)
                    bv3 = core.mv.Analyse(origsuper, isb=True, delta=3, overlap=aaov, blksize=aablk)
                    sDD = core.mv.Degrain3(sD, sDsuper, bv1, fv1, bv2, fv2, bv3, fv3)
                else:
                    raise ValueError(funcName + ': valid values of \"tradius\" are 1, 2 and 3!')
        else:
            raise ValueError(funcName + ': valid values of \"tradius\" are 1, 2 and 3!')

        sDD = core.std.Expr([sD, sDD], ['x {neutral} - abs y {neutral} - abs < x y ?'.format(neutral=1 << (bits-1))]).std.Merge(sDD, 0.6)

        last = core.std.MakeDiff(orig, sDD)
    else:
        last = merged

    if orig_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, orig_src], planes=list(range(orig_src.format.num_planes)),
            colorfamily=orig_src.format.color_family)


def TEdge(input: vs.VideoNode, min: int = 0, max: int = 65535, planes: PlanesType = None,
          rshift: int = 0
          ) -> vs.VideoNode:
    """Detects edge using TEdgeMask(type=2).

    Port from https://github.com/chikuzen/GenericFilters/blob/2044dc6c25a1b402aae443754d7a46217a2fddbf/src/convolution/tedge.c

    Args:
        input: Source clip.

        min: (int) If output pixel value is lower than this, it will be zero. Default is 0.

        max: (int) If output pixel value is same or higher than this, it will be maximum value of the format. Default is 65535.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

        rshift: (int) Shift the output values to right by this count before clamp. Default is 0.

    """

    funcName = 'TEdge'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    rshift = 1 << rshift

    bits = input.format.bits_per_sample
    floor = 0
    peak = (1 << bits) - 1

    gx = core.std.Convolution(input, [4, -25, 0, 25, -4], planes=planes, saturate=False, mode='h')
    gy = core.std.Convolution(input, [-4, 25, 0, -25, 4], planes=planes, saturate=False, mode='v')

    calcexpr = 'x x * y y * + {rshift} / sqrt'.format(rshift=rshift)
    expr = '{calc} {max} > {peak} {calc} {min} < {floor} {calc} ? ?'.format(calc=calcexpr, max=max, peak=peak, min=min, floor=floor)
    return core.std.Expr([gx, gy], [(expr if i in planes else '') for i in range(input.format.num_planes)])


def Sort(input: vs.VideoNode, order: int = 1, planes: PlanesType = None,
         mode: str = 'max'
         ) -> vs.VideoNode:
    """Simple filter to get nth largeest value in 3x3 neighbourhood.

    Args:
        input: Source clip.

        order: (int) The order of value to get in 3x3 neighbourhood. Default is 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

        mode: ("max" or "min") How to measure order. Default is "max".

    """

    funcName = 'Sort'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if order not in range(1, 10):
        raise ValueError(funcName + ': valid values of \"order\" are 1~9!')

    mode = mode.lower()
    if mode not in ['max', 'min']:
        raise ValueError(funcName + ': valid values of \"mode\" are \"max\" and \"min\"!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if mode == 'min':
        order = 10 - order # the nth smallest value in 3x3 neighbourhood is the same as the (10-n)th largest value

    if order == 1:
        sort = core.std.Maximum(input, planes=planes)
    elif order in range(2, 5):
        sort = core.rgvs.Repair(core.std.Maximum(input, planes=planes), input,
                                [(order if i in planes else 0) for i in range(input.format.num_planes)])
    elif order == 5:
        sort = core.std.Median(input, planes=planes)
    elif order in range(6, 9):
        sort = core.rgvs.Repair(core.std.Minimum(input, planes=planes), input,
                                [((10 - order) if i in planes else 0) for i in range(input.format.num_planes)])
    else: # order == 9
        sort = core.std.Minimum(input, planes=planes)

    return sort


def Soothe_mod(input: vs.VideoNode, source: vs.VideoNode, keep: float = 24, radius: int = 1,
               scenechange: int = 32, use_misc: bool = True
               ) -> vs.VideoNode:
    """Modified Soothe().

    Basd on Didée, 6th September 2005, http://forum.doom9.org/showthread.php?p=708217#post708217
    Modified by TheRyuu, 14th July 2007, http://forum.doom9.org/showthread.php?p=1024318#post1024318
    Modified by Muonium, 12th, December 2016, add args "radius", "scenechange" and "use_misc"

    Requires Filters
    misc (optional)

    Args:
        input: Filtered clip.

        source: Source clip. Must match "input" clip.

        keep: (0~100). Minimum percent of the original sharpening to keep. Default is 24.

        radius: (1~7 (use_misc=True) or 1~12 (use_misc=False)) Temporal radius of AverageFrames. Default is 1.

        scenechange: (int) Argument in scenechange detection. Default is 32.

        use_misc: (bint) Whether to use miscellaneous filters. Default is True.

    Examples: (in Avisynth)
        We use LimitedSharpen() as sharpener, and we'll keep at least 20% of its result:
        dull = last
        sharpener = dull.LimitedSharpen( ss_x=1.25, ss_y=1.25, strength=150, overshoot=1 )

        Soothe( sharp, dull, 20 )

    """

    funcName = 'Soothe_mod'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')
    if not isinstance(source, vs.VideoNode):
        raise TypeError(funcName + ': \"source\" must be a clip!')

    if input is source:
        return input

    if input.format.id != source.format.id:
        raise TypeError(funcName + ': \"source\" must be of the same format as \"input\"!')
    if input.width != source.width or input.height != source.height:
        raise TypeError(funcName + ': \"source\" must be of the same size as \"input\"!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    if source.format.color_family != vs.GRAY:
        source_src = source # type: Optional[vs.VideoNode]
        source = mvf.GetPlane(source, 0)
    else:
        source_src = None

    keep = max(min(keep, 100), 0)

    if use_misc:
        if not isinstance(radius, int) or (not(1 <= radius <= 12)):
            raise ValueError(funcName + ': \'radius\' have not a correct value! [1 ~ 12]')
    else:
        if not isinstance(radius, int) or (not(1 <= radius <= 7)):
            raise ValueError(funcName + ': \'radius\' have not a correct value! [1 ~ 7]')

    bits = source.format.bits_per_sample

    diff = core.std.MakeDiff(source, input)
    if use_misc:
        diff2 = TemporalSoften(diff, radius, scenechange)
    else:
        diff2 = haf_TemporalSoften(diff, radius, (1 << bits)-1, 0, scenechange)

    expr = 'x {neutral} - y {neutral} - * 0 < x {neutral} - {KP} * {neutral} + x {neutral} - abs y {neutral} - abs > x {KP} * y {iKP} * + x ? ?'.format(
        neutral=1 << (bits-1), KP=keep/100, iKP=1-keep/100)
    diff3 = core.std.Expr([diff, diff2], [expr])

    last = core.std.MakeDiff(source, diff3)

    if source_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, source_src], planes=list(range(source_src.format.num_planes)),
            colorfamily=source_src.format.color_family)


def TemporalSoften(input: vs.VideoNode, radius: int = 4, scenechange: int = 15) -> vs.VideoNode:
    """TemporalSoften filter without thresholding using Miscellaneous filters.

    There will be slight difference in result compare to havsfunc.TemporalSoften().
    It seems that this Misc-filter-based TemporalSoften is slower than the one in havsfunc.

    Read the document of Avisynth version for more details.

    """

    funcName = 'TemporalSoften'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if scenechange:
        input = haf_SCDetect(input, scenechange / 255)

    if _is_api4:
        return core.std.AverageFrames(input, [1] * (2 * radius + 1), scenechange=scenechange)
    else:
        return core.misc.AverageFrames(input, [1] * (2 * radius + 1), scenechange=scenechange)


def FixTelecinedFades(input: vs.VideoNode, mode: Union[int, Sequence[int]] = 0,
                      threshold: Sequence[float] = [0.0], color: Sequence[float] = [0.0],
                      full: Optional[bool] = None, planes: PlanesType = None
                      ) -> vs.VideoNode:
    """Fix Telecined Fades filter

    The main algorithm was proposed by feisty2 (http://forum.doom9.org/showthread.php?t=174151).
    The idea of thresholding was proposed by Myrsloik (http://forum.doom9.org/showthread.php?p=1791412#post1791412).
    Corresponding C++ code written by feisty2: https://github.com/IFeelBloated/Fix-Telecined-Fades/blob/7922a339629ed8ce93b540f3bdafb99fe97096b6/Source.cpp.

    the filter gives a mathematically perfect solution to such
    (fades were done AFTER telecine which made a picture perfect IVTC pretty much impossible) problem,
    and it's now time to kiss "vinverse" goodbye cuz "vinverse" is old and low quality.
    unlike vinverse which works as a dumb blurring + contra-sharpening combo and very harmful to artifacts-free frames,
    this filter works by matching the brightness of top and bottom fields with statistical methods, and also harmless to healthy frames.

    Args:
        input: Source clip. Can be 8-16 bits integer or 32 bits floating point based. Recommend to use 32 bits float format.

        mode: (0~2 []) Default is 0.
            0: adjust the brightness of both fields to match the average brightness of 2 fields.
            1: darken the brighter field to match the brightness of the darker field
            2: brighten the darker field to match the brightness of the brighter field

        threshold: (float [], positive) Default is 0.
            If the absolute difference between filtered pixel and input pixel is less than "threshold", then just copy the input pixel.
            The value is always scaled by 8 bits integer.
            The last value in the list will be used for the remaining plane.

        color: (float [], positive) Default is 0.
            (It is difficult for me to describe the effect of this parameter.)
            The value is always scaled by 8 bits integer.
            The last value in the list will be used for the remaining plane.

        full: (bint) If not set, assume False(limited range) for Gray and YUV input, assume True(full range) for other input.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    funcName = 'FixTelecinedFades'

    # set parameters
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    bits = input.format.bits_per_sample
    isFloat = input.format.sample_type == vs.FLOAT

    if isinstance(mode, int):
        mode = [mode]
    elif not isinstance(mode, abc.Sequence) or len(mode) == 0:
        raise TypeError(funcName + ': \"mode\" must be an int or a list of ints!')

    for i in mode:
        if i not in [0, 1, 2]:
            raise ValueError(funcName + ': valid values of \"mode\" are 0, 1 or 2!')

    if isinstance(threshold, (int, float)):
        threshold = [threshold]
    if not isinstance(threshold, abc.Sequence) or len(threshold) == 0:
        raise TypeError(funcName + ': \"threshold\" must be a list of floats!')

    if isFloat:
        _threshold = [(abs(thr) / 255) for thr in threshold]
    else:
        _threshold = [(abs(thr) * ((1 << bits) - 1) / 255) for thr in threshold]

    if isinstance(color, (int, float)):
        color = [color]
    if not isinstance(color, abc.Sequence) or len(color) == 0:
        raise TypeError(funcName + ': \"color\" must be a list of floats!')

    if isFloat:
        _color = [(c / 255) for c in color]
    else:
        _color = [(abs(c) * ((1 << bits) - 1) / 255) for c in color]

    if full is None:
        if input.format.color_family in [vs.GRAY, vs.YUV]:
            full = False
        else:
            full = True

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # internal function
    def GetExpr(scale: float, color: float, threshold: float) -> str:
        if color != 0.:
            flt = 'x {color} - {scale} * {color} +'.format(scale=scale, color=color)
        else:
            flt = 'x {scale} *'.format(scale=scale)
        return flt if threshold == 0. else '{flt} x - abs {threshold} > {flt} x ?'.format(flt=flt, threshold=threshold)


    def Adjust(n: int, f: List[vs.VideoFrame], clip: vs.VideoNode, core: vs.Core, mode: int,
               threshold: float, color: float
               ) -> vs.VideoNode:
        separated = core.std.SeparateFields(clip, tff=True)
        topField = core.std.SelectEvery(separated, 2, [0])
        bottomField = core.std.SelectEvery(separated, 2, [1])

        topAvg = typing.cast(float, f[0].props['PlaneStatsAverage'])
        bottomAvg = typing.cast(float, f[1].props['PlaneStatsAverage'])

        if color != 0.:
            if isFloat:
                topAvg -= color
                bottomAvg -= color
            else:
                topAvg -= color / ((1 << bits) - 1)
                bottomAvg -= color / ((1 << bits) - 1)

        if topAvg != bottomAvg:
            if mode == 0:
                meanAvg = (topAvg + bottomAvg) / 2
                topField = core.std.Expr([topField], [GetExpr(scale=meanAvg / topAvg, threshold=threshold, color=color)])
                bottomField = core.std.Expr([bottomField], [GetExpr(scale=meanAvg / bottomAvg, threshold=threshold, color=color)])
            elif mode == 1:
                minAvg = min(topAvg, bottomAvg)
                if minAvg == topAvg:
                    bottomField = core.std.Expr([bottomField], [GetExpr(scale=minAvg / bottomAvg, threshold=threshold, color=color)])
                else:
                    topField = core.std.Expr([topField], [GetExpr(scale=minAvg / topAvg, threshold=threshold, color=color)])
            elif mode == 2:
                maxAvg = max(topAvg, bottomAvg)
                if maxAvg == topAvg:
                    bottomField = core.std.Expr([bottomField], [GetExpr(scale=maxAvg / bottomAvg, threshold=threshold, color=color)])
                else:
                    topField = core.std.Expr([topField], [GetExpr(scale=maxAvg / topAvg, threshold=threshold, color=color)])

        woven = core.std.Interleave([topField, bottomField])
        woven = core.std.DoubleWeave(woven, tff=True).std.SelectEvery(2, [0])
        return woven

    # process
    input_src = input
    if not full and not isFloat:
        input = core.fmtc.bitdepth(input, fulls=False, fulld=True, planes=planes)

    separated = core.std.SeparateFields(input, tff=True)
    topField = core.std.SelectEvery(separated, 2, [0])
    bottomField = core.std.SelectEvery(separated, 2, [1])

    topFieldPlanes = {}
    bottomFieldPlanes = {}
    adjustedPlanes = {} # type: Dict[int, Optional[vs.VideoNode]]
    for i in range(input.format.num_planes):
        if i in planes:
            inputPlane = mvf.GetPlane(input, i)
            topFieldPlanes[i] = mvf.GetPlane(topField, i).std.PlaneStats()
            bottomFieldPlanes[i] = mvf.GetPlane(bottomField, i).std.PlaneStats()
            adjustedPlanes[i] = core.std.FrameEval(inputPlane, functools.partial(Adjust, clip=inputPlane, core=core,
                mode=mode[min(i, len(mode) - 1)], threshold=_threshold[min(i, len(_threshold) - 1)], color=_color[min(i, len(_color) - 1)]),
                prop_src=[topFieldPlanes[i], bottomFieldPlanes[i]])
        else:
            adjustedPlanes[i] = None

    adjusted = core.std.ShufflePlanes([(adjustedPlanes[i] if i in planes else input_src) for i in range(input.format.num_planes)], # type: ignore
        [(0 if i in planes else i) for i in range(input.format.num_planes)], input.format.color_family)
    if not full and not isFloat:
        adjusted = core.fmtc.bitdepth(adjusted, fulls=True, fulld=False, planes=planes)
        adjusted = core.std.ShufflePlanes([(adjusted if i in planes else input_src) for i in range(input.format.num_planes)],
            list(range(input.format.num_planes)), input.format.color_family)
    return adjusted


def TCannyHelper(input: vs.VideoNode, t_h: float = 8.0, t_l: float = 1.0, plane: int = 0,
                 returnAll: bool = False, **canny_args: Any
                 ) -> Union[vs.VideoNode, Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode]]:
    """A helper function for tcanny.TCanny(mode=0)

    Strong edge detected by "t_h" will be highlighted in white, and weak edge detected by "t_l" will be highlighted in gray.

    Args:
        input: Source clip. Can be 8-16 bits integer or 32 bits floating point based.

        t_h: (float) TCanny's high gradient magnitude threshold for hysteresis. Default is 8.0.

        t_l: (float) TCanny's low gradient magnitude threshold for hysteresis. Default is 1.0.

        plane: (int) Which plane to be processed. Default is 0.

        returnAll: (bint) Whether to return a tuple containing every 4 temporary clips
            (strongEdge, weakEdge, view, tcannyOutput) or just "view" clip.
            Default is False.

        canny_args: (dict) Additional parameters passed to core.tcanny.TCanny (except "mode" and "planes") in the form of keyword arguments.

    """

    funcName = 'TCannyHelper'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, plane)

    if "mode" in canny_args:
        del canny_args["mode"]

    bits = input.format.bits_per_sample
    isFloat = input.format.sample_type == vs.FLOAT

    strongEdge = core.tcanny.TCanny(input, t_h=t_h+1e-4, t_l=t_h, mode=0, **canny_args)
    weakEdge = core.tcanny.TCanny(input, t_h=t_l+1e-4, t_l=t_l, mode=0, **canny_args)

    expr = "x y and {peak} y {neutral} 0 ? ?".format(peak=1.0 if isFloat else (1 << bits) - 1, neutral=0.5 if isFloat else 1 << (bits - 1))
    view = core.std.Expr([strongEdge, weakEdge], [expr])

    if returnAll:
        tcannyOutput = core.tcanny.TCanny(input, t_h=t_h, t_l=t_l, mode=0, **canny_args)
        return (strongEdge, weakEdge, view, tcannyOutput)
    else:
        return view


def MergeChroma(clip1: vs.VideoNode, clip2: vs.VideoNode, weight: float = 1.0) -> vs.VideoNode:
    """Merges the chroma from one videoclip into another. Port from Avisynth's equivalent.

    There is an optional weighting, so a percentage between the two clips can be specified.

    Args:
        clip1: The clip that has the chroma pixels merged into (the base clip).

        clip2: The clip from which the chroma pixel data is taken (the overlay clip).

        weight: (float) Defines how much influence the new clip should have. Range is 0.0–1.0.

    """

    funcName = 'MergeChroma'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')

    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if weight >= 1.0:
        return core.std.ShufflePlanes([clip1, clip2], [0, 1, 2], vs.YUV)
    elif weight <= 0.0:
        return clip1
    else:
        if clip1.format.num_planes != 3:
            raise TypeError(funcName + ': \"clip1\" must have 3 planes!')
        if clip2.format.num_planes != 3:
            raise TypeError(funcName + ': \"clip2\" must have 3 planes!')

        clip1_u = mvf.GetPlane(clip1, 1)
        clip2_u = mvf.GetPlane(clip2, 1)
        output_u = core.std.Merge(clip1_u, clip2_u, weight)

        clip1_v = mvf.GetPlane(clip1, 2)
        clip2_v = mvf.GetPlane(clip2, 2)
        output_v = core.std.Merge(clip1_v, clip2_v, weight)

        output = core.std.ShufflePlanes([clip1, output_u, output_v], [0, 0, 0], vs.YUV)

        return output


def firniture(clip: vs.VideoNode, width: int, height: int, kernel: str = 'binomial7',
              taps: Optional[int] = None, gamma: bool = False, fulls: bool = False,
              fulld: bool = False, curve: str = '709', sigmoid: bool = False,
              **resample_args: Any
              ) -> vs.VideoNode:
    '''5 new interpolation kernels (via fmtconv)

    Proposed by *.mp4 guy (https://forum.doom9.org/showthread.php?t=166080)

    Args:
        clip: Source clip.

        width, height: (int) New picture width and height in pixels.

        kernel: (string) Default is "binomial7".
            "binomial5", "binomial7": A binomial windowed sinc filter with 5 or 7 taps.
                Should have the least ringing of any available interpolator, except perhaps "noaliasnoring4".
            "maxflat5", "maxflat8": 5 or 8 tap interpolation that is maximally flat in the passband.
                In English, these filters have a sharp and relatively neutral look, but can have ringing and aliasing problems.
            "noalias4": A 4 tap filter hand designed to be free of aliasing while having acceptable ringing and blurring characteristics.
                Not always a good choice, but sometimes very useful.
            "noaliasnoring4": Derived from the "noalias4" kernel, but modified to have reduced ringing. Other attributes are slightly worse.

        taps: (int) Default is the last num in "kernel".
            "taps" in fmtc.resample. This parameter is now mostly superfluous.
            It has been retained so that you can truncate the kernels to shorter taps then they would normally use.

        gamma: (bool) Default is False.
            Set to true to turn on gamma correction for the y channel.

        fulls: (bool) Default is False.
            Specifies if the luma is limited range (False) or full range (True)

        fulld: (bool) Default is False.
            Same as fulls, but for output.

        curve: (string) Default is '709'.
            Type of gamma mapping.

        sigmoid: (bool) Default is False.
            When True, applies a sigmoidal curve after the power-like curve (or before when converting from linear to gamma-corrected).
            This helps reducing the dark halo artefacts around sharp edges caused by resizing in linear luminance.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of keyword arguments.

    Examples:
        clip = muvsfunc.firniture(clip, 720, 400, kernel="noalias4", gamma=False)

    '''

    funcName = 'firniture'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    impulseCoefficents = dict(
        binomial5=[8, 0, -589, 0, 11203, 0, -93355, 0, 606836, 1048576, 606836, 0, -93355, 0, 11203, 0, -589, 0, 8],
        binomial7=[146, 0, -20294, 0, 744006, 0, -11528384, 0, 94148472, 0, -487836876, 0, 2551884458, 4294967296, 2551884458,
            0, -487836876, 0, 94148472, 0, -11528384, 0, 744006, 0, -20294, 0, 146],
        maxflat5=[-259, 1524, -487, -12192, 17356, 42672, -105427, -85344, 559764, 1048576, 559764, -85344, -105427, 42672,
            17356, -12192, -487, 1524, -259],
        maxflat8=[2, -26, 166, -573, 912, 412, 1524, -589, -12192, 17356, 42672, -105427, -85344, 606836, 1048576, 606836, -85344,
            -105427, 42672, 17356, -12192, -589, 1524, 412, 912, -573, 166, -26, 2],
        noalias4=[-1, 2, 4, -6, -17, 7, 59, 96, 59, 7, -17, -6, 4, 2, -1],
        noaliasnoring4=[-1, 8, 40, -114, -512, 360, 3245, 5664, 3245, 360, -512, -114, 40, 8, -1]
        )

    if taps is None:
        taps = int(kernel[-1])

    if clip.format.bits_per_sample != 16:
        clip = mvf.Depth(clip, 16)

    if gamma:
        import nnedi3_resample as nnrs
        clip = nnrs.GammaToLinear(clip, fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    clip = core.fmtc.resample(clip, width, height, kernel='impulse', impulse=impulseCoefficents[kernel], kovrspl=2,
        taps=taps, **resample_args)

    if gamma:
        clip = nnrs.LinearToGamma(clip, fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    return clip


def BoxFilter(input: vs.VideoNode, radius: int = 16, radius_v: Optional[int] = None, planes: PlanesType = None,
              fmtc_conv: int = 0, radius_thr: Optional[int] = None,
              resample_args: Optional[Dict[str, Any]] = None, keep_bits: bool = True,
              depth_args: Optional[Dict[str, Any]] = None
              ) -> vs.VideoNode:
    '''Box filter

    Performs a box filtering on the input clip.
    Box filtering consists in averaging all the pixels in a square area whose center is the output pixel.
    You can approximate a large gaussian filtering by cascading a few box filters.

    Args:
        input: Input clip to be filtered.

        radius, radius_v: (int) Size of the averaged square. The size is (radius*2-1) * (radius*2-1).
            If "radius_v" is None, it will be set to "radius".
            Default is 16.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fmtc_conv: (0~2) Whether to use fmtc.resample for convolution.
            It's recommended to input clip without chroma subsampling when using fmtc.resample, otherwise the output may be incorrect.
            0: False. 1: True (except both "radius" and "radius_v" is strictly smaller than 4).
                2: Auto, determined by radius_thr (exclusive).
            Default is 0.

        radius_thr: (int) Threshold of wheter to use fmtc.resample when "fmtc_conv" is 2.
            Default is 11 for integer input and 21 for float input.
            Only works when "fmtc_conv" is enabled.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict.
            It's recommended to set "flt" to True for higher precision, like:
                flt = muf.BoxFilter(src, resample_args=dict(flt=True))
            Only works when "fmtc_conv" is enabled.
            Default is {}.

        keep_bits: (bool) Whether to keep the bitdepth of the output the same as input.
            Only works when "fmtc_conv" is enabled and input is integer.

        depth_args: (dict) Additional parameters passed to mvf.Depth in the form of dict.
            Only works when "fmtc_conv" is enabled, input is integer and "keep_bits" is True.
            Default is {}.

    '''

    funcName = 'BoxFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if radius_v is None:
        radius_v = radius

    if radius == radius_v == 1:
        return input

    if radius_thr is None:
        radius_thr = 21 if input.format.sample_type == vs.FLOAT else 11 # Values are measured from my experiment

    if resample_args is None:
        resample_args = {}

    if depth_args is None:
        depth_args = {}

    planes2 = [(3 if i in planes else 2) for i in range(input.format.num_planes)]
    width = radius * 2 - 1
    width_v = radius_v * 2 - 1
    kernel = [1 / width] * width
    kernel_v = [1 / width_v] * width_v

    # process
    if input.format.sample_type == vs.FLOAT:
        if core.version_number() < 33:
            raise NotImplementedError(funcName + (': Please update your VapourSynth.'
                'BoxBlur on float sample has not yet been implemented on current version.'))
        elif radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                return flt # No bitdepth conversion is required since fmtc.resample outputs the same bitdepth as input

            elif core.version_number() >= 39:
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur on float sample has not been implemented
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input

    else: # input.format.sample_type == vs.INTEGER
        if radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                if keep_bits and input.format.bits_per_sample != flt.format.bits_per_sample:
                    flt = mvf.Depth(flt, depth=input.format.bits_per_sample, **depth_args)
                return flt

            elif hasattr(core.std, 'BoxBlur'):
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur was not found
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input


def SmoothGrad(input: vs.VideoNode, radius: int = 9, thr: float = 0.25,
               ref: Optional[vs.VideoNode] = None, elast: float = 3.0,
               planes: PlanesType = None, **limit_filter_args: Any) -> vs.VideoNode:
    '''Avisynth's SmoothGrad

    SmoothGrad smooths the low gradients or flat areas of a 16-bit clip.
    It proceeds by applying a huge blur filter and comparing the result with the input data for each pixel.
    If the difference is below the specified threshold, the filtered version is taken into account,
        otherwise the input pixel remains unchanged.

    Args:
        input: Input clip to be filtered.

        radius: (int) Size of the averaged square. Its width is radius*2-1. Range is 2-9.

        thr: (float) Threshold between reference data and filtered data, on an 8-bit scale.

        ref: Reference clip for the filter output comparison. Specify here the input clip when you cascade several SmoothGrad calls.
            When undefined, the input clip is taken as reference.

        elast: (float) To avoid artifacts, the threshold has some kind of elasticity.
            Value differences falling over this threshold are gradually attenuated, up to thr * elast > 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        limit_filter_args: (dict) Additional arguments passed to mvf.LimitFilter in the form of keyword arguments.

    '''

    funcName = 'SmoothGrad'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # process
    smooth = BoxFilter(input, radius, planes=planes)

    return mvf.LimitFilter(smooth, input, ref, thr, elast, planes=planes, **limit_filter_args)


def DeFilter(clip: vs.VideoNode, func: VSFuncType, iteration: int = 10, planes: PlanesType = None,
             step_size: float = 1., **func_args: Any) -> vs.VideoNode:
    '''Zero-order reverse filter

    Args:
        clip: Input clip to be reversed.

        func: The function of how the input clip is filtered.

        iteration: (int) Number of iterations. Default is 10.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

        step_size: (float, positive) Step size of updating.
            A lower value helps to prevent divergence, as analyzed in [3].
            The optimal value depends on the function.
            Default is 1.

        func_args: (dict) Additional arguments passed to "func" in the form of keyword arguments. Alternative to functools.partial.

    Ref:
        [1] Tao, X., Zhou, C., Shen, X., Wang, J., & Jia, J. (2017, October). Zero-Order Reverse Filtering.
            In Computer Vision (ICCV), 2017 IEEE International Conference on (pp. 222-230). IEEE.
        [2] https://github.com/jiangsutx/DeFilter
        [3] Milanfar, P. (2018). Rendition: Reclaiming what a black box takes away. SIAM Journal on Imaging Sciences, 11(4), 2722-2756.

    '''

    funcName = 'DeFilter'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # initialization
    flt = clip
    calc_expr = "x y z - +" if step_size == 1. else f"x y z - {step_size} * +"
    expr = [(calc_expr if i in planes else '') for i in range(clip.format.num_planes)]

    # iteration
    for _ in range(iteration):
        flt = core.std.Expr([flt, clip, func(flt, **func_args)], expr)

    # equivalence
    # flt = functools.reduce(lambda flt, src: core.std.Expr([flt, src, func(flt, **func_args)], expr), [clip] * iteration)

    return flt


def scale(val: float, bits: int) -> float:
    '''The old scale function in havsfunc.

    '''

    return val * ((1 << bits) - 1) // 255


def ColorBarsHD(clip: Optional[vs.VideoNode] = None, width: int = 1288, height: int = 720) -> vs.VideoNode:
    '''Avisynth's ColorBarsHD()

    It produces a video clip containing SMPTE color bars (Rec. ITU-R BT.709 / arib std b28 v1.0) scaled to any image size.
    By default, a 1288x720, YV24, TV range, 29.97 fps, 1 frame clip is produced.

    Requirment:
        lexpr (https://github.com/AkarinVS/vapoursynth-plugin), or
        mt_lutspa by tp7 (https://gist.githubusercontent.com/tp7/1e39044e1b660ef0a02c)

    Args:
        clip: 'clip' in std.Blankclip(). The output clip will copy its property.
        width, height: (int) The size of the returned clip.
            Nearest 16:9 pixel exact sizes
            56*X x 12*Y
             728 x  480  ntsc anamorphic
             728 x  576  pal anamorphic
             840 x  480
            1008 x  576
            1288 x  720 <- default
            1456 x 1080  hd anamorphic
            1904 x 1080

    '''

    funcName = 'ColorBarsHD'

    if clip is not None and not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    c = round(width * 3 / 28)
    d = round((width - c * 7) / 2)

    p4 = round(height / 4)
    p23 = round(height / 12)
    p1 = height - p23 * 2 - p4

    blkclip_args = dict(format=vs.YUV444P8, length=1, fpsnum=30000, fpsden=1001)

    pattern1_colors = dict(Gray40=[104, 128, 128], White75=[180, 128, 128], Yellow=[168, 44, 136], Cyan=[145, 147, 44], Green=[134, 63, 52],
        Magenta=[63, 193, 204], Red=[51, 109, 212], Blue=[28, 212, 120])
    Gray40 = core.std.BlankClip(clip, d, p1, color=pattern1_colors['Gray40'], **blkclip_args)
    White75 = core.std.BlankClip(clip, c, p1, color=pattern1_colors['White75'], **blkclip_args)
    Yellow = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Yellow'], **blkclip_args)
    Cyan = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Cyan'], **blkclip_args)
    Green = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Green'], **blkclip_args)
    Magenta = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Magenta'], **blkclip_args)
    Red = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Red'], **blkclip_args)
    Blue = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Blue'], **blkclip_args)
    pattern1 = core.std.StackHorizontal([Gray40, White75, Yellow, Cyan, Green, Magenta, Red, Blue, Gray40])

    pattern2_colors = dict(Cyan100=[188, 154, 16], plusI=[16, 98, 161], White75=[180, 128, 128], Blue100=[32, 240, 118])
    Cyan100 = core.std.BlankClip(clip, d, p23, color=pattern2_colors['Cyan100'], **blkclip_args)
    plusI = core.std.BlankClip(clip, c, p23, color=pattern2_colors['plusI'], **blkclip_args)
    White75 = core.std.BlankClip(clip, c*6, p23, color=pattern2_colors['White75'], **blkclip_args)
    Blue100 = core.std.BlankClip(clip, d, p23, color=pattern2_colors['Blue100'], **blkclip_args)
    pattern2 = core.std.StackHorizontal([Cyan100, plusI, White75, Blue100])

    pattern3_colors = dict(Yellow100=[219, 16, 138], Red100=[63, 102, 240])
    Yellow100 = core.std.BlankClip(clip, d, p23, color=pattern3_colors['Yellow100'], **blkclip_args)
    Y_Ramp_tmp = core.std.BlankClip(clip, c*7, 1, color=[0, 128, 128], **blkclip_args)
    if _has_lexpr_lutspa:
        Y_Ramp = core.akarin.Expr(Y_Ramp_tmp, ['220 X * {c} 7 * / 16 +'.format(c=c), ''])
    else:
        from mt_lutspa import lutspa
        Y_Ramp = lutspa(Y_Ramp_tmp, mode='absolute', y_expr='220 x * {c} 7 * / 16 +'.format(c=c), chroma='copy')
    Y_Ramp = core.resize.Point(Y_Ramp, c*7, p23)
    Red100 = core.std.BlankClip(clip, d, p23, color=pattern3_colors['Red100'], **blkclip_args)
    pattern3 = core.std.StackHorizontal([Yellow100, Y_Ramp, Red100])

    pattern4_colors = dict(Gray15=[49, 128, 128], Black0=[16, 128, 128], White100=[235, 128, 128], Black_neg2=[12, 128, 128],
        Black_pos2=[20, 128, 128], Black_pos4=[25, 128, 128])
    Gray15 = core.std.BlankClip(clip, d, p4, color=pattern4_colors['Gray15'], **blkclip_args)
    Black0_1 = core.std.BlankClip(clip, round(c*3/2), p4, color=pattern4_colors['Black0'], **blkclip_args)
    White100 = core.std.BlankClip(clip, c*2, p4, color=pattern4_colors['White100'], **blkclip_args)
    Black0_2 = core.std.BlankClip(clip, round(c*5/6), p4, color=pattern4_colors['Black0'], **blkclip_args)
    Black_neg2 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black_neg2'], **blkclip_args)
    Black0_3 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black0'], **blkclip_args)
    Black_pos2 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black_pos2'], **blkclip_args)
    Black0_4 = Black0_3
    Black_pos4 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black_pos4'], **blkclip_args)
    Black0_5 = core.std.BlankClip(clip, c, p4, color=pattern4_colors['Black0'], **blkclip_args)
    pattern4 = core.std.StackHorizontal([Gray15, Black0_1, White100, Black0_2, Black_neg2, Black0_3, Black_pos2, Black0_4,
        Black_pos4, Black0_5, Gray15])

    #pattern = core.std.StackVertical([pattern1, pattern2, pattern3, pattern4])
    #return pattern1, pattern2, pattern3, pattern4
    pattern = core.std.StackVertical([pattern1, pattern2, pattern3, pattern4])
    return pattern


def SeeSaw(clp: vs.VideoNode, denoised: Optional[vs.VideoNode] = None, NRlimit: int = 2,
           NRlimit2: Optional[int] = None, Sstr: float = 1.5, Slimit: Optional[int] = None,
           Spower: float = 4, SdampLo: Optional[float] = None, SdampHi: float = 24, Szp: float = 18,
           bias: float = 49, Smode: Optional[int] = None, sootheT: int = 49, sootheS: int = 0,
           ssx: float = 1.0, ssy: Optional[float] = None, diff: bool = False) -> vs.VideoNode:
    """Avisynth's SeeSaw v0.3e

    Author: Didée (http://avisynth.nl/images/SeeSaw.avs)

    (Full Name: "Denoiser-and-Sharpener-are-riding-the-SeeSaw" )

    This function provides a (simple) implementation of the "crystality sharpen" principle.
    In conjunction with a user-specified denoised clip, the aim is to enhance
    weak detail, hopefully without oversharpening or creating jaggies on strong
    detail, and produce a result that is temporally stable without detail shimmering,
    while keeping everything within reasonable bitrate requirements.
    This is done by intermixing source, denoised source and a modified sharpening process,
    in a seesaw-like manner.

    This version is considered alpha.

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip; the noisy source.

        deonised: Input clip; denoised clip.
            You're very much encouraged to feed your own custom denoised clip into SeeSaw.
            If the "denoised" clip parameter is omitted, a simple "spatial pressdown" filter is used.

        NRlimit: (int) Absolute limit for pixel change by denoising. Default is 2.

        NRlimit2: (int) Limit for intermediate denoising. Default is NRlimit+1.

        Sstr: (float) Sharpening strength (don't touch this too much). Default is 1.5.

        Slimit: (int) Positive: absolute limit for pixel change by sharpening.
            Negative: pixel's sharpening difference is reduced to diff = pow(diff,1/abs(limit)).
            Default is NRlimit+2.

        Spower: (float) Exponent for modified sharpener. Default is 4.

        Szp: (float) Zero point - below: overdrive sharpening - above: reduced sharpening. Default is 16+2.

        SdampLo: (float) Reduces overdrive sharpening for very small changes. Default is Spower+1.

        SdampHi: (float) Further reduces sharpening for big sharpening changes. Try 15~30. "0" disables. Default is 24.

        bias: (float) Bias towards detail ( >= 50 ), or towards calm result ( < 50 ). Default is 49.

        Smode: (int) RemoveGrain mode used in the modified sharpening function (sharpen2).
            Default: ssx<1.35 ? 11 : ssx<1.51 ? 20 : 19

        sootheT: (int) 0=minimum, 100=maximum soothing of sharpener's temporal instability.
            (-100 .. -1 : will chain 2 instances of temporal soothing.)
            Default is 49.

        sootheS: (int) 0=minimum, 100=maximum smoothing of sharpener's spatial effect. Default is 0.

        ssx, ssy: (int) SeeSaw doesn't require supersampling urgently, if at all, small values ~1.25 seem to be enough. Default is 1.0.

        diff: (bool) When True, limit the sharp-difference instead of the sharpened clip.
                     Relative limiting is more safe, less aliasing, but also less sharpening.

    Usage: (in Avisynth)
        a = TheNoisySource
        b = a.YourPreferredDenoising()
        SeeSaw( a, b, [parameters] )

    """

    funcName = 'SeeSaw'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample

    if NRlimit2 is None:
        NRlimit2 = NRlimit + 1

    if Slimit is None:
        Slimit = NRlimit + 2

    if SdampLo is None:
        SdampLo = Spower + 1

    if Smode is None:
        if ssx < 1.35:
            Smode = 11
        elif ssx < 1.51:
            Smode = 20
        else:
            Smode = 19

    if ssy is None:
        ssy = ssx

    Szp = Szp / pow(Sstr, 0.25) / pow((ssx + ssy) / 2, 0.5)
    SdampLo = SdampLo / pow(Sstr, 0.25) / pow((ssx + ssy) / 2, 0.5)

    ox = clp.width
    oy = clp.height
    xss = haf_m4(ox * ssx)
    yss = haf_m4(oy * ssy)
    NRL = scale(NRlimit, bits)
    NRL2 = scale(NRlimit2, bits)
    NRLL = scale(round(NRlimit2 * 100 / bias - 1), bits)
    SLIM = scale(Slimit, bits) if Slimit >= 0 else abs(Slimit)
    multiple = 1 << (bits - 8)
    neutral = 1 << (bits - 1)

    if denoised is None:
        dnexpr = 'x {NRL} + y < x {NRL} + x {NRL} - y > x {NRL} - y ? ?'.format(NRL=NRL)
        denoised = core.std.Expr([clp, core.std.Median(clp, [0])], [dnexpr] if isGray else [dnexpr, ''])
    else:
        if not isinstance(denoised, vs.VideoNode):
            raise TypeError(funcName + ': \"denoised\" must be a clip!')
        if denoised.format.id != clp.format.id:
            raise TypeError(funcName + ': \"denoised\" the same format as \"clp\"!')
        if denoised.width != clp.width or denoised.height != clp.height:
            raise TypeError(funcName + ': \"denoised\" must be of the same size as \"clp\"!')

    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)
        denoised_src = denoised
        denoised = mvf.GetPlane(denoised) if clp_src != denoised_src else clp

    NRdiff = core.std.MakeDiff(clp, denoised)

    tameexpr = 'x {NRLL} + y < x {NRL2} + x {NRLL} - y > x {NRL2} - x {BIAS1} * y {BIAS2} * + 100 / ? ?'.format(NRLL=NRLL,
        NRL2=NRL2, BIAS1=bias, BIAS2=100-bias)
    tame = core.std.Expr([clp, denoised], [tameexpr])

    head = _SeeSaw_sharpen2(tame, Sstr, Spower, Szp, SdampLo, SdampHi, 4, diff)

    if ssx == 1. and ssy == 1.:
        last = core.rgvs.Repair(_SeeSaw_sharpen2(tame, Sstr, Spower, Szp, SdampLo, SdampHi, Smode, diff), head, [1])
    else:
        last = core.rgvs.Repair(_SeeSaw_sharpen2(tame.resize.Lanczos(xss, yss), Sstr, Spower, Szp, SdampLo, SdampHi, Smode, diff),
            head.resize.Bicubic(xss, yss, filter_param_a=-0.2, filter_param_b=0.6), [1]).resize.Lanczos(ox, oy)

    if diff:
        last = core.std.MergeDiff(tame, last)

    last = _SeeSaw_SootheSS(last, tame, sootheT, sootheS)
    sharpdiff = core.std.MakeDiff(tame, last)

    if NRlimit == 0 or clp == denoised:
        last = clp
    else:
        NRdiff = core.std.MakeDiff(clp, denoised)
        last = core.std.Expr([clp, NRdiff], ['y {neutral} {NRL} + > x {NRL} - y {neutral} {NRL} - < x {NRL} + x y {neutral} - - ? ?'.format(
            neutral=neutral, NRL=NRL)])

    if Slimit >= 0:
        limitexpr = 'y {neutral} {SLIM} + > x {SLIM} - y {neutral} {SLIM} - < x {SLIM} + x y {neutral} - - ? ?'.format(
            neutral=neutral, SLIM=SLIM)
        last = core.std.Expr([last, sharpdiff], [limitexpr])
    else:
        limitexpr = 'y {neutral} = x x y {neutral} - abs {multiple} / 1 {SLIM} / pow {multiple} * y {neutral} - y {neutral} - abs / * - ?'.format(
            neutral=neutral, SLIM=SLIM, multiple=multiple)
        last = core.std.Expr([last, sharpdiff], [limitexpr])

    return last if isGray else core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)


def _SeeSaw_sharpen2(clp: vs.VideoNode, strength: float, power: float, zp: float, lodmp: float,
                     hidmp: float, rg: int, diff: bool
                     ) -> vs.VideoNode:
    """Modified sharpening function from SeeSaw()

    Only the first plane (luma) will be processed.

    """

    funcName = '_SeeSaw_sharpen2'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample
    multiple = 1 << (bits - 8)
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    power = max(power, 1)
    power = 1 / power

    # copied from havsfunc
    def get_lut1(x: int) -> int:
        if x == neutral:
            return x
        else:
            tmp1 = abs(x - neutral) / multiple
            tmp2 = tmp1 ** 2
            tmp3 = zp ** 2
            return min(max(math.floor(neutral + (tmp1 / zp) ** power * zp * (strength * multiple) * (1 if x > neutral else -1) *
                (tmp2 * (tmp3 + lodmp) / ((tmp2 + lodmp) * tmp3)) * ((1 + (0 if hidmp == 0. else (zp / hidmp) ** 4)) /
                    (1 + (0 if hidmp == 0. else (tmp1 / hidmp) ** 4))) + 0.5), 0), peak)

    if rg == 4:
        method = clp.std.Median(planes=[0])
    elif rg in [11, 12]:
        method = clp.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0])
    elif rg == 19:
        method = clp.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0])
    elif rg == 20:
        method = clp.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=[0])
    else:
        method = clp.rgvs.RemoveGrain([rg] if isGray else [rg, 0])

    sharpdiff = core.std.MakeDiff(clp, method, [0]).std.Lut(function=get_lut1, planes=[0])

    return sharpdiff if diff else core.std.MergeDiff(clp, sharpdiff, [0])


def _SeeSaw_SootheSS(sharp: vs.VideoNode, orig: vs.VideoNode, sootheT: float = 25, sootheS: float = 0) -> vs.VideoNode:
    """Soothe() function to stabilze sharpening from SeeSaw()

    Only the first plane (luma) will be processed.

    """

    funcName = '_SeeSaw_SootheSS'

    if not isinstance(sharp, vs.VideoNode) or sharp.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"sharp\" must be a Gray or YUV clip!')

    if not isinstance(orig, vs.VideoNode):
        raise TypeError(funcName + ': \"orig\" must be a clip!')
    if orig.format.id != sharp.format.id:
        raise TypeError(funcName + ': \"orig\" the same format as \"sharp\"!')
    if orig.width != sharp.width or orig.height != sharp.height:
        raise TypeError(funcName + ': \"orig\" must be of the same size as \"sharp\"!')

    sootheT = max(min(sootheT, 100), -100)
    sootheS = max(min(sootheS, 100), 0)
    ST = 100 - abs(sootheT)
    SSPT = 100 - abs(sootheS)
    last = core.std.MakeDiff(orig, sharp, [0])

    neutral = 1 << (sharp.format.bits_per_sample - 1)
    isGray = sharp.format.color_family == vs.GRAY

    if not isGray:
        sharp_src = sharp
        sharp = mvf.GetPlane(sharp)
        orig_src = orig
        orig = mvf.GetPlane(orig) if sharp_src != orig_src else sharp

    expr1 = ('x {neutral} < y {neutral} < xor x {neutral} - 100 / {SSPT} * {neutral} + x {neutral} - '
        'abs y {neutral} - abs > x {SSPT} * y {i} * + 100 / x ? ?'.format(neutral=neutral, SSPT=SSPT, i=100-SSPT))
    expr2 = ('x {neutral} < y {neutral} < xor x {neutral} - 100 / {ST} * {neutral} + x {neutral} - '
        'abs y {neutral} - abs > x {ST} * y {i} * + 100 / x ? ?'.format(neutral=neutral, ST=ST, i=100-ST))

    if sootheS != 0.:
        last = core.std.Expr([last, core.std.Convolution(last, [1]*9)], [expr1])
    if sootheT != 0.:
        last = core.std.Expr([last, TemporalSoften(last, 1, 0)], [expr2])
    if sootheT <= -1:
        last = core.std.Expr([last, TemporalSoften(last, 1, 0)], [expr2])

    last = core.std.MakeDiff(orig, last, [0])
    return last if isGray else core.std.ShufflePlanes([last, orig_src], list(range(orig_src.format.num_planes)), orig_src.format.color_family)


def abcxyz(clp: vs.VideoNode, rad: float = 3.0, ss: float = 1.5) -> vs.VideoNode:
    """Avisynth's abcxyz()

    Reduces halo artifacts that can occur when sharpening.

    Author: Didée (http://avisynth.nl/images/Abcxyz_MT2.avsi)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rad: (float) Radius for halo removal. Default is 3.0.

        ss: (float) Radius for supersampling / ss=1.0 -> no supersampling. Range: 1.0 - ???. Default is 1.5

    """

    funcName = 'abcxyz'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    ox = clp.width
    oy = clp.height

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample

    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)

    x = core.resize.Bicubic(clp, haf_m4(ox/rad), haf_m4(oy/rad), filter_param_a=1/3, filter_param_b=1/3).resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
    y = core.std.Expr([clp, x], ['x {a} + y < x {a} + x {b} - y > x {b} - y ? ? x y - abs * x {c} x y - abs - * + {c} /'.format(
        a=scale(8, bits), b=scale(24, bits), c=scale(32, bits))])

    z1 = core.rgvs.Repair(clp, y, [1])

    if ss != 1.:
        maxbig = core.std.Maximum(y).resize.Bicubic(haf_m4(ox*ss), haf_m4(oy*ss), filter_param_a=1/3, filter_param_b=1/3)
        minbig = core.std.Minimum(y).resize.Bicubic(haf_m4(ox*ss), haf_m4(oy*ss), filter_param_a=1/3, filter_param_b=1/3)
        z2 = core.resize.Lanczos(clp, haf_m4(ox*ss), haf_m4(oy*ss))
        z2 = core.std.Expr([z2, maxbig, minbig], ['x y min z max']).resize.Lanczos(ox, oy)
        z1 = z2  # for simplicity

    if not isGray:
        z1 = core.std.ShufflePlanes([z1, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return z1


def Sharpen(clip: vs.VideoNode, amountH: float = 1.0, amountV: Optional[float] = None,
            planes: PlanesType = None
            ) -> vs.VideoNode:
    """Avisynth's internel filter Sharpen()

    Simple 3x3-kernel sharpening filter.

    Args:
        clip: Input clip.

        amountH, amountV: (float) Sharpen uses the kernel is [(1-2^amount)/2, 2^amount, (1-2^amount)/2].
            A value of 1.0 gets you a (-1/2, 2, -1/2) for example.
            Negative Sharpen actually blurs the image.
            The allowable range for Sharpen is from -1.58 to +1.0.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

    """

    funcName = 'Sharpen'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1.5849625 or amountH > 1:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1.58 ~ 1]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1.5849625 or amountV > 1:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1.58 ~ 1]')

    if planes is None:
        planes = list(range(clip.format.num_planes))

    center_weight_v = math.floor(2 ** (amountV - 1) * 1023 + 0.5)
    outer_weight_v = math.floor((0.25 - 2 ** (amountV - 2)) * 1023 + 0.5)
    center_weight_h = math.floor(2 ** (amountH - 1) * 1023 + 0.5)
    outer_weight_h = math.floor((0.25 - 2 ** (amountH - 2)) * 1023 + 0.5)

    conv_mat_v = [outer_weight_v, center_weight_v, outer_weight_v]
    conv_mat_h = [outer_weight_h, center_weight_h, outer_weight_h]

    if math.fabs(amountH) >= 0.00002201361136: # log2(1+1/65536)
        clip = core.std.Convolution(clip, conv_mat_v, planes=planes, mode='v')

    if math.fabs(amountV) >= 0.00002201361136:
        clip = core.std.Convolution(clip, conv_mat_h, planes=planes, mode='h')

    return clip


def Blur(clip: vs.VideoNode, amountH: float = 1.0, amountV: Optional[float] = None,
         planes: PlanesType = None
         ) -> vs.VideoNode:
    """Avisynth's internel filter Blur()

    Simple 3x3-kernel blurring filter.

    In fact Blur(n) is just an alias for Sharpen(-n).

    Args:
        clip: Input clip.

        amountH, amountV: (float) Blur uses the kernel is [(1-1/2^amount)/2, 1/2^amount, (1-1/2^amount)/2].
            A value of 1.0 gets you a (1/4, 1/2, 1/4) for example.
            Negative Blur actually sharpens the image.
            The allowable range for Blur is from -1.0 to +1.58.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

    """

    funcName = 'Blur'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1 or amountH > 1.5849625:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1 ~ 1.58]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1 or amountV > 1.5849625:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1 ~ 1.58]')

    return Sharpen(clip, -amountH, -amountV, planes)


def BlindDeHalo3(clp: vs.VideoNode, rx: float = 3.0, ry: float = 3.0, strength: float = 125,
                 lodamp: float = 0, hidamp: float = 0, sharpness: float = 0, tweaker: float = 0,
                 PPmode: int = 0, PPlimit: Optional[int] = None, interlaced: bool = False
                 ) -> vs.VideoNode:
    """Avisynth's BlindDeHalo3() version: 3_MT2

    This script removes the light & dark halos from too strong "Edge Enhancement".

    Author: Didée (https://forum.doom9.org/attachment.php?attachmentid=5599&d=1143030001)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rx, ry: (float) The radii to use for the [quasi-] Gaussian blur, on which the halo removal is based. Default is 3.0.

        strength: (float) The overall strength of the halo removal effect. Default is 125.

        lodamp, hidamp: (float) With these two values, one can reduce the basic effect on areas that would change only little anyway (lodamp),
            and/or on areas that would change very much (hidamp).
            lodamp does a reasonable job in keeping more detail in affected areas.
            hidamp is intended to keep rather small areas that are very bright or very dark from getting processed too strong.
            Works OK on sources that contain only weak haloing - for sources with strong over sharpening,
                it should not be used, mostly. (Usage has zero impact on speed.)
            Range: 0.0 to ??? (try 4.0 as a start)
            Default is 0.0.

        sharpness: (float) By setting this bigger than 0.0, the affected areas will come out with better sharpness.
            However, strength must be chosen somewhat bigger as well, then, to get the same effect than without.
            (This is the same as initial version's "maskblur" option.)
            Range: 0.0 to 1.58.
            Default is 0.

        tweaker: (float) May be used to get a stronger effect, separately from altering "strength".
            (Also in accordance to initial version's working methodology. I had no better idea for naming this parameter.)
            Range: 0.0 - 1.00.
            Default is 0.

        PPmode: (int) When set to "1" or "2", a second cleaning operation after the basic halo removal is done.
            This deals with:
                a) Removing/reducing those corona lines that sometimes are left over by BlindDeHalo
                b) Improving on mosquito noise, if some is present.
            PPmode=1 uses a simple Gaussian blur for post-cleaning. PPmode=2 uses a 3*3 average, with zero weighting of the center pixel.
            Also, PPmode can be "-1" or "-2". In this case, the main dehaloing step is completely discarded, and *only* the PP cleaning is done.
            This has less effect on halos, but can deal for sources containing more mosquito noise than halos.
            Default is 0.

        PPlimit: (int) Can be used to make the PP routine change no pixel by more than [PPlimit].
            I'm not sure if this makes much sense in this context. However the option is there - you never know what it might be good for.
            Default is 0.

        interlaced: (bool) As formerly, this is intended for sources that were originally interlaced, but then made progressive by deinterlacing.
            It aims in particular at clips that made their way through Restore24.
            Default is False.

    """

    funcName = 'BlindDeHalo3'

    if not isinstance(clp, vs.VideoNode):
        raise TypeError(funcName + ': \"clp\" is not a clip!')

    if clp.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': Only integer clip is supported!')

    if PPlimit is None:
        PPlimit = 4 if abs(PPmode) == 3 else 0

    bits = clp.format.bits_per_sample
    isGray = clp.format.color_family == vs.GRAY
    neutral = 1 << (bits - 1)

    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)

    sharpness = min(sharpness, 1.58)
    tweaker = min(tweaker, 1.0)
    strength *= 1 + sharpness * 0.25
    RR = (rx + ry) / 2
    ST = strength / 100
    LD = scale(lodamp, bits)
    HD = hidamp ** 2
    TWK0 = 'x y - {i} /'.format(i=12 / ST / RR)
    TWK = 'x y - {i} / abs'.format(i=12 / ST / RR)
    TWK_HLIGHT = ('x y - abs {i} < {neutral} {TWK} {neutral} {TWK} - {TWK} {neutral} / * + {TWK0} {TWK} {LD} + / * '
        '{neutral} {TWK} - {j} / dup * {neutral} {TWK} - {j} / dup * {HD} + / * {neutral} + ?'.format(
            i=1 << (bits-8), neutral=neutral, TWK=TWK, TWK0=TWK0, LD=LD, j=scale(20, bits), HD=HD))

    i = clp if not interlaced else core.std.SeparateFields(clp, tff=True)
    oxi = i.width
    oyi = i.height
    sm = core.resize.Bicubic(i, haf_m4(oxi/rx), haf_m4(oyi/ry), filter_param_a=1/3, filter_param_b=1/3)
    mm = core.std.Expr([sm.std.Maximum(), sm.std.Minimum()], ['x y - 4 *']).std.Maximum().std.Deflate().std.Convolution([1]*9)
    mm = mm.std.Inflate().resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0).std.Inflate()
    sm = core.resize.Bicubic(sm, oxi, oyi, filter_param_a=1, filter_param_b=0)
    smd = core.std.Expr([Sharpen(i, tweaker), sm], [TWK_HLIGHT])
    if sharpness != 0.:
        smd = Blur(smd, sharpness)
    clean = core.std.Expr([i, smd], ['x y {neutral} - -'.format(neutral=neutral)])
    clean = core.std.MaskedMerge(i, clean, mm)

    if PPmode != 0:
        LL = scale(PPlimit, bits)
        LIM = 'x {LL} + y < x {LL} + x {LL} - y > x {LL} - y ? ?'.format(LL=LL)

        base = i if PPmode < 0 else clean
        small = core.resize.Bicubic(base, haf_m4(oxi / math.sqrt(rx * 1.5)), haf_m4(oyi / math.sqrt(ry * 1.5)), filter_param_a=1/3, filter_param_b=1/3)
        ex1 = Blur(small.std.Maximum(), 0.5)
        in1 = Blur(small.std.Minimum(), 0.5)
        hull = core.std.Expr([ex1.std.Maximum().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]), ex1, in1,
            in1.std.Minimum().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])],
            ['x y - {i} - 5 * z a - {i} - 5 * max'.format(i=1 << (bits-8))]).resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0)

        if abs(PPmode) == 1:
            postclean = core.std.MaskedMerge(base, small.resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0), hull)
        elif abs(PPmode) == 2:
            postclean = core.std.MaskedMerge(base, base.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]), hull)
        elif abs(PPmode) == 3:
            postclean = core.std.MaskedMerge(base, base.std.Median(), hull)
        else:
            raise ValueError(funcName + ': \"PPmode\" must be in [-3 ... 3]!')
    else:
        postclean = clean

    if PPlimit != 0:
        postclean = core.std.Expr([base, postclean], [LIM])

    last = haf_Weave(postclean, tff=True) if interlaced else postclean

    if not isGray:
        last = core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return last


def dfttestMC(input: vs.VideoNode, pp: Optional[vs.VideoNode] = None, mc: int = 2, mdg: bool = False,
              planes: PlanesType = None, sigma: Optional[float] = None, sbsize: Optional[int] = None,
              sosize: Optional[int] = None, tbsize: Optional[int] = None, mdgSAD: Optional[int] = None,
              thSAD: Optional[int] = None, thSCD1: Optional[int] = None, thSCD2: Optional[int] = None,
              pel: Optional[int] = None, blksize: Optional[int] = None, search: Optional[int] = None,
              searchparam: Optional[int] = None, overlap: int = 2, dct: Optional[int] = None,
              **dfttest_params: Any
              ) -> vs.VideoNode:
    """Avisynth's dfttestMC

    Motion-compensated dfttest
    Aka: Really Really Really Slow

    Author: thewebchat (https://forum.doom9.org/showthread.php?p=1295788#post1295788)

    Notes:
        \"lsb\" and \"dither\" are removed. The output always has the same bitdepth as input.
        "Y", "U" and "V" are replaced by "planes".
        "dfttest_params" is removed. Additional arguments will be passed to DFTTest by keyword arguments.
        mc can be 0, and the function will simply be a pure dfttest().

    Args:

        input: Input clip.

        pp: (clip) Clip to calculate vectors from. Default is \"input\".

        mc: (int) Number of frames in each direction to compensate. Range: 0 ~ 5. Default is 2.

        mdg: (bool) Run MDeGrain before dfttest. Default is False.

        mdgSAD: (int) thSAD for MDeGrain. Default is undefined.

        dfttest's sigma, sbsize, sosize and tbsize are supported. Extra dfttest parameters may be passed via "dfttest_params".

        pel, thSCD, thSAD, blksize, overlap, dct, search, and searchparam are also supported.

        sigma is the main control of dfttest strength.
        tbsize should not be set higher than mc * 2 + 1.

    """

    funcName = 'dfttestMC'

    if not isinstance(input, vs.VideoNode) or input.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"input\" must be a Gray or YUV clip!')

    mc = min(max(int(mc), 0), 5)

    if mc == 0:
        return core.dfttest.DFTTest(input, sigma=sigma, sbsize=sbsize, sosize=sosize, tbsize=tbsize, **dfttest_params)
    else:
        if pp is not None:
            if not isinstance(pp, vs.VideoNode):
                raise TypeError(funcName + ': \"pp\" must be a clip!')
            if input.format.id != pp.format.id:
                raise TypeError(funcName + ': \"pp\" must be of the same format as \"input\"!')
            if input.width != pp.width or input.height != pp.height:
                raise TypeError(funcName + ': \"pp\" must be of the same size as \"input\"!')

        # Set chroma parameters.
        if planes is None:
            planes = list(range(input.format.num_planes))
        elif isinstance(planes, int):
            planes = [planes]

        Y = 0 in planes
        U = 1 in planes
        V = 2 in planes

        chroma = U or V

        if not Y and U and not V:
            plane = 1
        elif not Y and not U and V:
            plane = 2
        elif not Y and chroma:
            plane = 3
        elif Y and chroma:
            plane = 4
        else:
            plane = 0

        # Prepare supersampled clips.
        pp_super = (
            haf_DitherLumaRebuild(pp if pp is not None else input, s0=1, chroma=chroma)
            .mv.Super(pel=pel, chroma=chroma))

        super = core.mv.Super(input, levels=1, pel=pel, chroma=chroma)

        # Motion vector search.
        analyse = functools.partial(
            core.mv.Analyse,
            super=pp_super, chroma=chroma, search=search, searchparam=searchparam,
            overlap=overlap, blksize=blksize, dct=dct)

        bvecs = [analyse(delta=i, isb=True) for i in range(1, mc+1)]
        fvecs = [analyse(delta=i, isb=False) for i in range(1, mc+1)]

        # Optional MDegrain.
        if mdg:
            r = min(mc, 3) # radius

            degrain = functools.partial(
                eval(f"core.mv.Degrain{r}"),
                thsad=mdgSAD, plane=plane, thscd1=thSCD1, thscd2=thSCD2)

            degrained = degrain(input, super, *itertools.chain.from_iterable(zip(bvecs[:r], fvecs[:r])))
            degrained_super = core.mv.Super(degrained, levels=1, pel=pel, chroma=chroma)
        else:
            degrained = input
            degrained_super = super

        # Motion Compensation.
        compensate = functools.partial(
            core.mv.Compensate,
            clip=degrained, super=degrained_super, thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)

        bclips = [compensate(vectors=bvec) for bvec in bvecs]
        fclips = [compensate(vectors=fvec) for fvec in fvecs]

        # Create compensated clip.
        interleaved = core.std.Interleave(fclips[::-1] + [degrained] + bclips)

        # Perform dfttest.
        filtered = core.dfttest.DFTTest(
            interleaved, sigma=sigma, sbsize=sbsize, sosize=sosize, tbsize=tbsize, **dfttest_params)

        return core.std.SelectEvery(filtered, mc * 2 + 1, mc)


def TurnLeft(clip: vs.VideoNode) -> vs.VideoNode:
    """Avisynth's internel function TurnLeft()"""

    return core.std.Transpose(clip).std.FlipVertical()


def TurnRight(clip: vs.VideoNode) -> vs.VideoNode:
    """Avisynth's internel function TurnRight()"""

    return core.std.FlipVertical(clip).std.Transpose()


def BalanceBorders(c: vs.VideoNode, cTop: int = 0, cBottom: int = 0, cLeft: int = 0, cRight: int = 0,
                   thresh: int = 128, blur: int = 999) -> vs.VideoNode:
    """Avisynth's BalanceBorders() Version: v0.2

    Author: PL (https://www.dropbox.com/s/v8fm6om7hm1dz0b/BalanceBorders.avs)

    The following documentaion is mostly translated by Google Translate from Russian.

    The function changes the values of the extreme pixels of the clip,
    so that they are "more similar" to the neighboring ones,
    which, perhaps, will prevent the "strong" use of Crop () to remove the "unpleasant edges"
    that are not very different from the "main" image.

    Args:
        c: Input clip. The image area "in the middle" does not change during processing.
            The clip can be any format, which differs from Avisynth's equivalent.

        cTop, cBottom, cLeft, cRight: (int) The number of variable pixels on each side.
            There will not be anything very terrible if you specify values that are greater than the minimum required in your case,
            but to achieve a good result, "it is better not to" ...
            Range: 0 will skip the processing. For RGB input, the range is 2~inf.
                For YUV input, the minimum accepted value depends on chroma subsampling.
                Specifically, for YV24, the range is also 2~inf. For YV12, the range is 4~inf.
            Default is 0.

        thresh: (int) Threshold of acceptable changes for local color matching in 8 bit scale.
            Range: 0~128. Recommend: [0~16 or 128].
            Default is 128.

        blur: (int) Degree of blur for local color matching.
            Smaller values give a more accurate color match,
            larger values give a more accurate picture transfer.
            Range: 1~inf. Recommend: [1~20 or 999].
            Default is 999.

    Notes:
        1) At default values ​​of thresh = 128 blur = 999,
            you will get a series of pixels that have been changed only by selecting the color for each row in its entirety, without local selection;
            The colors of neighboring pixels may be very different in some places, but there will be no change in the nature of the picture.

            And with thresh = 128 and blur = 1 you get almost the same rows of pixels,
            i.e. The colors between them will coincide completely, but the original pattern will be lost.

        2) Beware of using a large number of pixels to change in combination with a high level of "thresh",
            and a small "blur" that can lead to unwanted artifacts "in a clean place".
            For each function call, try to set as few pixels as possible to change and as low a threshold as possible "thresh" (when using blur 0..16).

    Examples:
        The variant of several calls of the order:
        last = muf.BalanceBorders(last, 7, 6, 4, 4)                    # "General" color matching
        last = muf.BalanceBorders(last, 5, 5, 4, 4, thresh=2, blur=10) # Very slightly changes a large area (with a "margin")
        last = muf.BalanceBorders(last, 3, 3, 2, 2, thresh=8, blur=4)  # Slightly changes the "main problem area"

    """

    funcName = 'BalanceBorders'

    if not isinstance(c, vs.VideoNode):
        raise TypeError(funcName + ': \"c\" must be a clip!')

    if c.format.sample_type != vs.INTEGER:
        raise TypeError(funcName+': \"c\" must be integer format!')

    if blur <= 0:
        raise ValueError(funcName + ': \'blur\' have not a correct value! (0 ~ inf]')

    if thresh <= 0:
        raise ValueError(funcName + ': \'thresh\' have not a correct value! (0 ~ inf]')

    last = c

    if cTop > 0:
        last = _BalanceTopBorder(last, cTop, thresh, blur)

    last = TurnRight(last)

    if cLeft > 0:
        last = _BalanceTopBorder(last, cLeft, thresh, blur)

    last = TurnRight(last)

    if cBottom > 0:
        last = _BalanceTopBorder(last, cBottom, thresh, blur)

    last = TurnRight(last)

    if cRight > 0:
        last = _BalanceTopBorder(last, cRight, thresh, blur)

    last = TurnRight(last)

    return last


def _BalanceTopBorder(c: vs.VideoNode, cTop: int, thresh: int, blur: int) -> vs.VideoNode:
    """BalanceBorders()'s helper function"""

    cWidth = c.width
    cHeight = c.height
    cTop = min(cTop, cHeight - 1)
    blurWidth = max(4, math.floor(cWidth / blur))

    c2 = mvf.PointPower(c, 1, 1)

    last = core.std.Crop(c2, 0, 0, cTop*2, (cHeight - cTop - 1) * 2)
    last = core.resize.Point(last, cWidth * 2, cTop * 2)
    last = core.resize.Bilinear(last, blurWidth * 2, cTop * 2)
    last = core.std.Convolution(last, [1, 1, 1], mode='h')
    last = core.resize.Bilinear(last, cWidth * 2, cTop * 2)
    referenceBlur = last

    original = core.std.Crop(c2, 0, 0, 0, (cHeight - cTop) * 2)

    last = original
    last = core.resize.Bilinear(last, blurWidth * 2, cTop * 2)
    last = core.std.Convolution(last, [1, 1, 1], mode='h')
    last = core.resize.Bilinear(last, cWidth * 2, cTop * 2)
    originalBlur = last

    """
    balanced = core.std.Expr([original, originalBlur, referenceBlur], ['z y - x +'])
    difference = core.std.MakeDiff(balanced, original)

    tp = scale(128 + thresh, c.format.bits_per_sample)
    tm = scale(128 - thresh, c.format.bits_per_sample)
    difference = core.std.Expr([difference], ['x {tp} min {tm} max'.format(tp=tp, tm=tm)])

    last = core.std.MergeDiff(original, difference)
    """
    tp = scale(thresh, c.format.bits_per_sample)
    tm = -tp
    last = core.std.Expr([original, originalBlur, referenceBlur], ['z y - {tp} min {tm} max x +'.format(tp=tp, tm=tm)])

    return core.std.StackVertical([last, core.std.Crop(c2, 0, 0, cTop * 2, 0)]).resize.Point(cWidth, cHeight)


def DisplayHistogram(clip: vs.VideoNode, factor: float = 100) -> vs.VideoNode:
    """A simple function to display the histogram of an image.

    The right and bottom of the output is the histogram along the horizontal/vertical axis,
    with the left(bottom) side of the graph represents luma=0 and the right(above) side represents luma=255.
    The bottom right is hist.Levels().

    More details of the graphs can be found at http://avisynth.nl/index.php/Histogram.

    Args:
        clip: Input clip. Must be constant format 8..16 bit integer YUV input.
            If the input's bitdepth is not 8, input will be converted to 8 bit before passing to hist.Levels().

        factor: (float) hist.Levels()'s argument.
            It specifies how the histograms are displayed, exaggerating the vertical scale.
            It is specified as percentage of the total population (that is number of luma or chroma pixels in a frame).
            Range: 0~100. Default is 100.

    """

    funcName = 'DisplayHistogram'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if clip.format.sample_type != vs.INTEGER or clip.format.bits_per_sample > 16 or clip.format.color_family != vs.YUV:
        raise TypeError(funcName+': \"clip\" must be 8..16 integer YUV format!')

    histogram_v = core.hist.Classic(clip)

    clip_8 = mvf.Depth(clip, 8)
    levels = core.hist.Levels(clip_8, factor=factor).std.Crop(left=clip.width, right=0, top=0, bottom=clip.height - 256)
    if clip.format.bits_per_sample != 8:
        levels = mvf.Depth(levels, clip.format.bits_per_sample)
    histogram_h = TurnLeft(core.hist.Classic(clip.std.Transpose()).std.Crop(left=clip.height))

    bottom = core.std.StackHorizontal([histogram_h, levels])

    return core.std.StackVertical([histogram_v, bottom])


def GuidedFilter(input: vs.VideoNode, guidance: Optional[vs.VideoNode] = None, radius: int = 4,
                 regulation: float = 0.01, regulation_mode: int = 0, use_gauss: bool = False,
                 fast: Optional[bool] = None, subsampling_ratio: float = 4, use_fmtc1: bool = False,
                 kernel1: str = 'point', kernel1_args: Optional[Dict[str, Any]] = None, use_fmtc2: bool = False,
                 kernel2: str = 'bilinear', kernel2_args: Optional[Dict[str, Any]] = None,
                 **depth_args: Any
                 ) -> vs.VideoNode:
    """Guided Filter - fast edge-preserving smoothing algorithm

    Author: Kaiming He et al. (http://kaiminghe.com/eccv10/)

    The guided filter computes the filtering output by considering the content of a guidance image.

    It can be used as an edge-preserving smoothing operator like the popular bilateral filter,
    but it has better behaviors near edges.

    The guided filter is also a more generic concept beyond smoothing:
    It can transfer the structures of the guidance image to the filtering output,
    enabling new filtering applications like detail enhancement, HDR compression,
    image matting/feathering, dehazing, joint upsampling, etc.

    All the internal calculations are done at 32-bit float.

    Args:
        input: Input clip.

        guidance: (clip) Guidance clip used to compute the coefficient of the linear translation on 'input'.
            It must has the same clip properties as 'input'.
            If it is None, it will be set to input, with duplicate calculations being omitted.
            Default is None.

        radius: (int) Box / Gaussian filter's radius.
            If box filter is used, the range of radius is 1 ~ 12(fast=False) or 1 ~ 12*subsampling_ratio in VapourSynth R38 or older
                because of the limitation of std.Convolution().
            For gaussian filter, the radius can be much larger, even reaching the width/height of the clip.
            Default is 4.

        regulation: (float) A criterion for judging whether a patch has high variance and should be preserved, or is flat and should be smoothed.
            Similar to the range variance in the bilateral filter.
            Default is 0.01.

        regulation_mode: (int) Tweak on regulation.
            It was mentioned in [1] that the local filters such as the Bilateral Filter (BF) or Guided Image Filter (GIF)
            would concentrate the blurring near these edges and introduce halos.

            The author of Weighted Guided Image Filter (WGIF) [3] argued that,
            the Lagrangian factor (regulation) in the GIF is fixed could be another major reason that the GIF produces halo artifacts.

            In [3], a WGIF was proposed to reduce the halo artifacts of the GIF.
            An edge aware factor was introduced to the constraint term of the GIF,
            the factor makes the edges preserved better in the result images and thus reduces the halo artifacts.

            In [4], a gradient domain guided image filter is proposed by incorporating an explicit first-order edge-aware constraint.
            The proposed filter is based on local optimization
            and the cost function is composed of a zeroth order data fidelity term and a first order regularization term.
            So the factors in the new local linear model can represent the images more accurately near edges.
            In addition, the edge-aware factor is multi-scale, which can separate edges of an image from fine details of the image better.

            0: Guided Filter [1]
            1: Weighted Guided Image Filter [3]
            2: Gradient Domain Guided Image Filter [4]
            Default is 0.

        use_gauss: (bool) Whether to use gaussian guided filter [1]. This replaces mean filter with gaussian filter.
            Guided filter is rotationally asymmetric and slightly biases to the x/y-axis because a box window is used in the filter design.
            The problem can be solved by using a gaussian weighted window instead. The resulting kernels are rotationally symmetric.
            The authors of [1] suggest that in practice the original guided filter is always good enough.
            Gaussian is performed by core.tcanny.TCanny(mode=-1).
            The sigma is set to r/sqrt(2).
            Default is False.

        fast: (bool) Whether to use fast guided filter [2].
            This method subsamples the filtering input image and the guidance image,
            computes the local linear coefficients, and upsamples these coefficients.
            The upsampled coefficients are adopted on the original guidance image to produce the output.
            This method reduces the time complexity from O(N) to O(N/s^2) for a subsampling ratio s.
            Default is True if the version number of VapourSynth is less than 39, otherwise is False.

        subsampling_ratio: (float) Only works when fast=True.
            Generally should be no less than 'radius'.
            Default is 4.

        use_fmtc1, use_fmtc2: (bool) Whether to use fmtconv in subsampling / upsampling.
            Default is False.
            Note that fmtconv's point subsampling may causes pixel shift.

        kernel1, kernel2: (string) Subsampling/upsampling kernels.
            Default is 'point'and 'bilinear'.

        kernel1_args, kernel2_args: (dict) Additional parameters passed to resizers in the form of dict.
            Default is {}.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] He, K., Sun, J., & Tang, X. (2013). Guided image filtering.
            IEEE transactions on pattern analysis and machine intelligence, 35(6), 1397-1409.
        [2] He, K., & Sun, J. (2015). Fast guided filter. arXiv preprint arXiv:1505.00996.
        [3] http://kaiminghe.com/eccv10/index.html
        [4] Li, Z., Zheng, J., Zhu, Z., Yao, W., & Wu, S. (2015). Weighted guided image filtering.
            IEEE Transactions on Image Processing, 24(1), 120-129.
        [5] Kou, F., Chen, W., Wen, C., & Li, Z. (2015). Gradient domain guided image filtering.
            IEEE Transactions on Image Processing, 24(11), 4528-4539.
        [6] http://koufei.weebly.com/

    """

    funcName = 'GuidedFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    # Get clip's properties
    bits = input.format.bits_per_sample
    sampleType = input.format.sample_type
    width = input.width
    height = input.height

    if guidance is not None:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if input.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"input\"!')
        if input.width != guidance.width or input.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"input\"!')
        if input == guidance: # Remove redundant computation
            guidance = None

    if fast is None:
        fast = False if core.version_number() >= 39 else True

    if kernel1_args is None:
        kernel1_args = {}
    if kernel2_args is None:
        kernel2_args = {}

    # Bitdepth conversion and variable names modification to correspond to the paper
    p = mvf.Depth(input, depth=32, sample=vs.FLOAT, **depth_args)
    I = mvf.Depth(guidance, depth=32, sample=vs.FLOAT, **depth_args) if guidance is not None else p
    r = radius
    eps = regulation
    s = subsampling_ratio

    # Back up guidance image
    I_src = I

    # Fast guided filter's subsampling
    if fast:
        down_w = math.floor(width / s)
        down_h = math.floor(height / s)
        kernel1 = kernel1.capitalize()
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, **kernel1_args) if guidance is not None else p
        else: # use zimg
            p = eval(f'core.resize.{kernel1}')(p, down_w, down_h, **kernel1_args)
            I = eval(f'core.resize.{kernel1}')(I, down_w, down_h, **kernel1_args) if guidance is not None else p

        r = math.floor(r / s)

    # Select the shape of the kernel. As the width of BoxFilter in this module is (radius*2-1) rather than (radius*2+1), radius should be increased by one.
    Filter = functools.partial(core.tcanny.TCanny, sigma=r/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=r+1)
    Filter_r1 = functools.partial(core.tcanny.TCanny, sigma=1/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=1+1)


    # Edge-Aware Weighting, equation (5) in [3], or equation (9) in [4].
    def FLT(n: int, f: vs.VideoFrame, clip: vs.VideoNode, core: vs.Core, eps0: float) -> vs.VideoNode:
        frameMean = f.props['PlaneStatsAverage']

        return core.std.Expr(clip, ['x {eps0} + {avg} *'.format(avg=frameMean, eps0=eps0)])


    # Compute the optimal value of a of Gradient Domain Guided Image Filter, equation (12) in [4]
    def FLT2(n: int, f: vs.VideoFrame, cov_Ip: vs.VideoNode, weight_in: vs.VideoNode, weight: vs.VideoNode,
             var_I: vs.VideoNode, core: vs.Core, eps: float
             ) -> vs.VideoNode:
        frameMean = typing.cast(float, f.props['PlaneStatsAverage'])
        frameMin = typing.cast(float, f.props['PlaneStatsMin'])

        alpha = frameMean
        kk = -4 / (frameMin - alpha - 1e-6) # Add a small num to prevent divided by 0

        return core.std.Expr([cov_Ip, weight_in, weight, var_I],
            ['x {eps} 1 1 1 {kk} y {alpha} - * exp + / - * z / + a {eps} z / + /'.format(eps=eps, kk=kk, alpha=alpha)])

    # Compute local linear coefficients.
    mean_p = Filter(p)
    mean_I = Filter(I) if guidance is not None else mean_p
    I_square = core.std.Expr([I], ['x dup *'])
    corr_I = Filter(I_square)
    corr_Ip = Filter(core.std.Expr([I, p], ['x y *'])) if guidance is not None else corr_I

    var_I = core.std.Expr([corr_I, mean_I], ['x y dup * -'])
    cov_Ip = core.std.Expr([corr_Ip, mean_I, mean_p], ['x y z * -']) if guidance is not None else var_I

    if regulation_mode: # 0: Original Guided Filter, 1: Weighted Guided Image Filter, 2: Gradient Domain Guided Image Filter
        if r != 1:
            mean_I_1 = Filter_r1(I)
            corr_I_1 = Filter_r1(I_square)
            var_I_1 = core.std.Expr([corr_I_1, mean_I_1], ['x y dup * -'])
        else: # r == 1
            var_I_1 = var_I

        if regulation_mode == 1: # Weighted Guided Image Filter
            weight_in = var_I_1
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.Expr([var_I, var_I_1], ['x y * sqrt'])

        eps0 = 0.001 ** 2 # Epsilon in [3] and [4]
        denominator = core.std.Expr([weight_in], ['1 x {} + /'.format(eps0)])

        denominator = core.std.PlaneStats(denominator, plane=0)
        # equation (5) in [3], or equation (9) in [4]
        if _has_lexpr:
            avg = 'y.PlaneStatsAverage'
            weight = core.akarin.Expr([weight_in, denominator], 'x {eps0} + {avg} *'.format(eps0=eps0, avg=avg))
        else:
            weight = core.std.FrameEval(denominator, functools.partial(FLT, clip=weight_in, core=core, eps0=eps0), prop_src=[denominator])

        if regulation_mode == 1: # Weighted Guided Image Filter
            a = core.std.Expr([cov_Ip, var_I, weight], ['x y {eps} z / + /'.format(eps=eps)])
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.PlaneStats(weight_in, plane=0)
            if _has_lexpr:
                alpha = 'y.PlaneStatsAverage'
                kk = '-4 y.PlaneStatsMin {alpha} - 1e-6 - /'.format(alpha=alpha)
                a = core.akarin.Expr([cov_Ip, weight_in, weight, var_I],
                    ['x {eps} 1 1 1 {kk} y {alpha} - * exp + / - * z / + a {eps} z / + /'.format(eps=eps, kk=kk, alpha=alpha)])
            else:
                a = core.std.FrameEval(weight, functools.partial(FLT2, cov_Ip=cov_Ip, weight_in=weight_in, weight=weight,
                    var_I=var_I, core=core, eps=eps), prop_src=[weight_in])
    else: # regulation_mode == 0, Original Guided Filter
        if cov_Ip is var_I:
            a = core.std.Expr([cov_Ip], ['x x {} + /'.format(eps)])
        else:
            a = core.std.Expr([cov_Ip, var_I], ['x y {} + /'.format(eps)])

    if mean_p is mean_I:
        b = core.std.Expr([mean_p, a], ['x y x * -'])
    else:
        b = core.std.Expr([mean_p, a, mean_I], ['x y z * -'])

    mean_a = Filter(a)
    mean_b = Filter(b)

    # Fast guided filter's upsampling
    if fast:
        kernel2 = kernel2.capitalize()
        if use_fmtc2:
            mean_a = core.fmtc.resample(mean_a, width, height, kernel=kernel2, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, **kernel2_args)
        else: # use zimg
            mean_a = eval(f'core.resize.{kernel2}')(mean_a, width, height, **kernel2_args)
            mean_b = eval(f'core.resize.{kernel2}')(mean_b, width, height, **kernel2_args)

    # Linear translation
    q = core.std.Expr([mean_a, I_src, mean_b], ['x y * z +'])

    # Final bitdepth conversion
    return mvf.Depth(q, depth=bits, sample=sampleType, **depth_args)


def GuidedFilterColor(input: vs.VideoNode, guidance: vs.VideoNode, radius: int = 4,
                      regulation: float = 0.01, use_gauss: bool = False, fast: Optional[bool] = None,
                      subsampling_ratio: float = 4, use_fmtc1: bool = False, kernel1: str = 'point',
                      kernel1_args: Optional[Dict[str, Any]] = None, use_fmtc2: bool = False,
                      kernel2: str = 'bilinear', kernel2_args: Optional[Dict[str, Any]] = None,
                      **depth_args: Any
                      ) -> vs.VideoNode:
    """Guided Filter Color - fast edge-preserving smoothing algorithm using a color image as the guidance

    Author: Kaiming He et al. (http://kaiminghe.com/eccv10/)

    Most of the description of the guided filter can be found in the documentation of native guided filter above.
    Only the native guided filter is implemented.

    A color guidance image can better preserve the edges that are not distinguishable in gray-scale.

    It is also essential in the matting/feathering and dehazing applications,
    because the local linear model is more likely to be valid in the RGB color space than in gray-scale.

    Args:
        input: Input clip. It should be a gray-scale/single channel image.

        guidance: Guidance clip used to compute the coefficient of the linear translation on 'input'.
            It must has no subsampling for the second and third plane in horizontal/vertical direction, e.g. RGB or YUV444.

        Descriptions of other parameter can be found in the documentation of native guided filter above.

    Ref:
        [1] He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE transactions on pattern analysis and machine intelligence, 35(6), 1397-1409.
        [2] He, K., & Sun, J. (2015). Fast guided filter. arXiv preprint arXiv:1505.00996.
        [3] http://kaiminghe.com/eccv10/index.html

    """

    funcName = 'GuidedFilterColor'

    if not isinstance(input, vs.VideoNode) or input.format.num_planes > 1:
        raise TypeError(funcName + ': \"input\" must be a gray-scale/single channel clip!')

    # Get clip's properties
    bits = input.format.bits_per_sample
    sampleType = input.format.sample_type
    width = input.width
    height = input.height

    if not isinstance(guidance, vs.VideoNode) or guidance.format.subsampling_w != 0 or guidance.format.subsampling_h != 0:
        raise TypeError(funcName + ': \"guidance\" must be a RGB or YUV444 clip!')
    if input.width != guidance.width or input.height != guidance.height:
        raise ValueError(funcName + ': \"guidance\" must be of the same size as \"input\"!')

    if fast is None:
        fast = False if core.version_number() >= 39 else True

    if kernel1_args is None:
        kernel1_args = {}
    if kernel2_args is None:
        kernel2_args = {}

    # Bitdepth conversion and variable names modification to correspond to the paper
    p = mvf.Depth(input, depth=32, sample=vs.FLOAT, **depth_args)
    I = mvf.Depth(guidance, depth=32, sample=vs.FLOAT, **depth_args)
    r = radius
    eps = regulation
    s = subsampling_ratio

    # Back up guidance image
    I_src_r = mvf.GetPlane(I, 0)
    I_src_g = mvf.GetPlane(I, 1)
    I_src_b = mvf.GetPlane(I, 2)

    # Fast guided filter's subsampling
    if fast:
        down_w = math.floor(width / s)
        down_h = math.floor(height / s)
        kernel1 = kernel1.capitalize()
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, **kernel1_args)
        else: # use zimg
            p = eval(f'core.resize.{kernel1}')(p, down_w, down_h, **kernel1_args)
            I = eval(f'core.resize.{kernel1}')(I, down_w, down_h, **kernel1_args) if guidance is not None else p

        r = math.floor(r / s)

    # Select kernel shape. As the width of BoxFilter in this module is (radius*2-1) rather than (radius*2+1), radius should be be incremented by one.
    Filter = functools.partial(core.tcanny.TCanny, sigma=r/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=r+1)

    # Seperate planes
    I_r = mvf.GetPlane(I, 0)
    I_g = mvf.GetPlane(I, 1)
    I_b = mvf.GetPlane(I, 2)

    # Compute local linear coefficients.
    mean_p = Filter(p)

    mean_I_r = Filter(I_r)
    mean_I_g = Filter(I_g)
    mean_I_b = Filter(I_b)

    corr_I_rr = Filter(core.std.Expr([I_r], ['x dup *']))
    corr_I_rg = Filter(core.std.Expr([I_r, I_g], ['x y *']))
    corr_I_rb = Filter(core.std.Expr([I_r, I_b], ['x y *']))
    corr_I_gg = Filter(core.std.Expr([I_g], ['x dup *']))
    corr_I_gb = Filter(core.std.Expr([I_g, I_b], ['x y *']))
    corr_I_bb = Filter(core.std.Expr([I_b], ['x dup *']))

    corr_Ip_r = Filter(core.std.Expr([I_r, p], ['x y *']))
    corr_Ip_g = Filter(core.std.Expr([I_g, p], ['x y *']))
    corr_Ip_b = Filter(core.std.Expr([I_b, p], ['x y *']))

    var_I_rr = core.std.Expr([corr_I_rr, mean_I_r], ['x y dup * - {} +'.format(eps)])
    var_I_gg = core.std.Expr([corr_I_gg, mean_I_g], ['x y dup * - {} +'.format(eps)])
    var_I_bb = core.std.Expr([corr_I_bb, mean_I_b], ['x y dup * - {} +'.format(eps)])

    cov_I_rg = core.std.Expr([corr_I_rg, mean_I_r, mean_I_g], ['x y z * -'])
    cov_I_rb = core.std.Expr([corr_I_rb, mean_I_r, mean_I_b], ['x y z * -'])
    cov_I_gb = core.std.Expr([corr_I_gb, mean_I_g, mean_I_b], ['x y z * -'])

    cov_Ip_r = core.std.Expr([corr_Ip_r, mean_I_r, mean_p], ['x y z * -'])
    cov_Ip_g = core.std.Expr([corr_Ip_g, mean_I_g, mean_p], ['x y z * -'])
    cov_Ip_b = core.std.Expr([corr_Ip_b, mean_I_b, mean_p], ['x y z * -'])

    # Inverse of Sigma + eps * I
    inv_rr = core.std.Expr([var_I_gg, var_I_bb, cov_I_gb], ['x y * z dup * -'])
    inv_rg = core.std.Expr([cov_I_gb, cov_I_rb, cov_I_rg, var_I_bb], ['x y * z a * -'])
    inv_rb = core.std.Expr([cov_I_rg, cov_I_gb, var_I_gg, cov_I_rb], ['x y * z a * -'])
    inv_gg = core.std.Expr([var_I_rr, var_I_bb, cov_I_rb], ['x y * z dup * -'])
    inv_gb = core.std.Expr([cov_I_rb, cov_I_rg, var_I_rr, cov_I_gb], ['x y * z a * -'])
    inv_bb = core.std.Expr([var_I_rr, var_I_gg, cov_I_rg], ['x y * z dup * -'])

    covDet = core.std.Expr([inv_rr, var_I_rr, inv_rg, cov_I_rg, inv_rb, cov_I_rb], ['x y * z a * + b c * +'])

    inv_rr = core.std.Expr([inv_rr, covDet], ['x y /'])
    inv_rg = core.std.Expr([inv_rg, covDet], ['x y /'])
    inv_rb = core.std.Expr([inv_rb, covDet], ['x y /'])
    inv_gg = core.std.Expr([inv_gg, covDet], ['x y /'])
    inv_gb = core.std.Expr([inv_gb, covDet], ['x y /'])
    inv_bb = core.std.Expr([inv_bb, covDet], ['x y /'])

    a_r = core.std.Expr([inv_rr, cov_Ip_r, inv_rg, cov_Ip_g, inv_rb, cov_Ip_b], ['x y * z a * + b c * +'])
    a_g = core.std.Expr([inv_rg, cov_Ip_r, inv_gg, cov_Ip_g, inv_gb, cov_Ip_b], ['x y * z a * + b c * +'])
    a_b = core.std.Expr([inv_rb, cov_Ip_r, inv_gb, cov_Ip_g, inv_bb, cov_Ip_b], ['x y * z a * + b c * +'])

    b = core.std.Expr([mean_p, a_r, mean_I_r, a_g, mean_I_g, a_b, mean_I_b], ['x y z * - a b * - c d * -'])

    mean_a_r = Filter(a_r)
    mean_a_g = Filter(a_g)
    mean_a_b = Filter(a_b)
    mean_b = Filter(b)

    # Fast guided filter's upsampling
    if fast:
        kernel2 = kernel2.capitalize()
        if use_fmtc2:
            mean_a_r = core.fmtc.resample(mean_a_r, width, height, kernel=kernel2, **kernel2_args)
            mean_a_g = core.fmtc.resample(mean_a_g, width, height, kernel=kernel2, **kernel2_args)
            mean_a_b = core.fmtc.resample(mean_a_b, width, height, kernel=kernel2, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, **kernel2_args)
        else: # use zimg
            mean_a_r = eval(f'core.resize.{kernel2}')(mean_a_r, width, height, **kernel2_args)
            mean_a_g = eval(f'core.resize.{kernel2}')(mean_a_g, width, height, **kernel2_args)
            mean_a_b = eval(f'core.resize.{kernel2}')(mean_a_b, width, height, **kernel2_args)
            mean_b = eval(f'core.resize.{kernel2}')(mean_b, width, height, **kernel2_args)

    # Linear translation
    q = core.std.Expr([mean_a_r, I_src_r, mean_a_g, I_src_g, mean_a_b, I_src_b, mean_b], ['x y * z a * + b c * + d +'])

    # Final bitdepth conversion
    return mvf.Depth(q, depth=bits, sample=sampleType, **depth_args)


def GMSD(clip1: vs.VideoNode, clip2: vs.VideoNode, plane: Optional[int] = None,
         downsample: bool = True, c: float = 0.0026, show_map: bool = False,
         **depth_args: Any
         ) -> vs.VideoNode:
    """Gradient Magnitude Similarity Deviation Calculator

    GMSD is a new effective and efficient image quality assessment (IQA) model, which utilizes the pixel-wise gradient magnitude similarity (GMS)
    between the reference and distorted images combined with standard deviation of the GMS map to predict perceptual image quality.

    The distortion degree of the distorted image will be stored as frame property 'PlaneGMSD' in the output clip.

    The value of GMSD reflects the range of distortion severities in an image.
    The lowerer the GMSD score, the higher the image perceptual quality.
    If "clip1" == "clip2", GMSD = 0.

    All the internal calculations are done at 32-bit float, only one channel of the image will be processed.

    Args:
        clip1: The distorted clip, will be copied to output if "show_map" is False.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        plane: (int) Specify which plane to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        c: (float) A positive constant that supplies numerical stability.
            According to the paper, for all the test databases, GMSD shows similar preference to the value of c.
            Default is 0.0026.

        show_map: (bool) Whether to return GMS map. If not, "clip1" will be returned. Default is False.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). Gradient magnitude similarity deviation:
            A highly efficient perceptual image quality index. IEEE Transactions on Image Processing, 23(2), 684-695.
        [2] http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm.

    """

    funcName = 'GMSD'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')
    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if clip1.format.id != clip2.format.id:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same format!')
    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    # Store the "clip1"
    clip1_src = clip1

    # Convert to float type grayscale image
    clip1 = mvf.GetPlane(clip1, plane)
    clip2 = mvf.GetPlane(clip2, plane)
    clip1 = mvf.Depth(clip1, depth=32, sample=vs.FLOAT, **depth_args)
    clip2 = mvf.Depth(clip2, depth=32, sample=vs.FLOAT, **depth_args)

    # Filtered by a 2x2 average filter and then down-sampled by a factor of 2, as in the implementation of SSIM
    if downsample:
        clip1 = _IQA_downsample(clip1)
        clip2 = _IQA_downsample(clip2)

    # Calculate gradients based on Prewitt filter
    clip1_dx = core.std.Convolution(clip1, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
    clip1_dy = core.std.Convolution(clip1, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
    clip1_grad_squared = core.std.Expr([clip1_dx, clip1_dy], ['x dup * y dup * +'])

    clip2_dx = core.std.Convolution(clip2, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
    clip2_dy = core.std.Convolution(clip2, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
    clip2_grad_squared = core.std.Expr([clip2_dx, clip2_dy], ['x dup * y dup * +'])

    # Compute the gradient magnitude similarity (GMS) map
    quality_map = core.std.Expr([clip1_grad_squared, clip2_grad_squared], ['2 x y * sqrt * {c} + x y + {c} + /'.format(c=c)])

    # The following code is modified from mvf.PlaneStatistics(), which is used to compute the standard deviation of the GMS map as GMSD
    if hasattr(core.std, 'PlaneStats'):
        map_mean = core.std.PlaneStats(quality_map, plane=0, prop='PlaneStats')
    else:
        map_mean = core.std.PlaneAverage(quality_map, plane=0, prop='PlaneStatsAverage') # type: ignore

    def _PlaneSDFrame(n: int, f: vs.VideoFrame, clip: vs.VideoNode, core: vs.Core) -> vs.VideoNode:
        mean = f.props['PlaneStatsAverage']
        expr = "x {mean} - dup *".format(mean=mean)
        return core.std.Expr(clip, expr)
    if _has_lexpr:
        mean = "y.PlaneStatsAverage"
        SDclip = core.akarin.Expr([quality_map, map_mean], "x {mean} - dup *".format(mean=mean))
    else:
        SDclip = core.std.FrameEval(quality_map, functools.partial(_PlaneSDFrame, clip=quality_map, core=core), map_mean)

    if hasattr(core.std, 'PlaneStats'):
        SDclip = core.std.PlaneStats(SDclip, plane=0, prop='PlaneStats')
    else:
        SDclip = core.std.PlaneAverage(SDclip, plane=0, prop='PlaneStatsAverage') # type: ignore

    def _PlaneGMSDTransfer(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['PlaneGMSD'] = math.sqrt(f[1].props['PlaneStatsAverage']) # type: ignore
        return fout
    output_clip = quality_map if show_map else clip1_src
    output_clip = core.std.ModifyFrame(output_clip, [output_clip, SDclip], selector=_PlaneGMSDTransfer)

    return output_clip


def SSIM(clip1: vs.VideoNode, clip2: vs.VideoNode, plane: Optional[int] = None,
         downsample: bool = True, k1: float = 0.01, k2: float = 0.03,
         fun: Optional[VSFuncType] = None, dynamic_range: int = 1,
         show_map: bool = False, **depth_args: Any
         ) -> vs.VideoNode:
    """Structural SIMilarity Index Calculator

    The Structural SIMilarity (SSIM) index is a method for measuring the similarity between two images.
    It is based on the hypothesis that the HVS is highly adapted for extracting structural information,
    which compares local patterns of pixel intensities that have been normalized for luminance and contrast.

    The mean SSIM (MSSIM) index value of the distorted image will be stored as frame property 'PlaneSSIM' in the output clip.

    The value of SSIM measures the structural similarity in an image.
    The higher the SSIM score, the higher the image perceptual quality.
    If "clip1" == "clip2", SSIM = 1.

    All the internal calculations are done at 32-bit float, only one channel of the image will be processed.

    Args:
        clip1: The distorted clip, will be copied to output if "show_map" is False.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        plane: (int) Specify which plane to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        k1, k2: (float) Constants in the SSIM index formula.
            According to the paper, the performance of the SSIM index algorithm is fairly insensitive to variations of these values.
            Default are 0.01 and 0.03.

        fun: (function or float) The function of how the clips are filtered.
            If it is None, it will be set to a gaussian filter whose standard deviation is 1.5.
            Note that the size of gaussian kernel is different from the one in MATLAB.
            If it is a float, it specifies the standard deviation of the gaussian filter. (sigma in core.tcanny.TCanny)
            According to the paper, the quality map calculated from gaussian filter exhibits a locally isotropic property,
            which prevents the present of undesirable “blocking” artifacts in the resulting SSIM index map.
            Default is None.

        dynamic_range: (float) Dynamic range of the internal float point clip. Default is 1.

        show_map: (bool) Whether to return SSIM index map. If not, "clip1" will be returned. Default is False.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), 600-612.
        [2] https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    """

    funcName = 'SSIM'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')
    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if clip1.format.id != clip2.format.id:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same format!')
    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    if fun is None:
        fun = functools.partial(core.tcanny.TCanny, sigma=1.5, mode=-1)
    elif isinstance(fun, (int, float)):
        fun = functools.partial(core.tcanny.TCanny, sigma=fun, mode=-1)
    elif not callable(fun):
        raise TypeError(funcName + ': \"fun\" must be a function or a float!')

    # Store the "clip1"
    clip1_src = clip1

    # Convert to float type grayscale image
    clip1 = mvf.GetPlane(clip1, plane)
    clip2 = mvf.GetPlane(clip2, plane)
    clip1 = mvf.Depth(clip1, depth=32, sample=vs.FLOAT, **depth_args)
    clip2 = mvf.Depth(clip2, depth=32, sample=vs.FLOAT, **depth_args)

    # Filtered by a 2x2 average filter and then down-sampled by a factor of 2
    if downsample:
        clip1 = _IQA_downsample(clip1)
        clip2 = _IQA_downsample(clip2)

    # Core algorithm
    mu1 = fun(clip1)
    mu2 = fun(clip2)
    mu1_sq = core.std.Expr([mu1], ['x dup *'])
    mu2_sq = core.std.Expr([mu2], ['x dup *'])
    mu1_mu2 = core.std.Expr([mu1, mu2], ['x y *'])
    sigma1_sq_pls_mu1_sq = fun(core.std.Expr([clip1], ['x dup *']))
    sigma2_sq_pls_mu2_sq = fun(core.std.Expr([clip2], ['x dup *']))
    sigma12_pls_mu1_mu2 = fun(core.std.Expr([clip1, clip2], ['x y *']))

    if c1 > 0 and c2 > 0:
        expr = '2 x * {c1} + 2 y x - * {c2} + * z a + {c1} + b c - d e - + {c2} + * /'.format(c1=c1, c2=c2)
        expr_clips = [mu1_mu2, sigma12_pls_mu1_mu2, mu1_sq, mu2_sq, sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq]
        ssim_map = core.std.Expr(expr_clips, [expr])
    else:
        denominator1 = core.std.Expr([mu1_sq, mu2_sq], ['x y + {c1} +'.format(c1=c1)])
        denominator2 = core.std.Expr([sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq], ['x y - z a - + {c2} +'.format(c2=c2)])

        numerator1_expr = '2 z * {c1} +'.format(c1=c1)
        numerator2_expr = '2 a z - * {c2} +'.format(c2=c2)
        expr = 'x y * 0 > {numerator1} {numerator2} * x y * / x 0 = not y 0 = and {numerator1} x / {i} ? ?'.format(numerator1=numerator1_expr,
            numerator2=numerator2_expr, i=1)
        ssim_map = core.std.Expr([denominator1, denominator2, mu1_mu2, sigma12_pls_mu1_mu2], [expr])

    # The following code is modified from mvf.PlaneStatistics(), which is used to compute the mean of the SSIM index map as MSSIM
    if hasattr(core.std, 'PlaneStats'):
        map_mean = core.std.PlaneStats(ssim_map, plane=0, prop='PlaneStats')
    else:
        map_mean = core.std.PlaneAverage(ssim_map, plane=0, prop='PlaneStatsAverage') # type: ignore

    def _PlaneSSIMTransfer(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['PlaneSSIM'] = f[1].props['PlaneStatsAverage']
        return fout

    output_clip = ssim_map if show_map else clip1_src
    output_clip = core.std.ModifyFrame(output_clip, [output_clip, map_mean], selector=_PlaneSSIMTransfer)

    return output_clip


def _IQA_downsample(clip: vs.VideoNode) -> vs.VideoNode:
    """Downsampler for image quality assessment model.

    The “clip” is first filtered by a 2x2 average filter, and then down-sampled by a factor of 2.
    """

    return core.std.Convolution(clip, [1, 1, 0, 1, 1, 0, 0, 0, 0]).resize.Point(clip.width // 2, clip.height // 2, src_left=-1, src_top=-1)


def SSIM_downsample(clip: vs.VideoNode, w: int, h: int, smooth: Union[float, VSFuncType] = 1,
                    kernel: Optional[str] = None, use_fmtc: bool = False, gamma: bool = False,
                    fulls: bool = False, fulld: bool = False, curve: str = '709', sigmoid: bool = False,
                    epsilon: float = 1e-6, depth_args: Optional[Dict[str, Any]] = None,
                    **resample_args: Any) -> vs.VideoNode:
    """SSIM downsampler

    SSIM downsampler is an image downscaling technique that aims to optimize for the perceptual quality of the downscaled results.
    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured using famous Structural SIMilarity (SSIM) index.
    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.

    This is an pseudo-implementation of SSIM downsampler with slight modification.
    The pre-downsampling is performed by vszimg/fmtconv, and the behaviour of convolution at the border is uniform.

    All the internal calculations are done at 32-bit float, except gamma correction is done at integer.

    Args:
        clip: The input clip.

        w, h: The size of the output clip.

        smooth: (int, float or function) The method to smooth the image.
            If it's an int, it specifies the "radius" of the internel used boxfilter, i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
            If it's a float, it specifies the "sigma" of core.tcanny.TCanny, i.e. the standard deviation of gaussian blur.
            If it's a function, it acs as a general smoother.
            Default is 1. The 3x3 boxfilter will be performed.

        kernel: (string) Resample kernel of vszimg/fmtconv.
            Default is 'Bicubic'.

        use_fmtc: (bool) Whether to use fmtconv for downsampling. If not, vszimg (core.resize.*) will be used.
            Default is False.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

        gamma: (bool) Default is False.
            Set to true to turn on gamma correction for the y channel.

        fulls: (bool) Default is False.
            Specifies if the luma is limited range (False) or full range (True)

        fulld: (bool) Default is False.
            Same as fulls, but for output.

        curve: (string) Default is '709'.
            Type of gamma mapping.

        sigmoid: (bool) Default is False.
            When True, applies a sigmoidal curve after the power-like curve (or before when converting from linear to gamma-corrected).
            This helps reducing the dark halo artefacts around sharp edges caused by resizing in linear luminance.

        resample_args: (dict) Additional arguments passed to vszimg/fmtconv in the form of keyword arguments.
            Refer to the documentation of downsample() as an example.

            Default is {}.

    Ref:
        [1] Oeztireli, A. C., & Gross, M. (2015). Perceptually based downscaling of images. ACM Transactions on Graphics (TOG), 34(4), 77.

    """

    funcName = 'SSIM_downsample'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if depth_args is None:
        depth_args = {}

    if callable(smooth):
        Filter = smooth
    elif isinstance(smooth, int):
        Filter = functools.partial(BoxFilter, radius=smooth+1)
    elif isinstance(smooth, float):
        Filter = functools.partial(core.tcanny.TCanny, sigma=smooth, mode=-1)
    else:
        raise TypeError(funcName + ': \"smooth\" must be a int, float or a function!')

    if kernel is None:
        kernel = 'Bicubic'

    if gamma:
        import nnedi3_resample as nnrs
        clip = nnrs.GammaToLinear(mvf.Depth(clip, 16), fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    kernel = kernel.capitalize()
    if use_fmtc:
        l = core.fmtc.resample(clip, w, h, kernel=kernel, **resample_args)
        l2 = core.fmtc.resample(core.std.Expr([clip], ['x dup *']), w, h, kernel=kernel, **resample_args)
    else: # use vszimg
        l = eval(f'core.resize.{kernel}')(clip, w, h, **resample_args)
        l2 = eval(f'core.resize.{kernel}')(core.std.Expr([clip], ["x dup *"]), w, h, **resample_args)

    m = Filter(l)
    sl_plus_m_square = Filter(core.std.Expr([l], ['x dup *']))
    sh_plus_m_square = Filter(l2)
    m_square = core.std.Expr([m], ['x dup *'])
    r = core.std.Expr([sl_plus_m_square, sh_plus_m_square, m_square], ['x z - {eps} < 0 y z - x z - / sqrt ?'.format(eps=epsilon)])
    t = Filter(core.std.Expr([r, m], ['x y *']))
    m = Filter(m)
    r = Filter(r)
    d = core.std.Expr([m, r, l, t], ['x y z * + a -'])

    if gamma:
        d = nnrs.LinearToGamma(mvf.Depth(d, 16), fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    return d


def LocalStatisticsMatching(
    src: vs.VideoNode, ref: vs.VideoNode, radius: Union[int, VSFuncType] = 1,
    return_all: bool = False, epsilon: float = 1e-4, **depth_args: Any
) -> Union[vs.VideoNode, Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode]]:

    """Local statistics matcher

    Match the local statistics (mean, variance) of "src" with "ref".

    The idea is similar to "adaptive instance normalization" in deep learning literature.

    All the internal calculations are done at 32-bit float.

    Args:
        src, ref: Inputs.

        radius: (int or function) If it is an integer, it specifies the radius of mean filter.
            It can also be a custom function.
            Default is 1.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

        epsilon: (float) Small positive number to avoid dividing by 0.
            Default is 1e-4.
    """

    funcName = 'LocalStatisticsMatching'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if not isinstance(ref, vs.VideoNode):
        raise TypeError(funcName + ': \"ref\" must be a clip!')

    bits = src.format.bits_per_sample
    sampleType = src.format.sample_type

    src, src_mean, src_var = LocalStatistics(src, radius=radius, **depth_args)
    _, ref_mean, ref_var = LocalStatistics(ref, radius=radius, **depth_args)

    flt = core.std.Expr([src, src_mean, src_var, ref_mean, ref_var], ['x y - z sqrt {} + / b sqrt * a +'.format(epsilon)])

    flt = mvf.Depth(flt, depth=bits, sample=sampleType, **depth_args)

    if return_all:
        return flt, src_mean, src_var, ref_mean, ref_var
    else:
        return flt


def LocalStatistics(clip: vs.VideoNode, radius: Union[int, VSFuncType] = 1,
                    **depth_args: Any
                    ) -> Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
    """Local statistics calculator

    The local mean and variance will be returned.

    All the internal calculations are done at 32-bit float.

    Args:
        clip: Inputs.

        radius: (int or function) If it is an integer, it specifies the radius of mean filter.
            It can also be a custom function.
            Default is 1.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    Returns:
        A list containing three clips (source, mean, variance) in 32bit float.

    """

    funcName = 'LocalStatistics'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    Expectation = radius if callable(radius) else functools.partial(BoxFilter, radius=radius+1)

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    mean = Expectation(clip)
    squared = Expectation(core.std.Expr(clip, 'x dup *'))
    var = core.std.Expr([squared, mean], 'x y dup * -')

    return clip, mean, var


def TextSub16(src: vs.VideoNode, file: str, mod: bool = False, tv_range: bool = True,
              matrix: Optional[Union[int, str]] = None, dither: Optional[str] = None,
              **vsfilter_args: Any
              ) -> vs.VideoNode:
    """TextSub16 for VapourSynth

    Author: mawen1250 (http://nmm.me/109)

    Unofficial description:
        Generate mask in YUV and use it to mask high-precision subtitles overlayed in RGB.

    Args:
        src: Input clip, must be of YUV color family.

        file: Path to subtitle.

        mod: (bool) Whether to use VSFilterMod. If not, VSFilter will be used.
            Default is False.

        tv_range: (bool) Define if input clip is of tv range(limited range).
            Default is True.

        matrix: (int|str) Color matrix of input clip.
            Default is None, guessed according to the color family and size of input clip.

        dither: (str) Dithering method of vszimg.
            The following dithering methods are available: "none", "ordered", "random", "error_diffusion".
            Default is "error_diffusion".

        vsfilter_args: (dict) Additional arguments passed to subtitle plugin.
            Default is {}.

    Requirments:
        1. VSFilter (https://github.com/HomeOfVapourSynthEvolution/VSFilter)
        2. VSFilterMod (https://github.com/sorayuki/VSFilterMod)

    """

    funcName = 'TextSub16'

    if not isinstance(src, vs.VideoNode) or src.format.color_family != vs.YUV:
        raise TypeError(funcName + ': \"src\" must be a YUV clip!')

    matrix = mvf.GetMatrix(src, matrix, True)
    css = src.format.name[3:6]
    sw = src.width
    sh = src.height

    if dither is None:
        dither = 'error_diffusion'

    if src.format.id != vs.YUV420P8:
        src8 = core.resize.Bicubic(src, format=vs.YUV420P8, range_in=tv_range)
    else:
        src8 = src

    src16 = mvf.Depth(src, depth=16, sample=vs.INTEGER, fulls=tv_range, dither='none')

    if mod:
        src8sub = core.vsfm.TextSubMod(src8, file=file, **vsfilter_args)
    else:
        src8sub = core.vsf.TextSub(src8, file=file, **vsfilter_args)

    submask = core.std.Expr([src8, src8sub], ['x y = 0 255 ?']).resize.Bilinear(format=vs.YUV444P8, range=True, range_in=True)
    submaskY = mvf.GetPlane(submask, 0)
    submaskU = mvf.GetPlane(submask, 1)
    submaskV = mvf.GetPlane(submask, 2)
    submask = mvf.Max(mvf.Max(submaskY, submaskU), submaskV).std.Inflate()
    submaskY = core.resize.Bilinear(submask, format=vs.GRAY16, range_in=True, range=True)
    if css == '444':
        submaskC = submaskY
    elif css == '422':
        submaskC = core.resize.Bilinear(submask, sw // 2, sh, format=vs.GRAY16, range_in=True, range=True, src_left=-0.5)
    elif css == '420':
        submaskC = core.resize.Bilinear(submask, sw // 2, sh // 2, format=vs.GRAY16, range_in=True, range=True, src_left=-0.5)
    else:
        raise TypeError(funcName + 'the subsampling of \"src\" must be 444/422/420!')

    submask = core.std.ShufflePlanes([submaskY, submaskC], [0, 0, 0], vs.YUV)

    last = core.resize.Bicubic(src16, format=vs.RGB24, matrix_in_s=matrix, range=tv_range, dither_type=dither) # type: ignore

    if mod:
        last = core.vsfm.TextSubMod(last, file=file, **vsfilter_args)
    else:
        last = core.vsf.TextSub(last, file=file, **vsfilter_args)

    sub16 = core.resize.Bicubic(last, format=src16.format.id, matrix_s=matrix, range=tv_range, dither_type=dither) # type: ignore

    return core.std.MaskedMerge(src16, sub16, submask, planes=[0, 1, 2])


def TMinBlur(clip: vs.VideoNode, r: int = 1, thr: float = 2) -> vs.VideoNode:
    """Thresholded MinBlur

    Use another MinBlur with larger radius to guide the smoothing effect of current MinBlur.

    For detailed motivation and description (in Chinese), see:
    https://gist.github.com/WolframRhodium/1e3ae9276d70aa1ddc93ea833cdce9c6#file-05-minblurmod-md

    Args:
        clip: Input clip.

        r: (int) Radius of MinBlur() filter.
            Default is 1.

        thr: (float) Threshold in 8 bits scale.
            If it is larger than 255, the output will be identical to original MinBlur().
            Default is 2.

    """

    funcName = 'TMinBlur'

    if not isinstance(clip, vs.VideoNode) or clip.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': \"clip\" must be an integer clip!')

    thr = scale(thr, clip.format.bits_per_sample)

    pre1 = haf_MinBlur(clip, r=r)
    pre2 = haf_MinBlur(clip, r=r+1)

    return core.std.Expr([clip, pre1, pre2], ['y z - abs {thr} <= y x ?'.format(thr=thr)])


def mdering(clip: vs.VideoNode, thr: float = 2) -> vs.VideoNode:
    """A simple light and bright DCT ringing remover

    It is a special instance of TMinBlur (r=1 and only filter the bright part) for higher performance.
    Post-processing is needed to reduce degradation of flat and texture areas.

    Args:
        clip: Input clip.

        thr: (float) Threshold in 8 bits scale.
            Default is 2.

    """

    funcName = 'mdering'

    if not isinstance(clip, vs.VideoNode) or clip.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': \"clip\" must be an integer clip!')

    bits = clip.format.bits_per_sample
    thr = scale(thr, bits)

    rg11_1 = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    rg11_2 = core.std.Convolution(rg11_1, [1]*9)
    rg4_1 = core.std.Median(clip)

    if bits <= 12:
        rg4_2 = core.ctmf.CTMF(clip, radius=2)
    else:
        rg4_2 = core.fmtc.bitdepth(clip, bits=12, dmode=1).ctmf.CTMF(radius=2).fmtc.bitdepth(bits=bits)
        rg4_2 = mvf.LimitFilter(clip, rg4_2, thr=0.0625, elast=2)

    minblur_1 = core.std.Expr([clip, rg11_1, rg4_1], ['x y - x z - xor x x y - abs x z - abs < y z ? ?'])
    minblur_2 = core.std.Expr([clip, rg11_2, rg4_2], ['x y - x z - xor x x y - abs x z - abs < y z ? ?'])
    dering = core.std.Expr([clip, minblur_1, minblur_2], ['y z - abs {thr} <= y x <= and y x ?'.format(thr=thr)])

    return dering


def BMAFilter(clip: vs.VideoNode, guidance: Optional[vs.VideoNode] = None, radius: int = 1,
              lamda: float = 1e-2, epsilon: float = 1e-5, mode: int = 3,
              **depth_args: Any
              ) -> vs.VideoNode:
    """Edge-Aware BMA Filter

    Edge-aware BMA filter is a family of edge-aware filters proposed based on optimal parameter estimation and Bayesian model averaging (BMA).
    The problem of filtering a pixel in a local pixel patch is formulated as an optimal estimation problem,
    and multiple estimates of the same pixel are combined using BMA.
    Filters in this family differs from different settings of cost functions and log-likelihood and log-prior functions.

    However, only four of six BMA filters are implemented.
    The implementation is modified to allow the filtering to be guided by another source, like GuidedFilter().

    Most of the internal calculations are done at 32-bit float, except median filtering with radius larger than 1 is done at integer.

    Args:
        clip: Input clip.

        guidance: (clip) Guidance clip used to compute the coefficient of the translation on "clip".
            It must has the same clip properties as 'clip'.
            If it is None, it will be set to 'clip', with duplicate calculations being omitted.
            Default is None.

        radius: (int) The radius of box filter and median filter.
            Default is 1.

        lamda: (float) A criterion for judging whether a patch has high variance and should be preserved, or is flat and should be smoothed.
            It only takes effects when `mode` is 3 or 4.
            The limit of filter of `mode` 3 [resp. 4] as `lamda` approaches infinity is filter of `mode` 1 [resp. 2].
            Default is 0.01.

        epsilon: (float) Small number to avoid divide by 0.
            Default is 0.00001.

        mode: (1~4): Number of different BMA filters.
            1: l2-norm based cost function, constant prior and gaussian likelihood.
            2: l1-norm based cost function, constant prior and laplacian likelihood.
            3: 'hit-or-miss' cost function, gaussian prior and gaussian likelihood.
            4: 'hit-or-miss' cost function, gaussian prior and laplacian likelihood.
            Default is 3.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    Ref:
        [1] Deng, G. (2016). Edge-aware BMA filters. IEEE Transactions on Image Processing, 25(1), 439-454.
        [2] https://www.researchgate.net/publication/284391731_Edge-aware_BMA_filters

    """

    funcName = 'BMAFilter'

    if guidance is None:
        guidance = clip
    else:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if clip.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"clip\"!')
        if clip.width != guidance.width or clip.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"clip\"!')

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type
    clip_src = clip
    clip = mvf.Depth(clip, depth=32, **depth_args)
    guidance = mvf.Depth(guidance, depth=32, **depth_args) if guidance != clip_src else clip

    if mode in (2, 4):
        def Filter(clip: vs.VideoNode) -> vs.VideoNode:
            if radius == 1:
                clip = core.std.Median(clip)
            else:
                clip = mvf.Depth(clip, 12, **depth_args)
                clip = core.ctmf.CTMF(clip, radius=radius)
            return mvf.Depth(clip, 32, **depth_args)
    elif mode in (1, 3):
        Filter = functools.partial(BoxFilter, radius=radius+1)

    Expectation = functools.partial(BoxFilter, radius=radius+1)

    if mode in (1, 2):
        mean_guidance = Expectation(guidance)
        corr_guidance = Expectation(core.std.Expr([guidance], ['x dup *']))
        unscaled_alpha = core.std.Expr([corr_guidance, mean_guidance], ['1 x y dup * - {epsilon} + /'.format(epsilon=epsilon)]) # Eqn. 10
        alpha_scale = Expectation(unscaled_alpha)

        if mode == 1:
            mean_clip = Filter(clip) if clip != guidance else mean_guidance
            res = Expectation(core.std.Expr([unscaled_alpha, mean_clip], ['x y *'])) # Eqn. 11
        else: # mode == 2
            median_clip = Filter(clip_src)
            res = Expectation(core.std.Expr([unscaled_alpha, median_clip], ['x y *'])) # Eqn. 12

        res = core.std.Expr([res, alpha_scale], ['x y /'])
    elif mode in (3, 4):
        mean_guidance = Expectation(guidance)

        guidance_square = core.std.Expr([guidance], ['x dup *'])
        var_guidance = core.std.Expr([Expectation(guidance_square), mean_guidance], ['x y dup * -'])
        unscaled_alpha = core.std.Expr([var_guidance], ['1 x {epsilon} + /'.format(epsilon=epsilon)]) # Eqn. 10
        alpha_scale = Expectation(unscaled_alpha)
        beta = core.std.Expr([var_guidance], ['1 x {epsilon} + {lamda} * 1 + /'.format(epsilon=epsilon, lamda=1/lamda)]) # Eqn. 18
        tmp1 = core.std.Expr([unscaled_alpha, beta], ['x y *'])

        if mode == 3:
            mean_clip = Filter(clip) if clip != guidance else mean_guidance
            tmp2 = Expectation(core.std.Expr([tmp1, mean_clip], ['x y *'])) # Eqn. 19, left
        else: # mode == 4
            median_clip = Filter(clip_src)
            tmp2 = Expectation(core.std.Expr([tmp1, median_clip], ['x y *'])) # Eqn. 25, left

        tmp3 = Expectation(tmp1) # Eqn. 19 / 25, right
        res = core.std.Expr([tmp2, alpha_scale, tmp3, clip], ['x y / 1 z y / - a * +']) # Eqn.19 / 25
    else:
        raise ValueError(funcName + '\"mode\" must be in [1, 2, 3, 4]!')

    return mvf.Depth(res, depth=bits, sample=sampleType, **depth_args)


def LLSURE(clip: vs.VideoNode, guidance: Optional[vs.VideoNode] = None, radius: int = 2,
           sigma: Union[float, vs.VideoNode] = 0, epsilon: float = 1e-5,
           **depth_args: Any
           ) -> vs.VideoNode:
    """Local Linear SURE-Based Edge-Preserving Image Filtering

    LLSURE is based on a local linear model and using the principle of Stein’s unbiased risk estimate (SURE)
    as an estimator for the mean squared error from the noisy image.
    Multiple estimates are aggregated using Variance-based Weighted Average (WAV).

    Most of the internal calculations are done at 32-bit float, except median filtering with radius larger than 1 is done at integer.

    Args:
        clip: Input clip.

        guidance: (clip) Guidance clip used to compute the coefficient of the translation on "clip".
            It must has the same clip properties as 'clip'.
            If it is None, it will be set to 'clip', with duplicate calculations being omitted.
            It is not recommended to use such feature since there might be severe numerical precision problem in this implementation.
            Default is None.

        radius: (int) The radius of box filter and median filter.
            Default is 2.

        sigma: (float or clip) Estimation of noise variance.
            If it is 0, it is automatically calculated using MAD (median absolute deviation).
            If it is smaller than 0, the result is MAD multiplied by the absolute value of "sigma".
            Default is 0.

        epsilon: (float) Small number to avoid divide by 0.
            Default is 0.00001.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    Ref:
        [1] Qiu, T., Wang, A., Yu, N., & Song, A. (2013). LLSURE: local linear SURE-based edge-preserving image filtering.
            IEEE Transactions on Image Processing, 22(1), 80-90.

    """

    funcName = 'LLSURE'

    if guidance is None:
        guidance = clip
    else:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if clip.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"clip\"!')
        if clip.width != guidance.width or clip.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"clip\"!')

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    Expectation = functools.partial(BoxFilter, radius=radius+1)

    clip_src = clip
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)
    guidance = mvf.Depth(guidance, depth=32, **depth_args) if guidance != clip_src else clip

    mean_guidance = Expectation(guidance)
    guidance_square = core.std.Expr([guidance], ['x dup *'])
    var_guidance = core.std.Expr([Expectation(guidance_square), mean_guidance], ['x y dup * -'])
    inv_var = core.std.Expr([var_guidance], ['1 x {epsilon} + /'.format(epsilon=epsilon)])
    normalized_w = Expectation(inv_var)

    if not isinstance(sigma, vs.VideoNode):
        if sigma <= 0:
            absolute_deviation = core.std.Expr([guidance, mean_guidance], ['x y - abs'])

            if radius == 1:
                sigma_tmp = core.std.Median(absolute_deviation)
            else:
                absolute_deviation = core.fmtc.bitdepth(absolute_deviation, bits=12, dmode=1, fulls=True, fulld=True)
                sigma_tmp = core.ctmf.CTMF(absolute_deviation, radius=radius)
                sigma_tmp = mvf.Depth(sigma_tmp, depth=32, sample=vs.FLOAT, fulls=True, fulld=True)

            sigma = sigma_tmp if sigma == 0 else core.std.Expr([sigma_tmp], ['x {sigma} *'.format(sigma=-sigma)])
        else:
            sigma = core.std.BlankClip(clip, color=[sigma**2] * clip.format.num_planes)

    if guidance == clip:
        a_star = core.std.Expr([var_guidance, sigma, inv_var], ['x y - 0 max z *']) # Eqn. 10 (a)
        b_star = core.std.Expr([a_star, mean_guidance], ['1 x - y *']) # Eqn. 10 (b)
    else: # Joint LLSURE
        mean_clip = Expectation(clip)
        corr_clip_guidance = Expectation(core.std.Expr([clip, guidance], ['x y *']))
        cov_clip_guidance = core.std.Expr([corr_clip_guidance, mean_clip, mean_guidance], ['x y z * -'])
        a_star = core.std.Expr([cov_clip_guidance, sigma, inv_var], ['x 0 > 1 -1 ? x abs y - 0 max * z *']) # Eqn. 20 (a)
        b_star = core.std.Expr([mean_clip, cov_clip_guidance, sigma, inv_var, mean_guidance], ['x y z - a * b * -']) # Eqn. 20 (b)

    bar_a = Expectation(core.std.Expr([a_star, inv_var], ['x y *']))
    bar_b = Expectation(core.std.Expr([b_star, inv_var], ['x y *']))
    res = core.std.Expr([bar_a, guidance, bar_b, normalized_w], ['x y * z + a {epsilon} + /'.format(epsilon=epsilon)]) # Eqn. 17 / 21

    return mvf.Depth(res, depth=bits, sample=sampleType, **depth_args)


def YAHRmod(clp: vs.VideoNode, blur: int = 2, depth: int = 32, **limit_filter_args: Any) -> vs.VideoNode:
    """Modification of YAHR with better texture preserving property

    The YAHR() is a simple and powerful script to reduce halos from over enhanced edges.
    It simply creates two versions of ringing-free result and uses the difference of the source-deringed
    pairs to restore the texture.
    However, it still suffers from texture degradation due to the unconstrained use of MinBlur() in texture area.
    Inspired by the observation that the Repair(13) used in YAHR() has the characteristics of preserving the
    source signal if it is closed to the reference in the same location, i.e. the source signal will be output
    if the two filtered results are closed, we simply add an LimitFilter() before the repair procedure to utilize
    this property to preserve the texture.

    Experiment can denmonstrate its better texture preserving performance over the original version.

    The source code is modified from
    havsfunc(https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/2048fcb320ef8121c842d087191708d61f39416b/havsfunc.py#L644-L671).

    Args:
        clp: Input clip.

        blur: (int) "blur" parameter of AWarpSharp2.
            Default is 2.

        depth: (int) "depth" parameter of AWarpSharp2.
            Default is 32.

        limit_filter_args: (dict) Additional arguments passed to mvf.LimitFilter in the form of keyword arguments.

    """

    funcName = 'YAHRmod'

    if not isinstance(clp, vs.VideoNode):
        raise TypeError(funcName + ': \"clp\" must be a clip!')

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp # type: Optional[vs.VideoNode]
        clp = mvf.GetPlane(clp, 0)
    else:
        clp_orig = None

    b1 = core.std.Convolution(haf_MinBlur(clp, 2), matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    b1D = core.std.MakeDiff(clp, b1)
    w1 = haf_Padding(clp, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)
    w1b1 = core.std.Convolution(haf_MinBlur(w1, 2), matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    w1b1D = core.std.MakeDiff(w1, w1b1)
    w1b1D = mvf.LimitFilter(b1D, w1b1D, **limit_filter_args) # The only modification
    DD = core.rgvs.Repair(b1D, w1b1D, 13)
    DD2 = core.std.MakeDiff(b1D, DD)
    last = core.std.MakeDiff(clp, DD2)

    """
    it's also possible to place the LimitFilter() here, e.g.

    last = mvf.LimitFilter(clp, last, **limit_filter_args)

    To achieve a similar amount of filtering, one should decrease "thr" in the later case.

    The difference between the two is usually marginal, and the later case looks more versatile.
    However, it seems to me that the former looks better in most cases.
    """

    if clp_orig is not None:
        return core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    else:
        return last


def RandomInterleave(clips: Sequence[vs.VideoNode], seed: Optional[int] = None,
                     rand_list: Optional[Sequence[int]] = None) -> vs.VideoNode:
    """Returns a clip with the frames from all clips randomly interleaved

    Useful for blinded-experiment.

    Args:
        clips: Input clips with same formats.

        seed: (int) Random number generator initializer.
            Default is None.

        rand_list: (list) A list containing frame mappings of the interleaved clip.
            For example, [0, 0, 1] stats that the first two frames of the output clip
                are obtained from the first clip in "clips", while the third frame is
                obtained from the second clip in "clips".
            Default is None.

    """

    funcName = 'RandomInterleave'

    if not isinstance(clips, abc.Sequence):
        raise TypeError(funcName + ': \"clips\" must be a list of clips!')
    else:
        clips = list(clips)

    length = len(clips)
    if length == 1:
        return clips[0]

    if rand_list is None:
        import random
        random.seed(seed)

        tmp = list(range(length))

        rand_list = []

        for i in range(clips[0].num_frames):
            random.shuffle(tmp)
            rand_list += tmp

    frames = [i for i in range(clips[0].num_frames) for j in range(length - 1)]
    for i in range(length):
        clips[i] = core.std.DuplicateFrames(clips[i], frames=frames)

    def selector(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        return f[rand_list[n]] # type: ignore

    clip = core.std.ModifyFrame(clips[0], clips=clips, selector=selector)

    return core.std.AssumeFPS(clip, fpsnum=clips[0].fps.numerator * length, fpsden=clips[0].fps.denominator)


def super_resolution(clip: vs.VideoNode, model_filename: str, epoch: int = 0, up_scale: int = 2,
                     block_w: int = 128, block_h: Optional[int] = None, is_rgb_model: bool = True,
                     pad: Optional[Tuple[int, int, int, int]] = None,
                     crop: Optional[Tuple[int, int, int, int]] = None,
                     pre_upscale: bool = False, upscale_uv: bool = False, merge_source: bool = False,
                     use_fmtc: bool = False, resample_kernel: Optional[str] = None,
                     resample_args: Optional[Dict[str, Any]] = None, pad_mode: Optional[str] = None,
                     framework: Optional[str] = None, data_format: Optional[str] = None,
                     device_id: Union[int, Sequence[int]] = 0) -> vs.VideoNode:
    '''Use MXNet to accelerate Image-Processing in VapourSynth using C++ interface

    Drop-in replacement of muvsfunc_numpy's counterpart using core.mx.Predict().
    The plugin can be downloaded from https://github.com/kice/vs_mxnet

    The results from two versions of the functinos may not identical when the size of block is smaller than the frame
        or padding is used, due to different implementation.

    Currently only MXNet backend is supported. Multi-GPU data parallelism is supported.

    The color space and bit depth of the output depends on the super resolution algorithm.
    Currently only RGB and GRAY models are supported.

    All the internal calculations are done at 32-bit float.

    Demo:
        https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/super_resolution_mxnet.vpy

    Args:
        clip: Input clip.
            The color space will be automatically converted by mvf.ToRGB/YUV if it is not
            compatiable with the super resolution algorithm.

        model_filename: Path to the pre-trained model.
            This specifies the path prefix of saved model files.
            You should have "model_filename-symbol.json", "model_filename-xxxx.params", where xxxx is the epoch number.

        epoch: (int) Epoch to load of MXNet model file.
            Default is 0.

        up_scale: (int) Upscaling factor.
            Should be compatiable with the model.
            Default is 2.

        block_w, block_h: (int) The horizontal/vertical block size for dividing the image during processing.
            The optimal value may vary according to different graphics card and image size.
            Default is 128.

        is_rgb_model: (bool) Whether the model is RGB model.
            If not, it is assumed to be Y model, and RGB input will be converted to YUV before feeding to the network
            Default is True.

        pad: (tuple of four ints) Patch-wise padding before upscaling.
            The four values indicate padding at top, bottom, left, right of each patch respectively.
            Default is None.

        crop: (tuple of four ints) Patch-wise cropping after upscaling.
            The four values indicate cropping at top, bottom, left, right of each patch respectively.
            Moreover, due to the implementation of vs_mxnet, the values at top and left should be zero.
            Default is None.

        pre_upscale: (bool) Whether to upscale the image before feed to the network.
            If true, currently the function will only upscale the whole image directly rather than upscale
                the patches separately, which may results in blocking artifacts on some algorithms.
            Default is False.

        upscale_uv: (bool) Whether to upscale UV channels when using Y model.
            If not, the UV channels will be discarded.
            Only works when "is_rgb_model" is False.
            Default is False.

        merge_source: (bool) Whether to merge the output of the network to the (nearest/bilinear/bicubic) enlarged source image.
            Default is False.

        use_fmtc: (bool) Whether to use fmtconv for enlarging. If not, vszimg (core.resize.*) will be used.
            Only works when "pre_upscale" is True.
            Default is False.

        resample_kernel: (str) Resample kernel.
            If can be 'Catmull-Rom', i.e. BicubicResize with b=0 and c=0.5.
            Only works when "pre_upscale" is True.
            Default is 'Catmull-Rom'.

        resample_args: (dict) Additional arguments passed to vszimg/fmtconv resample kernel.
            Only works when "pre_upscale" is True.
            Default is {}.

        pad_mode: (str) Padding type to use.
            If set to "source", the pixels in the source image will be used.
            Only "source" is supported. Please switch to muvsfunc_numpy's implementation for other modes.
            Default is "source"

        framework: INVALID. Please switch to muvsfunc_numpy's implementation.

        data_format: INVALID. Please switch to muvsfunc_numpy's implementation.

        device_id: (int or list of ints) Which device(s) to use.
            Starting with 0. If it is smaller than 0, CPU will be used.
            It can be a list of integers, indicating devices for multi-GPU data parallelism.
            Default is 0.

    '''

    funcName = 'super_resolution'

    isGray = clip.format.color_family == vs.GRAY
    isRGB = clip.format.color_family == vs.RGB

    symbol_filename = model_filename + '-symbol.json'
    param_filename = model_filename + '-{:04d}.params'.format(epoch)

    if block_h is None:
        block_h = block_w

    if pad is None:
        pad = (0, 0, 0, 0)

    if crop is None:
        crop = (0, 0, 0, 0)
    else:
        if crop[0] != 0 or crop[2] != 0:
            raise ValueError(funcName + ': Cropping at left or top should be zero! Please switch to muvsfunc_numpy\'s implementation.')

    if resample_kernel is None:
        resample_kernel = 'Bicubic'

    if resample_args is None:
        resample_args = {}

    if pad_mode is not None and pad_mode.lower() != 'source':
        raise ValueError(funcName + ': Only source padding mode is supported! Please switch to muvsfunc_numpy\'s implementation.')

    if framework is None:
        framework = 'MXNet'
    framework = framework.lower()
    if framework.lower() != 'mxnet':
        raise ValueError(funcName + ': Only MXNet framework is supported! Please switch to muvsfunc_numpy\'s implementation.')
    else:
        import mxnet as mx

        if not hasattr(core, 'mx'):
            core.std.LoadPlugin(r'vs_mxnet.dll', altsearchpath=True)

    if data_format is None:
        data_format = 'NCHW'
    else:
        data_format = data_format.upper()
        if data_format != 'NCHW':
            raise ValueError(funcName + ': Only NCHW data format is supported! Please switch to muvsfunc_numpy\'s implementation.')

    if isinstance(device_id, int):
        device_id = [device_id]

    # color space conversion
    if is_rgb_model and not isRGB:
        clip = mvf.ToRGB(clip, depth=32)

    elif not is_rgb_model:
        if isRGB:
            clip = mvf.ToYUV(clip, depth=32)

        if not isGray:
            if not upscale_uv: # isYUV/RGB and only upscale Y
                clip = mvf.GetPlane(clip)
            else:
                clip = core.std.Expr([clip], ['', 'x 0.5 +']) # change the range of UV from [-0.5, 0.5] to [0, 1]

    # bit depth conversion
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT)

    # pre-upscaling
    if pre_upscale:
        if up_scale != 1:
            if use_fmtc:
                if resample_kernel.lower() == 'catmull-rom':
                    clip = core.fmtc.resample(clip, clip.width*up_scale, clip.height*up_scale, kernel='bicubic', a1=0, a2=0.5, **resample_args)
                else:
                    clip = core.fmtc.resample(clip, clip.width*up_scale, clip.height*up_scale, kernel=resample_kernel, **resample_args)
            else: # use vszimg
                if resample_kernel.lower() == 'catmull-rom':
                    clip = core.resize.Bicubic(clip, clip.width*up_scale, clip.height*up_scale, filter_param_a=0, filter_param_b=0.5, **resample_args)
                else:
                    kernel = resample_kernel.capitalize()
                    clip = eval(f'core.resize.{kernel}')(clip, clip.width*up_scale, clip.height*up_scale, **resample_args)

            up_scale = 1

    # inference
    def inference(clip: vs.VideoNode, dev_id: int) -> vs.VideoNode:
        '''wrapper function for inference'''

        if is_rgb_model or not upscale_uv:
            w, h = clip.width, clip.height

            if (pad[0]-crop[0]//up_scale > 0 or pad[1]-crop[1]//up_scale > 0 or # type: ignore
                pad[2]-crop[2]//up_scale > 0 or pad[3]-crop[3]//up_scale > 0): # type: ignore

                clip = haf_Padding(clip, pad[2]-crop[2]//up_scale, pad[3]-crop[3]//up_scale, # type: ignore
                    pad[0]-crop[0]//up_scale, pad[1]-crop[1]//up_scale) # type: ignore

            super_res = core.mx.Predict(clip, symbol=symbol_filename, param=param_filename,
                patch_w=block_w+pad[2]+pad[3], patch_h=block_h+pad[0]+pad[1], scale=up_scale, # type: ignore
                output_w=block_w*up_scale+crop[2]+crop[3], output_h=block_h*up_scale+crop[0]+crop[1], # type: ignore # crop[0] == crop[2] == 0
                frame_w=w*up_scale, frame_h=h*up_scale, step_w=block_w, step_h=block_h,
                outstep_w=block_w*up_scale, outstep_h=block_h*up_scale, # type: ignore
                ctx=2 if dev_id >= 0 else 1, dev_id=max(dev_id, 0))

        else: # Y model, YUV input that may have subsampling, need to upscale uv
            num_planes = clip.format.num_planes
            yuv_list = [mvf.GetPlane(clip, i) for i in range(num_planes)]

            for i in range(num_planes):
                w, h = yuv_list[i].width, yuv_list[i].height

                if (pad[0]-crop[0]//up_scale > 0 or pad[1]-crop[1]//up_scale > 0 or # type: ignore
                    pad[2]-crop[2]//up_scale > 0 or pad[3]-crop[3]//up_scale > 0): # type: ignore

                    yuv_list[i] = haf_Padding(yuv_list[i], pad[2]-crop[2]//up_scale, pad[3]-crop[3]//up_scale, # type: ignore
                        pad[0]-crop[0]//up_scale, pad[1]-crop[1]//up_scale) # type: ignore

                yuv_list[i] = core.mx.Predict(yuv_list[i], symbol=symbol_filename, param=param_filename,
                    patch_w=block_w+pad[2]+pad[3], patch_h=block_h+pad[0]+pad[1], scale=up_scale, # type: ignore
                    output_w=block_w*up_scale+crop[2]+crop[3], output_h=block_h*up_scale+crop[0]+crop[1], # type: ignore # crop[0] == crop[2] == 0
                    frame_w=w*up_scale, frame_h=h*up_scale, step_w=block_w, step_h=block_h,
                    outstep_w=block_w*up_scale, outstep_h=block_h*up_scale, # type: ignore
                    ctx=2 if dev_id >= 0 else 1, dev_id=max(dev_id, 0))

            super_res = core.std.ShufflePlanes(yuv_list, [0] * num_planes, clip.format.color_family)

        return super_res

    if len(device_id) == 1:
        super_res = inference(clip, device_id[0])
    else: # multi-GPU data parallelism
        workers = len(device_id)
        super_res_list = [inference(clip[i::workers], device_id[i]) for i in range(workers)]
        super_res = core.std.Interleave(super_res_list)

    # post-processing
    if not is_rgb_model and not isGray and upscale_uv:
        super_res = core.std.Expr([super_res], ['', 'x 0.5 -']) # restore the range of UV

    if merge_source:
        if up_scale != 1:
            if use_fmtc:
                if resample_kernel.lower() == 'catmull-rom':
                    low_res = core.fmtc.resample(clip, super_res.width, super_res.height, kernel='bicubic', a1=0, a2=0.5, **resample_args)
                else:
                    low_res = core.fmtc.resample(clip, super_res.width, super_res.height, kernel=resample_kernel, **resample_args)
            else: # use vszimg
                if resample_kernel.lower() == 'catmull-rom':
                    low_res = core.resize.Bicubic(clip, super_res.width, super_res.height, filter_param_a=0, filter_param_b=0.5, **resample_args)
                else:
                    kernel = resample_kernel.capitalize()
                    low_res = eval(f'core.resize.{kernel}')(clip, super_res.width, super_res.height, **resample_args)
        else:
            low_res = clip

        super_res = core.std.Expr([super_res, low_res], ['x y +'])

    return super_res


def MDSI(clip1: vs.VideoNode, clip2: vs.VideoNode, down_scale: int = 1, alpha: float = 0.6,
         show_maps: bool = False
         ) -> Union[vs.VideoNode, Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode]]:
    """Mean Deviation Similarity Index Calculator

    MDSI is a full reference IQA model that utilize gradient similarity (GS), chromaticity similarity (CS), and deviation pooling (DP).

    The lowerer the MDSI score, the higher the image perceptual quality.
    Larger MDSI values indicate to the more severe distorted images, while an image with perfect quality is assessed by a quality score of zero.

    The distortion degree of the distorted image will be stored as frame property 'FrameMDSI' in the output clip.

    Note that bilinear downsampling is used in this implementation (but disabled by default), as opposed to the original paper.
    The gradient-chromaticity similarity map is saturated before deviation pooling, as described in II.D.
    Matrix used by rgb2gray() from MATLAB (similar to BT.601 matrix) is used for computation of luma component.

    Args:
        clip1: The first clip to be evaluated, will be copied to output.

        clip2: The second clip, to be compared with the first one.

        down_scale: (int) Factor of downsampling before quality assessment.
            Default is 1.

        alpha: (float, 0~1) Weight used to merge gradient similarity (GS) map and chromaticity similarity (CS) map.
            Default is 0.6.

        show_maps: (bool) Whether to return gradient similarity (GS), chromaticity similarity (CS) and GCS (linear combination of CS and GCS) maps in GRAYS.
            If it is true, a tuple containing (clip1, gs, cs, gcs) clips will be returned.
            Example:
                src, gs, cs, gcs = MDSI(clip1=src, clip2=ref, show_maps=True)

            If not, only "clip1" will be returned.
            Default is False.

    Ref:
        [1] Nafchi, H. Z., Shahkolaei, A., Hedjam, R., & Cheriet, M. (2016).
            Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
            IEEE Access, 4, 5579-5590.
        [2] https://ww2.mathworks.cn/matlabcentral/fileexchange/59809-mdsi-ref-dist-combmethod
    """

    funcName = 'MDSI'

    if not isinstance(clip1, vs.VideoNode) or clip1.format.color_family != vs.RGB:
        raise TypeError(funcName + ': \"clip1\" must be an RGB clip!')
    if not isinstance(clip2, vs.VideoNode) or clip2.format.color_family != vs.RGB:
        raise TypeError(funcName + ': \"clip2\" must be an RGB clip!')

    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    c1 = 140 / (255 ** 2)
    c2 = 55 / (255 ** 2)
    c3 = 550 / (255 ** 2)

    if down_scale > 1 or clip1.format.sample_type != vs.FLOAT:
        down1 = core.resize.Bilinear(clip1, clip1.width // down_scale, clip1.height // down_scale, format=vs.RGBS)
    else:
        down1 = clip1

    if down_scale > 1 or clip2.format.sample_type != vs.FLOAT:
        down2 = core.resize.Bilinear(clip2, clip2.width // down_scale, clip2.height // down_scale, format=vs.RGBS)
    else:
        down2 = clip2

    r1, g1, b1 = [mvf.GetPlane(down1, i) for i in range(3)]
    r2, g2, b2 = [mvf.GetPlane(down2, i) for i in range(3)]

    # luminance
    l1 = core.std.Expr([r1, g1, b1], ['x 0.2989 * y 0.5870 * + z 0.1140 * +'])
    l2 = core.std.Expr([r2, g2, b2], ['x 0.2989 * y 0.5870 * + z 0.1140 * +'])
    f = core.std.Merge(l1, l2, 0.5) # fusion

    # gradient magnitudes
    ix_l1 = core.std.Convolution(l1, [1, 0, -1, 1, 0, -1, 1, 0, -1])
    iy_l1 = core.std.Convolution(l1, [1, 1, 1, 0, 0, 0, -1, -1, -1])
    g_r = core.std.Expr([ix_l1, iy_l1], ['x dup * y dup * + sqrt'])

    ix_l2 = core.std.Convolution(l2, [1, 0, -1, 1, 0, -1, 1, 0, -1])
    iy_l2 = core.std.Convolution(l2, [1, 1, 1, 0, 0, 0, -1, -1, -1])
    g_d = core.std.Expr([ix_l2, iy_l2], ['x dup * y dup * + sqrt'])

    ix_f = core.std.Convolution(f, [1, 0, -1, 1, 0, -1, 1, 0, -1])
    iy_f = core.std.Convolution(f, [1, 1, 1, 0, 0, 0, -1, -1, -1])
    g_f = core.std.Expr([ix_f, iy_f], ['x dup * y dup * + sqrt'])

    # gradient similarity
    gs12 = core.std.Expr([g_r, g_d], ['x y * 2 * {0} + x dup * y dup * + {0} + /'.format(c1)])
    gs13 = core.std.Expr([g_r, g_f], ['x y * 2 * {0} + x dup * y dup * + {0} + /'.format(c2)])
    gs23 = core.std.Expr([g_d, g_f], ['x y * 2 * {0} + x dup * y dup * + {0} + /'.format(c2)])
    gs_hvs = core.std.Expr([gs12, gs13, gs23], ['x y + z -']) # HVS-based gradient similarity

    # opponent color space
    h1 = core.std.Expr([r1, g1, b1], ['x 0.30 * y 0.04 * + z 0.35 * -'])
    h2 = core.std.Expr([r2, g2, b2], ['x 0.30 * y 0.04 * + z 0.35 * -'])
    m1 = core.std.Expr([r1, g1, b1], ['x 0.34 * y 0.60 * - z 0.17 * +'])
    m2 = core.std.Expr([r2, g2, b2], ['x 0.34 * y 0.60 * - z 0.17 * +'])

    # chromaticity similarity
    cs = core.std.Expr([h1, h2, m1, m2], ['x y * z a * + 2 * {0} + x dup * y dup * + z dup * + a dup * + {0} + /'.format(c3)])

    # gradient-chromaticity
    gcs = core.std.Expr([gs_hvs, cs], ['x {} * y {} * + 0 max 1 min 0.25 pow'.format(alpha, 1-alpha)]) # clamp to [0.0, 1.0] before deviation pooling

    # The following code is modified from mvf.PlaneStatistics()
    mean_gcs = mvf.PlaneAverage(gcs, 0, "PlaneMean")

    def _PlaneADFrame(n: int, f: vs.VideoFrame, clip: vs.VideoNode, core: vs.Core) -> vs.VideoNode:
        mean = f.props['PlaneMean']
        expr = "x {mean} - abs".format(mean=mean)
        return core.std.Expr(clip, expr)
    if _has_lexpr:
        mean = "y.PlaneMean"
        ADclip = core.akarin.Expr([gcs, mean_gcs], "x {mean} - abs".format(mean=mean))
    else:
        ADclip = core.std.FrameEval(gcs, functools.partial(_PlaneADFrame, clip=gcs, core=core), mean_gcs)
    ADclip = mvf.PlaneAverage(ADclip, 0, "PlaneMAD")

    def _FrameMDSITransfer(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['FrameMDSI'] = f[1].props['PlaneMAD'] ** 0.25 # type: ignore
        return fout
    clip1 = core.std.ModifyFrame(clip1, [clip1, ADclip], selector=_FrameMDSITransfer)

    if show_maps:
        return clip1, gs_hvs, cs, gcs
    else:
        return clip1


def MaskedLimitFilter(flt: vs.VideoNode, src: vs.VideoNode, ref: Optional[vs.VideoNode] = None,
                      thr: Union[float, vs.VideoNode] = 1.0, elast: Union[float, vs.VideoNode] = 2.0,
                      brighten_thr: Optional[Union[float, vs.VideoNode]] = None,
                      planes: PlanesType = None) -> vs.VideoNode:
    """Masked limit fIlter

    Modified from mvsfunc.LimitFilter().

    It is an extension of mvsfunc.LimitFilter(), in the sense that
    each of "thr", "elast" and "brighten_thr" can be either a value as in mvsfunc.LimitFilter(),
    or a clip which stores values in corresponding location, which enables spatial varying filtering.

    Similar to the AviSynth function Dither_limit_dif16() and HQDeringmod_limit_dif16(),
    it acts as a post-processor, and is very useful to limit the difference of filtering while avoiding artifacts.

    Args:
        flt: Filtered clip, to compute the filtering difference.
            Can be of YUV/RGB/Gray color family, can be of 8-16 bit integer or 16/32 bit float.

        src: Source clip, to apply the filtering difference.
            Must be of the same format and dimension as "flt"

        ref: (clip) Reference clip, to compute the weight to be applied on filtering difference.
            Must be of the same format and dimension as "flt".
            Default is "src".

        thr: (float or clip) Threshold (8-bit scale) to limit filtering difference.
            If it is a clip, it must be of the same color space, subsampling and dimension as "flt". Its bit-depth can be different from that of "flt".
            Default is 1.0.

        elast: (float or clip) Elasticity of the soft threshold.
            If it is a clip, it must be of the same color space, subsampling and dimension as "flt". Its bit-depth can be different from that of "flt".
            Default is 2.0.

        brighten_thr: (float or clip) Threshold (8-bit scale) for filtering difference that brightening the image (Y/R/G/B plane).
            Set a value different from "thr" is useful to limit the overshoot/undershoot/blurring introduced in sharpening/de-ringing.
            If it is a clip, it must be of the same color space, subsampling and dimension as "flt". Its bit-depth can be different from that of "flt".
            Default is the same as "thr".

        planes: (int []) Specify which planes to process.
            Unprocessed planes will be copied from the first clip "flt".
            By default, all planes will be processed.

    Example:
        "mvsfunc.LimitFilter(flt_gray8, src_gray8, thr=1.5, elast=2.0)" is equivalent to:
            "MaskedLimitFilter(flt_gray8, src_gray8, thr=1.5, elast=2.0)",
            or "MaskedLimitFilter(flt_gray8, src_gray8, thr=core.std.BlankClip(flt_gray8, color=1.5, format=vs.GRAYS), elast=2.0)"
            or "MaskedLimitFilter(flt_gray8, src_gray8, thr=1.5, elast=core.std.BlankClip(flt_gray8, color=2, format=vs.GRAY8))"
            or "MaskedLimitFilter(flt_gray8, src_gray8, thr=1.5, elast=core.std.BlankClip(flt_gray8, color=2.0, format=vs.GRAYS))"

    """

    funcName = 'MaskedLimitFilter'

    if not isinstance(flt, vs.VideoNode):
        raise TypeError(f'{funcName}: "flt" must be a clip!')

    if not isinstance(src, vs.VideoNode):
        raise TypeError(f'{funcName}: "src" must be a clip!')
    elif flt.format.id != src.format.id or flt.width != src.width or flt.height != src.height:
        raise ValueError(f'{funcName}: "flt" and "src" must be of the same format, width and height!')

    if ref is not None:
        if not isinstance(ref, vs.VideoNode):
            raise TypeError(f'{funcName}: "ref" must be a clip!')
        elif flt.format.id != ref.format.id or flt.width != ref.width or flt.height != ref.height:
            raise ValueError(f'{funcName}: "flt" and "ref" must be of the same format, width and height!')
    else:
        ref = src

    # parameters
    if not isinstance(thr, (int, float)):
        if not isinstance(thr, vs.VideoNode):
            raise TypeError(f'{funcName}: "thr" must be a clip, an int or a float!')
        elif (flt.width != thr.width or flt.height != thr.height or flt.format.color_family != thr.format.color_family or
            flt.format.subsampling_w != thr.format.subsampling_w or flt.format.subsampling_h != thr.format.subsampling_h):
            raise ValueError(f'{funcName}: "flt" and "thr" must be of the same width, height, color space and subsampling!')
    elif thr < 0:
        raise ValueError(f'{funcName}: valid range of "thr" is [0, +inf)')

    if not isinstance(elast, (int, float)):
        if not isinstance(elast, vs.VideoNode):
            raise TypeError(f'{funcName}: "elast" must be a clip, an int or a float!')
        elif (flt.width != elast.width or flt.height != elast.height or flt.format.color_family != elast.format.color_family or
            flt.format.subsampling_w != elast.format.subsampling_w or flt.format.subsampling_h != elast.format.subsampling_h):
            raise ValueError(f'{funcName}: "flt" and "elast" must be of the same width, height, color space and subsampling!')
    elif elast < 1:
        raise ValueError(f'{funcName}: valid range of "elast" is [1, +inf)')

    if brighten_thr is None:
        brighten_thr = thr
    elif not isinstance(brighten_thr, (int, float)):
        if not isinstance(brighten_thr, vs.VideoNode):
            raise TypeError(f'{funcName}: "brighten_thr" must be a clip, an int or a float!')
        elif (flt.width != brighten_thr.width or flt.height != brighten_thr.height or flt.format.color_family != brighten_thr.format.color_family or
            flt.format.subsampling_w != brighten_thr.format.subsampling_w or flt.format.subsampling_h != brighten_thr.format.subsampling_h):
            raise ValueError(f'{funcName}: "flt" and "brighten_thr" must be of the same width, height, color space and subsampling!')
    elif brighten_thr < 0:
        raise ValueError(f'{funcName}: valid range of "brighten_thr" is [0, +inf)')

    # planes
    if planes is None:
        planes = list(range(flt.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]
    elif isinstance(planes, list):
        for plane in planes:
            if not isinstance(plane, int):
                raise TypeError(f'{funcName}: "planes" must be a (list of) int!')

            if plane < 0 or plane >= flt.format.num_planes:
                raise ValueError(funcName + ': plane index out of range')
    else:
        raise TypeError(f'{funcName}: "planes" must be a (list of) int!')

    # process
    value_range = (1 << flt.format.bits_per_sample) - 1 if flt.format.sample_type == vs.INTEGER else 1

    var = (chr((i + ord('x') - ord('a')) % 26 + ord('a')) for i in range(26))

    flt_str = next(var)
    src_str = next(var)
    ref_str = next(var) if ref != src else src_str
    thr_str = next(var) if isinstance(thr, vs.VideoNode) else f"{thr}"
    elast_str = next(var) if isinstance(elast, vs.VideoNode) else f"{elast}"

    dif_str = f"{flt_str} {src_str} -"
    dif_ref_str = f"{flt_str} {ref_str} -"
    dif_abs_str = f"{dif_ref_str} abs"

    def foldable(string: str) -> bool:
        return not any(char.isalpha() for char in string)

    if value_range == 255:
        thr_1_str = thr_str

        if foldable(thr_str) and foldable(elast_str):
            thr_2_str = f"{float(thr_str) * float(elast_str)}"
        else:
            thr_2_str = f"{thr_str} {elast_str} *"

    else: # value_range / 255 != 1
        thr_1_str = (f"{float(thr_str) * (value_range / 255)}" if foldable(thr_str)
            else f"{thr_str} {value_range / 255} *")

        if foldable(thr_str):
            if foldable(elast_str):
                thr_2_str = f"{float(thr_str) * (value_range / 255) * float(elast_str)}"
            else:
                thr_2_str = f"{float(thr_str) * (value_range / 255)} {elast_str} *"
        else:
            if foldable(elast_str):
                thr_2_str = f"{thr_str} {(value_range / 255) * float(elast_str)} *"
            else:
                thr_2_str = f"{thr_str} {value_range / 255} * {elast_str} *"

    thr_slope_str = (f"{1 / (float(thr_2_str) - float(thr_1_str))}" if foldable(thr_1_str) and foldable(thr_2_str)
        else f"1 {thr_2_str} {thr_1_str} - /")

    # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
    limitExpr = f"{src_str} {dif_str} {thr_2_str} {dif_abs_str} - * {thr_slope_str} * +"
    limitExpr = f"{dif_abs_str} {thr_1_str} <= {flt_str} {dif_abs_str} {thr_2_str} >= {src_str} {limitExpr} ? ?"

    if brighten_thr is thr:
        if ref is src:
            clips = [clip for clip in [flt, src, thr, elast] if isinstance(clip, vs.VideoNode)]
        else:
            clips = [clip for clip in [flt, src, ref, thr, elast] if isinstance(clip, vs.VideoNode)]

        return core.std.Expr(clips, [(limitExpr if i in planes else "") for i in range(flt.format.num_planes)])

    else:
        brighten_thr_str = next(var) if isinstance(brighten_thr, vs.VideoNode) else f"{brighten_thr}"

        if value_range == 255:
            brighten_thr_1_str = brighten_thr_str

            if foldable(brighten_thr_str) and foldable(elast_str):
                brighten_thr_2_str = f"{float(brighten_thr_str) * float(elast_str)}"
            else:
                brighten_thr_2_str = f"{brighten_thr_str} {elast_str} *"

        else: # value_range / 255 != 1
            brighten_thr_1_str = (f"{float(brighten_thr_str) * (value_range / 255)}" if foldable(brighten_thr_str)
                else f"{brighten_thr_str} {value_range / 255} *")

            if foldable(brighten_thr_str):
                if foldable(elast_str):
                    brighten_thr_2_str = f"{float(brighten_thr_str) * (value_range / 255) * float(elast_str)}"
                else:
                    brighten_thr_2_str = f"{float(brighten_thr_str) * (value_range / 255)} {elast_str} *"
            else:
                if foldable(elast_str):
                    brighten_thr_2_str = f"{brighten_thr_str} {(value_range / 255) * float(elast_str)} *"
                else:
                    brighten_thr_2_str = f"{brighten_thr_str} {value_range / 255} * {elast_str} *"

        brighten_thr_slope_str = (f"{1 / (float(brighten_thr_2_str) - float(brighten_thr_1_str))}"
            if foldable(brighten_thr_1_str) and foldable(brighten_thr_2_str)
            else f"1 {brighten_thr_2_str} {brighten_thr_1_str} - /")

        # final = src + dif * (brighten_thr_2 - dif_abs) / (brighten_thr_2 - brighten_thr_1)
        brighten_limitExpr = f"{src_str} {dif_str} {brighten_thr_2_str} {dif_abs_str} - * {brighten_thr_slope_str} * +"
        brighten_limitExpr = f"{dif_abs_str} {brighten_thr_1_str} <= {flt_str} {dif_abs_str} {brighten_thr_2_str} >= {src_str} {brighten_limitExpr} ? ?"

        limitExpr = f"{flt_str} {ref_str} > {brighten_limitExpr} {limitExpr} ?"

        if ref is src:
            clips = [clip for clip in [flt, src, thr, elast, brighten_thr] if isinstance(clip, vs.VideoNode)]
        else:
            clips = [clip for clip in [flt, src, ref, thr, elast, brighten_thr] if isinstance(clip, vs.VideoNode)]

        return core.std.Expr(clips, [(limitExpr if i in planes else "") for i in range(flt.format.num_planes)])


def multi_scale(func: Optional[Callable[..., vs.VideoNode]] = None, down_scale: float = 1.5,
                up_scale_func: Optional[Callable[[vs.VideoNode, int, int], vs.VideoNode]] = None,
                down_scale_func: Optional[Callable[[vs.VideoNode, int, int], vs.VideoNode]] = None,
                multi_scale_mode: int = 1, num_levels: int = 2
                ) -> vs.VideoNode:
    """A decorator that "multi-scale" a given function

    Note that the resulting function may be significantly different from its single-scale counterpart.

    Args:
        func: Function to be decorated.
            The function should not change properties (width, height, etc) of its input.

        down_scale: (float) Down-scaling factor of succesive levels.
            Default is 1.5.

        up_scale_func, down_scale_func: (function) Functions used for up-scaling / down-scaling.
            Each function should take a clip (vapoursynth.VideoNode) as the first argument, and the output image
                dimensions (width and height) as the second and third arguments, respectively.
            Examples include "core.resize.*", "core.fmtc.resample", "nnedi3_resample.nnedi3_resample".
            Default is core.resize.Spline36.

        multi_scale_mode: (int, -3~-1 or 1~3) Controls how multi-scale filtering is done.
            Default is 1.

        num_levels: (int) Number of levels of the gaussian pyramid.
            Default is 2.

    Examples:
        # (reference) single-scale RemoveGrain

        last = core.rgvs.RemoveGrain(gray8, mode=11)

        ########################################################################
        # (1) multi-scale RemoveGrain

        last = multi_scale(core.rgvs.RemoveGrain)(gray8, mode=11)

        ########################################################################
        # (2) parameters of the decorator can be specified by

        last = multi_scale(core.rgvs.RemoveGrain, up_scale_func=core.resize.Lanczos)(gray8, mode=11)

        ########################################################################
        # (3) @decorator syntax makes the decoration on custom functions easier
        # this example is equivalent to the first example

        @multi_scale
        def rg(clip, mode):
            return core.rgvs.RemoveGrain(clip, mode=mode)

        last = rg(gray8, mode=11)

        ########################################################################
        # (4) the second example can also be written as

        @multi_scale(up_scale_func=core.resize.Lanczos)
        def rg(clip, mode):
            return core.rgvs.RemoveGrain(clip, mode=mode)

        last = rg(gray8, mode=11)

        ########################################################################
        # (5) multiple decorations are allowed

        @multi_scale
        @multi_scale() # this line is equivalent to the line above
        def rg(clip, mode):
            return core.rgvs.RemoveGrain(clip, mode=mode)

        last = rg(gray8, mode=11)

    """

    funcName = "multi_scale"

    if up_scale_func is None:
        up_scale_func = core.resize.Spline36

    if down_scale_func is None:
        down_scale_func = core.resize.Spline36

    if num_levels < 0:
        raise ValueError(f'{funcName}: "num_levels" must be greater than 0! (got {num_levels})')

    if func is None:
        return lambda func: functools.wraps(func)(lambda clip, *args, **kwargs: _multi_scale_filtering( # type: ignore
            clip, func, down_scale, up_scale_func, down_scale_func, multi_scale_mode, num_levels, *args, **kwargs)) # type: ignore
    elif callable(func):
        return functools.wraps(func)(lambda clip, *args, **kwargs: _multi_scale_filtering( # type: ignore
            clip, func, down_scale, up_scale_func, down_scale_func, multi_scale_mode, num_levels, *args, **kwargs)) # type: ignore
    else:
        raise TypeError(f'{funcName}: Unknown type of "func"! (got {type(func)})')


def _multi_scale_filtering(clip: vs.VideoNode, func: Callable[..., vs.VideoNode], down_scale: float,
                           up_scale_func: Callable[[vs.VideoNode, int, int], vs.VideoNode],
                           down_scale_func: Callable[[vs.VideoNode, int, int], vs.VideoNode],
                           multi_scale_mode: int, num_levels: int, *args: Any, **kwargs: Any
                           ) -> vs.VideoNode:
    """"Internal function used by multi_scale()"""

    if num_levels == 0:
        # In this implementation, the bottom-most level (0-level) is defined to be the coarsest level
        return clip

    else:
        down_w = haf_m4(clip.width / down_scale)
        down_h = haf_m4(clip.height / down_scale)

        # current level of unfiltered gaussian pyramid (low-res)
        low_res = down_scale_func(clip, down_w, down_h)

        # filtered result from lower levels
        lower_result = _multi_scale_filtering(low_res, func, down_scale, up_scale_func, down_scale_func, multi_scale_mode, num_levels-1, *args, **kwargs)

        if abs(multi_scale_mode) == 1:
            # current level of filtered gaussian pyramid (low-res)
            filtered_lower_result = func(lower_result, *args, **kwargs)

            # current level of filtered gaussian pyramid (high-res)
            filtered_low_result_upscaled = up_scale_func(filtered_lower_result, clip.width, clip.height)

            # current level of unfiltered gaussian pyramid (high-res)
            low_res_upscaled = up_scale_func(low_res, clip.width, clip.height)

            return core.std.Expr([clip, low_res_upscaled, filtered_low_result_upscaled], ['x y - z +' if multi_scale_mode > 0 else 'x y + z -'])

        elif abs(multi_scale_mode) == 2:
            # current level of filtered gaussian pyramid (low-res)
            filtered_lower_result = func(lower_result, *args, **kwargs)

            # current level of filtered gaussian pyramid (high-res)
            filtered_low_result_upscaled = up_scale_func(filtered_lower_result, clip.width, clip.height)

            # current level of unfiltered gaussian pyramid (high-res)
            lower_result_upscaled = up_scale_func(lower_result, clip.width, clip.height)

            return core.std.Expr([clip, lower_result_upscaled, filtered_low_result_upscaled], ['x y - z +' if multi_scale_mode > 0 else 'x y + z -'])

        elif abs(multi_scale_mode) == 3:
            # current level of filtered gaussian pyramid (low-res)
            filtered_low_res = func(low_res, *args, **kwargs)

            # current level of filtered gaussian pyramid (high-res)
            filtered_low_res_upscaled = up_scale_func(filtered_low_res, clip.width, clip.height)

            # current level of unfiltered gaussian pyramid (high-res)
            lower_result_upscaled = up_scale_func(lower_result, clip.width, clip.height)

            return core.std.Expr([clip, lower_result_upscaled, filtered_low_res_upscaled], ['x y - z +' if multi_scale_mode > 0 else 'x y + z -'])

        else:
            raise ValueError(f'multi_scale: Unknown value of "multi_scale_mode"! (-3~-1 or 1~3, got {multi_scale_mode})')


def avg_decimate(clip: vs.VideoNode, clip2: Optional[vs.VideoNode] = None, weight: float = 0.5,
                 **vdecimate_kwargs: Any) -> vs.VideoNode:
    """Averaging-based decimation filter

    Proposed by feisty2.

    It is a decimation filter, which averages duplicates frames before drops one in every cycle frames.

    Args:
        clip: Input clip.
            Must have constant format and dimensions, known length, integer sample type, and bit depth between 8 and 16 bits per sample.

        clip2: (clip) Clip that will use to create the output frames.
            If clip2 is used, this filter will perform all calculations based on clip, but will copy the chosen fields from clip2.
            This can be used to work around VDecimate’s video format limitations.
            Default is None.

        weight: (float, 0~1) Weight used to merge duplicates frames.
            Default is 0.5.

        vdecimate_args: (dict) Additional parameters passed to core.vivtc.VDecimate in the form of dict.

    """

    funcName = 'avg_decimate'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    assert clip.num_frames >= 2

    def avg_func(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        if f.props["VDecimateDrop"] == 1:
            # forward averaging
            if _is_api4:
                return core.std.AverageFrames(clip, weights=[0, 1-weight, weight])
            else:
                return core.misc.AverageFrames(clip, weights=[0, 1-weight, weight])
        else:
            return clip

    analysis = core.vivtc.VDecimate(clip, clip2=clip2, dryrun=True, **vdecimate_kwargs)

    average = core.std.FrameEval(analysis, functools.partial(avg_func, clip=analysis), prop_src=analysis[1:])

    decimate = core.vivtc.VDecimate(clip, clip2=average, dryrun=False, **vdecimate_kwargs)

    return decimate


def YAHRmask(clp: vs.VideoNode, expand: float = 5, warpdepth: int = 32, blur: int = 2,
             useawarp4: bool = False, yahr: Optional[vs.VideoNode] = None
             ) -> vs.VideoNode:
    """YAHRmask

    Author: Tophf

    Source: https://pastebin.com/raw/LUErwWR8

    Modified from Holy's havsfunc.YAHR()

    binomialblur(variance=expand*2) in Avisynth is implemented by core.tcanny.TCanny(sigma=sqrt(expand*2), mode=-1)

    Args:
        clp: Input clip. Must have constant format and dimensions, 8..16 bit integer pixels, and it must not be RGB.

        expand: (float) Expansion of edge mask.
            Default is 5.

        warpdepth: (int) AWarpSharp()'s "depth". Controls how far to warp.
            Default is 32.

        blur: (int) AWarpSharp()'s "blur". Controls the number of times to blur the edge mask for AWarp().
            Default is 2.

        useawarp4: (bool) Useful for better subpixel interpolation in warping.
            Default is False.

        yahr: (clip) User'defined YAHR result.
            Must be of the same size and format as "clp".
            Default is None.
    """

    funcName = 'YAHRmask'

    if not isinstance(clp, vs.VideoNode):
        raise TypeError(f'{funcName}: "clp" must be a clip!')

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp # type: Optional[vs.VideoNode]
        clp = mvf.GetPlane(clp, 0)
    else:
        clp_orig = None

    if yahr is None:
        # YAHR2(warpdepth, blur, useawarp4)
        b1 = core.std.Convolution(haf_MinBlur(clp, 2), matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
        b1D = core.std.MakeDiff(clp, b1)

        if not useawarp4:
            w1 = haf_Padding(clp, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=warpdepth).std.Crop(6, 6, 6, 6)
        else:
            awarp_mask = core.warp.ASobel(clp).warp.ABlur(blur=blur)
            clp4 = clp.resize.Bilinear(clp.width*4, clp.height*4, src_left=0.375, src_top=0.375)
            w1 = core.warp.AWarp(clp4, awarp_mask, depth=warpdepth)

        w1b1 = core.std.Convolution(haf_MinBlur(w1, 2), matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
        w1b1D = core.std.MakeDiff(w1, w1b1)
        DD = core.rgvs.Repair(b1D, w1b1D, 13)
        DD2 = core.std.MakeDiff(b1D, DD)
        yahr = core.std.MakeDiff(clp, DD2)
    else:
        if not isinstance(yahr, vs.VideoNode):
            raise TypeError(f'{funcName}: "yahr" must be a clip!')

        yahr = mvf.GetPlane(yahr, 0)

        if yahr.format.id != clp.format.id or yahr.width != clp.width or yahr.height != clp.height: # type: ignore
            raise ValueError(f'{funcName}: "yahr" must be of the same size and format as "clp"!')

    # mt_lutxy(clp, mt_expand().mt_expand(),"x y - abs 8 - 7 <<")
    vEdge = core.std.Expr([clp, clp.std.Maximum().std.Maximum()], ['y x - {i} - 128 *'.format(i=8 * ((1 << clp.format.bits_per_sample) - 1) / 255)])

    # vEdge.binomialblur(expand*2).mt_lut("x 4 <<")
    mask1 = core.tcanny.TCanny(vEdge, sigma=math.sqrt(expand*2), mode=-1).std.Expr(['x 16 *'])

    # vEdge.removegrain(12, -1).mt_invert()
    mask2 = core.std.Convolution(vEdge, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Invert()

    # mt.logic(mask1, mask2, "min")
    mask = core.std.Expr([mask1, mask2], ['x y min'])

    # mt_merge(clp, yahr, mask)
    last = core.std.MaskedMerge(clp, yahr, mask) # type: ignore

    if clp_orig is not None:
        return core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    else:
        return last


def Cdeblend(input: vs.VideoNode, omode: int = 0, bthresh: float = 0.1, mthresh: float = 0.6,
             xr: float = 1.5, yr: float = 2.0, fnr: bool = False,
             dclip: Optional[vs.VideoNode] = None,
             sequential: Optional[bool] = None,
             preroll: int = 14
             ) -> vs.VideoNode:
    """A simple blend replacing function like unblend or removeblend

    Port from Cdeblend v1.1b (http://avisynth.nl/images/Cdeblend.avs)

    Args:
        input: Input clip.
            If "dclip" is None, input must be a YUV/Gray clip.

        omode: (int, 0~4) Stands for the output mode. There are five different omodes:
            omode 0 -> The previous frame will be duplicated to avoid a blend.
            omode 1 -> The next frame is used instead of a blend.
            omode 2 -> Like omode 0 but with some double-blend detection (only without missing fields).
            omode 3 -> A special mode for 12fps sources.
            omode 4 -> Does nothing with the source. It just subtitles the blend factor.
            Default: 0

        bthresh: (float, 0~2) For omode 0 and 1 bthresh will be just compared with the calculated blend factor.
            If the blend factor is higher a blend is detected.
            Omode 3 uses this threshold to detect clears.
            If blendfactor<-bthresh the frame will be output also if there is another frame with a smaller blendfactor.
            This can give a better motion.
            Default: 0.1

        mthresh: (float, 0.3~1.6) Used for (m)otion (thresh)olding.
            It regulates the blend detection of frames with small pixel value differences.
            A better quality of the source allows lower values and a more accurate detection.
            Default: 0.6

        xr, yr: (float, 1.0-4.0) The scaled detection radius (xr & yr) blurs the internal detection clip
            and can speed up the function a little bit.
            Default: 1.5, 2.0

        fnr: (bool) With fnr=True you enable a (f)ast (n)oise (r)eduction.
            It's sometimes useful for noisy sources and typical area motion (anime).
            Don't use a too big radius if you enable this feature and don't use it on very clean sources
            (speed decreasing is not the only negative effect).
            Default: False

        dclip: (clip) The detection clip can be set to improve the blend detection (cleaning the clip before).
            This clip is only used for the blend detection and not for output.
            Must be a YUV/Gray clip and must be of the same number of frames as "input".
            Default: "input".

        sequential: (bool) Experimental support for sequential processing.
            Default: False

        preroll: (int) How many frames before the current one metrics are computed for.
            Alternative to true sequential processing.
            Must be 1 or bigger, consistent results start around 14.
            Default: 14.
    """

    funcName = 'Cdeblend'

    # check
    if not isinstance(input, vs.VideoNode):
        raise TypeError(f'{funcName}: "input" must be a clip!')

    if dclip is None:
        if input.format.color_family not in [vs.YUV, vs.GRAY]:
            raise TypeError(f'{funcName}: "input" must be a YUV/Gray clip if "dclip" is not provided!')
    else:
        if not isinstance(dclip, vs.VideoNode) or dclip.format.color_family not in [vs.YUV, vs.GRAY]:
            raise TypeError(f'{funcName}: "dclip" must be a YUV/Gray clip!')

        if dclip.num_frames != input.num_frames:
            raise TypeError(f'{funcName}: "dclip" must of the same number of frames as "input"!')

    if sequential is None:
        sequential = False

    preproc = {i: input[0:2] + input[2+i:] for i in range(-2, 3)}

    def evaluate(n: int, f: List[vs.VideoFrame], clip: vs.VideoNode, core: vs.Core) -> vs.VideoNode:
        # coefficients
        Cbp2 = 128.0 if bthresh > 0 else 2.0
        Cbp1 = 128.0 if bthresh > 0 else 2.0
        Cbc0 = 128.0 if bthresh > 0 else 2.0
        Cbn1 = 128.0 if bthresh > 0 else 2.0
        Cbn2 = 128.0 if bthresh > 0 else 2.0

        Cdp1 = mthresh * 0.5
        Cdc0 = mthresh * 0.5
        Cdn1 = mthresh * 0.5

        current = 0

        def computestate(Cdiff: float, Cbval: float):
            nonlocal Cdp1, Cdc0, Cdn1, Cbp2, Cbp1, Cbc0, Cbn1, Cbn2, current
            
            Cdp1 = Cdc0
            Cdc0 = Cdn1 if omode > 1 else Cdiff # type: ignore
            Cdn1 = Cdiff # type: ignore

            Cbp2 = Cbp1
            Cbp1 = Cbc0
            Cbc0 = Cbn1 if omode > 1 else (0.0 if Cdc0 < mthresh else (Cbval - Cbn2) / ((max(Cdc0, Cdp1) + mthresh) ** 0.8)) # type: ignore
            Cbn1 = 0.0 if (Cdc0 < mthresh or Cdn1 < mthresh) else (Cbval - Cbn2) / ((max(Cdc0, Cdp1) + mthresh) ** 0.8) # type: ignore
            Cbn2 = Cbval # type: ignore

            current = (1 if ((Cbn1 < -bthresh and Cbc0 > bthresh) or (Cbn1 < 0 and Cbc0 > 0 and Cbc0 + Cbp1 > 0 and Cbc0 + Cbp1 + Cbp2 > 0)) else
                       0 if ((Cbc0 < -bthresh and Cbp1 > bthresh) or (Cbc0 < 0 and Cbc0 + Cbn1 < 0 and Cbp1 > 0 and Cbp1 + Cbp2 > 0)) else
                       (2 if Cbn1 > 0 else 1) if current == -2 else
                       current - 1)

            if omode == 2:
                current = (-1 if min(-Cbp1, Cbc0 + Cbn1) > bthresh and abs(Cbn1) > abs(Cbc0) else
                           1 if min(-Cbp2 - Cbp1, Cbc0) > bthresh and abs(Cbp2) > abs(Cbp1) else
                           -1 if min(-Cbp1, Cbc0) > bthresh else
                           0)

            if omode <= 1:
                current = (0 if min(-Cbp1, Cbc0) < bthresh else
                           -1 if omode == 0 else
                           1)

        for i in range(max(0, preroll - n), preroll + 1):
            Cdiff = f[i * 2 + 0].props["PlaneMAE"] * 255 # type: ignore
            Cbval = f[i * 2 + 1].props["PlaneMean"] * 255 # type: ignore
            computestate(Cdiff, Cbval)

        if omode != 4:
            return preproc[current]
        else:
            text = f'{min(-Cbp1, Cbc0) if Cbc0 > 0 and Cbp1 < 0 else 0.0}{" -> BLEND!!" if min(-Cbp1, Cbc0) >= bthresh else " "}'
            return core.text.Text(clip, text)

    def evaluate_seq(
        n: int, f: List[vs.VideoFrame], clip: vs.VideoNode, core: vs.Core,
        real_n: int, input: vs.VideoNode
    ) -> vs.VideoNode:
        # nonlocal Cdp3, Cdp2, Cbp3
        # nonlocal Cdp1, Cdc0, Cdn1, Cbp2, Cbp1, Cbc0, Cbn1, Cbn2, current

        Cdiff = f[0].props["PlaneMAE"] * 255 # type: ignore
        Cbval = f[1].props["PlaneMean"] * 255 # type: ignore

        if real_n > 0:
            state = f[2].props
            Cdp1 = state["Cdp1"]
            Cdc0 = state["Cdc0"]
            Cdn1 = state["Cdn1"]
            Cbp2 = state["Cbp2"]
            Cbp1 = state["Cbp1"]
            Cbc0 = state["Cbc0"]
            Cbn1 = state["Cbn1"]
            Cbn2 = state["Cbn2"]
            current = state["current"]
        else:
            # coefficients
            # Cbp3 = 128.0 if bthresh > 0 else 2.0
            Cbp2 = 128.0 if bthresh > 0 else 2.0
            Cbp1 = 128.0 if bthresh > 0 else 2.0
            Cbc0 = 128.0 if bthresh > 0 else 2.0
            Cbn1 = 128.0 if bthresh > 0 else 2.0
            Cbn2 = 128.0 if bthresh > 0 else 2.0

            # Cdp3 = mthresh * 0.5
            # Cdp2 = mthresh * 0.5
            Cdp1 = mthresh * 0.5
            Cdc0 = mthresh * 0.5
            Cdn1 = mthresh * 0.5

            current = 0

        # Cdp3 = Cdp2
        # Cdp2 = Cdp1
        Cdp1 = Cdc0
        Cdc0 = Cdn1 if omode > 1 else Cdiff # type: ignore
        Cdn1 = Cdiff # type: ignore

        # Cbp3 = Cbp2
        Cbp2 = Cbp1
        Cbp1 = Cbc0
        Cbc0 = Cbn1 if omode > 1 else (0.0 if Cdc0 < mthresh else (Cbval - Cbn2) / ((max(Cdc0, Cdp1) + mthresh) ** 0.8)) # type: ignore
        Cbn1 = 0.0 if (Cdc0 < mthresh or Cdn1 < mthresh) else (Cbval - Cbn2) / ((max(Cdc0, Cdp1) + mthresh) ** 0.8) # type: ignore
        Cbn2 = Cbval # type: ignore

        current = (1 if ((Cbn1 < -bthresh and Cbc0 > bthresh) or (Cbn1 < 0 and Cbc0 > 0 and Cbc0 + Cbp1 > 0 and Cbc0 + Cbp1 + Cbp2 > 0)) else
                   0 if ((Cbc0 < -bthresh and Cbp1 > bthresh) or (Cbc0 < 0 and Cbc0 + Cbn1 < 0 and Cbp1 > 0 and Cbp1 + Cbp2 > 0)) else
                   (2 if Cbn1 > 0 else 1) if current == -2 else
                   current - 1)

        if omode == 2:
            current = (-1 if min(-Cbp1, Cbc0 + Cbn1) > bthresh and abs(Cbn1) > abs(Cbc0) else
                       1 if min(-Cbp2 - Cbp1, Cbc0) > bthresh and abs(Cbp2) > abs(Cbp1) else
                       -1 if min(-Cbp1, Cbc0) > bthresh else
                       0)

        if omode <= 1:
            current = (0 if min(-Cbp1, Cbc0) < bthresh else
                       -1 if omode == 0 else
                       1)

        if omode != 4:
            ret = input[min(max(0, real_n + current), input.num_frames - 1)]
        else:
            text = f'{min(-Cbp1, Cbc0) if Cbc0 > 0 and Cbp1 < 0 else 0.0}{" -> BLEND!!" if min(-Cbp1, Cbc0) >= bthresh else " "}'
            ret = core.text.Text(clip, text)

        kwargs = dict(Cdp1=Cdp1, Cdc0=Cdc0, Cdn1=Cdn1, Cbp2=Cbp2, Cbp1=Cbp1, Cbc0=Cbc0, Cbn1=Cbn1, Cbn2=Cbn2, current=current)
        if hasattr(core.std, "SetFrameProps"):
            return core.std.SetFrameProps(ret, **kwargs)
        else:
            for k, v in kwargs.items():
                ret = core.std.SetFrameProp(ret, prop=k, intval=v)
            return ret

    # process
    blendclip = input if dclip is None else dclip
    blendclip = mvf.GetPlane(blendclip, 0)
    if fnr:
        blendclip = core.median.TemporalMedian(blendclip, radius=1) # type: ignore
    blendclip = core.resize.Bilinear(blendclip, int(blendclip.width * 0.125 / xr) * 8, int(blendclip.height * 0.125 / yr) * 8)

    diff = core.std.MakeDiff(blendclip, blendclip[1:])

    if input.format.sample_type == vs.INTEGER:
        neutral = 1 << (input.format.bits_per_sample - 1)
        neutral2 = 1 << input.format.bits_per_sample
        peak = (1 << input.format.bits_per_sample) - 1
        inv_scale = "" if input.format.bits_per_sample == 8 else f"{255 / peak} *"
        scale = 1 if input.format.bits_per_sample == 8 else peak / 255

    elif input.format.sample_type == vs.FLOAT:
        neutral = 0.5 # type: ignore
        neutral2 = 1
        peak = 1
        inv_scale = f"{255 / peak} *"
        scale = peak / 255

    x_diff = f"x {neutral} - abs"
    y_diff = f"y {neutral} - abs"
    xy_diff = f"x y + {neutral2} - abs"

    expr = (f"{x_diff} {y_diff} < {x_diff} {xy_diff} < and "                       # ((abs(x-128) < abs(y-128)) && (abs(x-128) < abs(x+y-256)) ?
            f"{x_diff} {inv_scale} dup sqrt - dup * "                              #  sqrt(abs(x-128) - sqrt(abs(x-128)))^2 :
            f"{y_diff} {xy_diff} < "                                               #  abs(y-128) < abs(x+y-256) ?
            f"{y_diff} {inv_scale} dup sqrt - dup * "                              #  sqrt(abs(y-128) - sqrt(abs(y-128)))^2 :
            f"{xy_diff} {inv_scale} dup sqrt - dup * ? ? "                         #  sqrt(abs(x+y-256) - sqrt(abs(x+y-256)))^2
            f"x {neutral} - y {neutral} - * 0 > {-scale} {scale} ? * {neutral} +") # ) * ((x-128) * (y-128) > 0 ? -1 : 1) + 128

    mask = core.std.Expr([diff, diff[1:]], [expr])

    if omode > 1:
        # LumaDifference(blendclip.trim(1,0),blendclip.trim(3,0))
        Cdiff = mvf.PlaneCompare(blendclip[1:], blendclip[3:], mae=True, rmse=False, psnr=False, cov=False, corr=False)

        # AverageLuma(mask.trim(1,0))
        Cbval = mvf.PlaneStatistics(mask[1:], mean=True, mad=False, var=False, std=False, rms=False)
    else:
        # LumaDifference(blendclip,blendclip.trim(2,0)
        Cdiff = mvf.PlaneCompare(blendclip, blendclip[2:], mae=True, rmse=False, psnr=False, cov=False, corr=False)

        # AverageLuma(mask)
        Cbval = mvf.PlaneStatistics(mask, mean=True, mad=False, var=False, std=False, rms=False)

    if sequential:
        states = []
        for n, (input_frame, Cdiff_frame, Cbval_frame) in enumerate(zip(input, Cdiff, Cbval)):
            state = core.std.FrameEval(
                input_frame,
                functools.partial(evaluate_seq, clip=input_frame, core=core, real_n=n, input=input),
                prop_src=[Cdiff_frame, Cbval_frame] + ([] if n == 0 else [state])
            )
            states.append(state)
        last = core.std.Splice(states)
    else:
        prop_src = []
        for i in range(preroll, 0, -1):
            prop_src.extend([Cdiff[0] * i + Cdiff, Cbval[0] * i + Cbval])
        prop_src.extend([Cdiff, Cbval])
            
        last = core.std.FrameEval(input, functools.partial(evaluate, clip=input, core=core), prop_src=prop_src)

    recl = haf_ChangeFPS(haf_ChangeFPS(last, last.fps_num * 2, last.fps_den * 2), last.fps_num, last.fps_den)

    if not _is_api4:
        recl = recl.std.Cache(make_linear=True)

    return recl


def S_BoxFilter(clip: vs.VideoNode, radius: int = 1, planes: PlanesType = None) -> vs.VideoNode:
    """Side window box filter

    Side window box filter is a local edge-preserving filter that is derived from
    conventional box filter and side window filtering technique.

    Args:
        clip: Input clip.

        radius: (int) Radius of filtering.
            The size is (radius*2+1) * (radius*2+1).
            Default: 1

        planes: (int []) Specify which planes to process.
            Unprocessed planes will be copied from the first clip "flt".
            By default, all planes will be processed.

    Ref:
        [1] Yin, H., Gong, Y., & Qiu, G. (2019) Side Window Filtering. arXiv preprint arXiv:1905.07177.
        [2] https://github.com/YuanhaoGong/SideWindowFilter

    """

    funcName = 'S_BoxFilter'

    # check
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    half_kernel = [(1 if i <= 0 else 0) for i in range(-radius, radius+1)]

    # set of side window filters
    # the third filter of each set can be further optimized using results of the first two filters, but the code will look complicated
    vrt_filters = [
        functools.partial(core.std.Convolution, matrix=half_kernel, planes=planes, mode='v'),
        functools.partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes, mode='v'),
        functools.partial(core.std.BoxBlur, planes=planes, hpasses=0, vradius=radius, vpasses=1)
    ]

    hrz_filters = [
        functools.partial(core.std.Convolution, matrix=half_kernel, planes=planes, mode='h'),
        functools.partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes, mode='h'),
        functools.partial(core.std.BoxBlur, planes=planes, hradius=radius, hpasses=1, vpasses=0)
    ]

    # process
    vrt_intermediates = (vrt_flt(clip) for vrt_flt in vrt_filters)
    intermediates = (hrz_flt(vrt_intermediate)
        for i, vrt_intermediate in enumerate(vrt_intermediates)
        for j, hrz_flt in enumerate(hrz_filters)
        if not i==j==2)

    expr = [("x z - abs y z - abs < x y ?" if i in planes else "") for i in range(clip.format.num_planes)]
    res = functools.reduce(lambda x, y: core.std.Expr([x, y, clip], expr), intermediates)

    return res


def VFRSplice(clips: Sequence[vs.VideoNode], tcfile: Optional[Union[str, os.PathLike]] = None,
              v2: bool = True, precision: int = 6, cfr_ref: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    """fractions-based VFRSplice()

    This function is modified from mvsfunc.VFRSplice().

    Same as mvsfunc.VFRSplice(), with possibly higher precision for timecode v2 output
    since it is implemented based on rational number arithmetic
    rather than floating-point arithmetic.

    In addition, added constant frame rate output, for x265 (or other encoders).

    Args:
        clips: List of clips to be spliced.
            Each clip should be CFR (constant frame rate).

        tcfile: (str or os.PathLike) Timecode file output. Supports recursive directory creation.
            Default: None.

        v2: (bool) Timecode format.
            True for v2 output and False for v1 output.
            Default: True.

        precision: (int) Precision of time and frame rate,
            indicating how many digits should be displayed after the decimal point for a fixed-point value.
            Default: 6.

        cfr_ref: (VideoNode)
            You can input a reference clip (can be your source clip) for CFR output.
            If you input a reference clip, it should also be CFR, and the time of the output clip should same as the reference clip.
            Default: None.

        An example:
            source = core.std.BlankClip(length=3000, fpsnum=30000, fpsden=1001).text.FrameNum()
            clip1 = core.std.BlankClip(length=1200, fpsnum=24000, fpsden=1001).text.FrameNum()
            clip2 = core.std.BlankClip(length=3000, fpsnum=60000, fpsden=1001).text.FrameNum()
            # The time of the output clip(clip1 + clip2) is same as the reference(source) clip.
            out = muvsfunc.VFRSplice([clip1, clip2], cfr_ref=source)
            out.set_output()

    """

    funcName = "VFRSplice"

    # Arguments
    if isinstance(clips, vs.VideoNode):
        clips = [clips]
    elif isinstance(clips, abc.Sequence):
        for clip in clips:
            if not isinstance(clip, vs.VideoNode):
                raise TypeError(f'{funcName}: each element in "clips" must be a clip!')
            if clip.fps.numerator == 0 and clip.fps.denominator == 1:
                raise ValueError(f'{funcName}: each clip in "clips" must be CFR!')
    else:
        raise TypeError(f'{funcName}: "clips" must be a clip or a list of clips!')

    if cfr_ref:
        if not isinstance(cfr_ref, vs.VideoNode):
            raise TypeError(f'{funcName}: "cfr_ref" must be a clip!')
        if cfr_ref.fps.numerator == 0 and cfr_ref.fps.denominator == 1:
            raise ValueError(f'{funcName}: "cfr_ref" must be CFR!')

    T = TypeVar("T")
    def exclusive_accumulate(iterable: Iterable[T], func: Callable[[T, T], T] = operator.add,
                             initializer: Optional[T] = None
                             ) -> Iterable[T]:
        """Exclusive scan

        Make an iterator that returns accumulated results of binary functions,
        specified via the func argument.

        Examples:
            exclusive_accumulate([1,2,3,4,5], operator.add) --> 1 3 6 10
            exclusive_accumulate([1,2,3,4,5], operator.add, 0) --> 0 1 3 6 10
        """

        it = iter(iterable)

        if initializer is None:
            try:
                total = next(it)
            except StopIteration:
                return
        else:
            total = initializer

        while True:
            try:
                element = next(it)
            except StopIteration:
                return

            yield total

            total = func(total, element)


    # Timecode file
    if tcfile is not None:
        from collections import namedtuple
        from functools import reduce

        # Get timecode v1 generator

        # record of timecodes of each clip in clips
        tc_record = namedtuple("tc_record", ["start", "end", "fps"])
        raw_record_gen = (tc_record(0, clip.num_frames - 1, clip.fps) for clip in clips)

        # raw timecode v1 generator for spliced clip
        record_push = lambda x, y: tc_record(x.end + 1, x.end + y.end + 1, y.fps) # requires y.start == 0
        raw_tc_gen = itertools.accumulate(raw_record_gen, record_push)

        # simplified timecode v1 generator for spliced clip (merges successive records with equal fps)
        record_concat = lambda x, y: tc_record(x.start, y.end, y.fps) # requires x.fps == y.fps and x.end + 1 == y.start
        tc_gen = (reduce(record_concat, g) for (_, g) in itertools.groupby(raw_tc_gen, lambda x: x.fps)) # type: ignore

        # tc_list = list(tc_gen) # result is equal to mvsfunc's counterpart


        # Write to timecode file

        # fraction to str function
        frac2str = lambda frac, precision: f"{float(frac):<.{precision}F}" # type: Callable[[numbers.Real, int], str]

        # write
        if v2: # timecode v2
            # fps to milesecond function
            fps2ms = lambda fps: 1000 / fps # type: Callable[[numbers.Real], numbers.Real] # type: ignore

            # exclusive scan to calculate time stamps
            ms_gen = (fps2ms(tc.fps) for tc in tc_gen for _ in range(tc.start, tc.end + 1))
            time_gen = exclusive_accumulate(ms_gen, initializer=0)

            # time to string function
            time2strln = lambda time, precision: "{time}\n".format(time=frac2str(time, precision)) # type: Callable[[numbers.Real, int], str]

            olines = itertools.chain(
                ["# timecode format v2\n"],
                (time2strln(time, precision) for time in time_gen) # type: ignore
            )
        else: # timecode v1
            # timecode to string function
            tc2strln = lambda tc, precision: f"{tc.start},{tc.end},{frac2str(tc.fps, precision)}\n" # type: Callable[[tc_record, int], str]

            first_tc = next(tc_gen)

            olines = itertools.chain(
                ["# timecode format v1\n"],
                ["Assume {fps}\n".format(fps=frac2str(first_tc.fps, precision))],
                [tc2strln(first_tc, precision)],
                (tc2strln(tc, precision) for tc in tc_gen)
            )

        dirname = os.path.dirname(tcfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(tcfile, 'w') as ofile:
            ofile.writelines(olines)

    # Output spliced clip
    output = core.std.Splice(clips, mismatch=True)

    # For CFR output.
    if cfr_ref:
        clipa = output
        clipb = cfr_ref
        # fpn = clipa.num_frames * clipb.fps.numerator
        # fpd = clipb.fps.denominator * clipb.num_frames
        fpn = round(clipa.num_frames * clipb.fps.numerator / clipb.num_frames)
        fpd = clipb.fps.denominator
        return core.std.AssumeFPS(output, fpsnum=fpn, fpsden=fpd)
    else:
        return output


def MSR(clip: vs.VideoNode, *passes: numbers.Real, radius: int = 1, planes: PlanesType = None) -> vs.VideoNode:
    """Multiscale Retinex

    Multiscale retinex is a local contrast enhancement algorithm,
    which could be useful as a pre-processing step for texture detection.

    The output will always be in floating-point format.

    Args:
        clip: Input clip.

        passes: (int, args) List of passes for box filtering.

        radius: (int) Radius of box filtering.
            Default is 1.

        planes: (int []) Specify which planes to process.
            Unprocessed planes will be copied from the first clip "clip".
            By default, all planes will be processed.

    Examples:
        mask1 = muf.MSR(clip, 3).tcanny.TCanny(mode=1)
        mask2 = muf.MSR(clip, 2, 7, 15).tcanny.TCanny(mode=1)

    Ref:
        [1] Petro, A. B., Sbert, C., & Morel, J. M. (2014). Multiscale Retinex. Image Processing On Line, 71-88.
    """

    funcName = "MSR"

    # check
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    assert 1 <= len(passes) <= 25

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    in_format = clip.format
    query_video_format = core.query_video_format if _is_api4 else core.register_format
    out_format = query_video_format(
        color_family=in_format.color_family,
        sample_type=vs.FLOAT,
        bits_per_sample=32,
        subsampling_w=in_format.subsampling_w,
        subsampling_h=in_format.subsampling_h
    )

    # process
    passes = sorted(passes) # type: ignore

    pass_diffs = [(y - x) for x, y in zip([0] + passes[:-1], passes)] # type: ignore

    def blur(clip: vs.VideoNode, num_pass: int) -> vs.VideoNode:
        if clip.format.sample_type == vs.INTEGER and radius == 1:
            for _ in range(num_pass):
                clip = clip.rgvs.RemoveGrain([(20 if i in planes else 0) for i in range(clip.format.num_planes)]) # type: ignore
            return clip
        else:
            return clip.std.BoxBlur(hpasses=num_pass, hradius=radius, vpasses=num_pass, vradius=radius, planes=planes)

    flts = list(itertools.accumulate([clip] + pass_diffs, blur)) # type: ignore
    # flts = list(itertools.accumulate(sigma_diffs, blur, initial=clip)) # Python 3.8 is required

    var = (chr((i + ord('y') - ord('a')) % 26 + ord('a')) for i in range(len(passes)))
    expr = 'x'
    expr += f" {next(var)} "
    expr += ''.join(f" {v} *" for v in var)
    if len(passes) > 1:
        expr += f" {1/len(passes)} pow"
    expr += " 0.00001 + / log"

    return core.std.Expr(flts, [(expr if i in planes else '') for i in range(clip.format.num_planes)], format=out_format) # type: ignore


def arange(start, stop, step=1):
    current = start
    while current < stop:
        yield current
        current += step


class rescale:
    """Auxilary class to do descale and rescale with fractional source height.

    You can easily do descale and upscale with muf.rescale.Rescaler objects:

        rescaler = muf.rescale.Bilinear()  # get a muf.rescale.Rescaler object
        descaled_clip = rescaler.descale(clip, 1280, 720)
        rescaled_clip = rescaler.upscale(descaled_clip, 1920, 1080)

    You can also do descale with fractional source height:

        rescaler = muf.rescale.Bilinear()  # get a muf.rescale.Rescaler object
        descaled_clip = rescaler.descale(clip, 1270.6, 714.7, base_height=720)
        rescaled_clip = rescaler.upscale(descaled_clip, 1920, 1080)  # rescaler will handle scale args to correctly rescale the clip

    You can directly generate a rescaled clip:

        rescaler = muf.rescale.Bilinear()  # get a muf.rescale.Rescaler object
        # we assume that descaled clip has the SAME aspect ratio as source clip, so you don't need to pass src_width to rescale()
        rescaled_clip = rescaler.rescale(clip, src_height=714.7, base_height=720)

    Also, you can rescale the clip with other scalers you want such as nnedi3_resample:

        from nnedi3_resample import nnedi3_resample
        rescaler = muf.rescale.Bilinear()  # get a muf.rescale.Rescaler object
        rescaled_clip = rescaler.rescale(clip, 714.7, 720, upscaler=nnedi3_resample)

    And you can directly call the Rescaler to ignore the base_height argument:

        from nnedi3_resample import nnedi3_resample
        rescaler = muf.rescale.Bilinear()  # get a muf.rescale.Rescaler object
        rescaled_clip = rescaler(clip, 714.7, upscaler=nnedi3_resample)

    """

    @staticmethod
    def _get_descale_args(W: int, H: int, width: Union[float, int], height: Union[float, int], base_height: int = None):
        if base_height is None:
            width, height = round(width), round(height)
            src_width, src_height = width, height
            src_left, src_top = 0, 0
        else:
            base_width = round(W / H * base_height)
            src_width = width
            src_height = height
            width = base_width - 2 * int((base_width - width) / 2)
            height = base_height - 2 * int((base_height - height) / 2)
            src_top = (height - src_height) / 2
            src_left = (width - src_width) / 2

        return {
            "width": width,
            "height": height,
            "src_left": src_left,
            "src_top": src_top,
            "src_width": src_width,
            "src_height": src_height,
        }

    @staticmethod
    def _get_descale_args_pro(width: Union[float, int], height: Union[float, int], base_height: int = None, base_width: int = None):
        if base_height is None:
            height = round(height)
            src_height = height
            src_top = 0
        else:
            src_height = height
            height = base_height - 2 * int((base_height - height) / 2)
            src_top = (height - src_height) / 2

        if base_width is None:
            width = round(width)
            src_width = width
            src_left= 0
        else:
            src_width = width
            width = base_width - 2 * int((base_width - width) / 2)
            src_left = (width - src_width) / 2
        return {
            "width": width,
            "height": height,
            "src_left": src_left,
            "src_top": src_top,
            "src_width": src_width,
            "src_height": src_height,
        }

    @staticmethod
    def Upscale(clip: vs.VideoNode, width: int, height: int, kernel: str = "bicubic", taps: int = 3, b: float = 0.0, c: float = 0.5,
        src_left: float = None, src_top: float = None, src_width: float = None, src_height: float = None) -> vs.VideoNode:
        upscaler = getattr(core.resize, kernel.capitalize())
        if kernel.lower() == "bicubic":
            upscaler = functools.partial(upscaler, filter_param_a=b, filter_param_b=c)
        elif kernel.lower() == "lanczos":
            upscaler = functools.partial(upscaler, filter_param_a=taps)

        return upscaler(clip, width, height, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height)

    class Rescaler:
        def __init__(self, kernel: str = "bicubic", taps: int = 3, b: float = 0.0, c: float = 0.5, upscaler: Callable = None):
            kernel = kernel.lower()
            self.kernel = kernel
            self.taps, self.b, self.c = taps, b, c
            self.upscaler = upscaler
            bc_name = f"_{float(b):.3}_{float(c):.3}" if kernel == "bicubic" else ""
            taps_name = f"{taps}" if kernel == "lanczos" else ""
            self.name = f"{kernel}{bc_name}{taps_name}"
            self.descale_args = {}

        def __call__(self, clip: vs.VideoNode, src_height: Union[int, float], upscaler: Optional[Callable] = None) -> Any:
            base_height = clip.height if isinstance(src_height, float) else None
            return self.rescale(clip, src_height, base_height, upscaler)

        def rescale(self, clip: vs.VideoNode, src_height: Union[int, float], base_height: Optional[int] = None, upscaler: Optional[Callable] = None) -> vs.VideoNode:
            W, H = clip.width, clip.height
            src_width = W / H * src_height
            descaled = self.descale(clip, src_width, src_height, base_height)
            rescaled = self.upscale(descaled, W, H, upscaler)
            return rescaled

        def rescale_pro(self, clip: vs.VideoNode, src_width: Union[int, float] = None, src_height: Union[int, float] = None, base_width: Optional[int] = None, base_height: Optional[int] = None, upscaler: Optional[Callable] = None) -> vs.VideoNode:

            if ((src_height is None) and (src_width is None)):
                raise TypeError("At least one of the 'src_height' and 'src_width' must be set.")

            descaled = self.descale_pro(clip, src_width, src_height, base_width, base_height)
            rescaled = self.upscale(descaled, clip.width, clip.height, upscaler)
            return rescaled

        def descale(self, clip: vs.VideoNode, width: Union[int, float], height: Union[int, float], base_height: int = None):
            W, H = clip.width, clip.height
            self.descale_args = rescale._get_descale_args(W, H, width, height, base_height)
            kwargs = self.descale_args.copy()
            return core.descale.Descale(clip, kernel=self.kernel, taps=self.taps, b=self.b, c=self.c, **kwargs)

        def descale_pro(self, clip: vs.VideoNode, width: Union[int, float] = None, height: Union[int, float] = None, base_width: int = None, base_height: int = None):
            if width is None:
                width = clip.width
            if height is None:
                height = clip.height
            self.descale_args = rescale._get_descale_args_pro(width, height, base_height, base_width)
            kwargs = self.descale_args.copy()
            return core.descale.Descale(clip, kernel=self.kernel, taps=self.taps, b=self.b, c=self.c, **kwargs)

        def upscale(self, clip: vs.VideoNode, width: int, height: int, upscaler: Optional[Callable] = None) -> vs.VideoNode:
            from inspect import signature
            kwargs = self.descale_args.copy()
            kwargs.pop("width")
            kwargs.pop("height")
            if upscaler is None:
                return rescale.Upscale(clip, width, height, kernel=self.kernel, taps=self.taps, b=self.b, c=self.c, **kwargs)
            else:
                param_dict = signature(upscaler).parameters
                if all(key in param_dict for key in ("src_left", "src_top", "src_width", "src_height")):
                    pass
                elif all(key in param_dict for key in ("sx", "sy", "sw", "sh")):
                    kwargs["sx"] = kwargs.pop("src_left")
                    kwargs["sy"] = kwargs.pop("src_top")
                    kwargs["sw"] = kwargs.pop("src_width")
                    kwargs["sh"] = kwargs.pop("src_height")
                else:
                    raise TypeError("Your upscaler must have resize-like (src_left, src_width) or fmtc-like (sx, sw) argument names")
                return upscaler(clip, width, height, **kwargs)

    @staticmethod
    def Bilinear():
        return rescale.Rescaler(kernel="bilinear")

    @staticmethod
    def Bicubic(b: float = 0.0, c: float = 0.5):
        return rescale.Rescaler(kernel="bicubic", b=b, c=c)

    @staticmethod
    def Lanczos(taps: int = 3):
        return rescale.Rescaler(kernel="lanczos", taps=taps)

    @staticmethod
    def Spline16():
        return rescale.Rescaler(kernel="spline16")

    @staticmethod
    def Spline36():
        return rescale.Rescaler(kernel="spline36")

    @staticmethod
    def Spline64():
        return rescale.Rescaler(kernel="spline64")


def measurediff(
    clip1: vs.VideoNode,
    clip2: vs.VideoNode,
    norm_order: int = 1,
    propname: str = "PlaneDiffMeasure",
    ex_thr: float = 0.015,
    crop_size: int = 5
) -> vs.VideoNode:
    '''
    Calculate the p-norm for matrix of each frame of |clip1-clip2| and store the result in the frameprop.
    The larger the index "norm_order" is, the more significantly the high-level errors play a role in the result.

    Args:
        clip1, clip2: Input clip, vs.GRAYS.

        norm_order: (positive int) The index p of p-norm for matrix. Default is 1.

        propname: (str) The name of Frameprop the result will be saved in.

        crop_size: (int) Range of pixels around the border to be excluded in calculation.
            Default is 5.

        ex_thr: (float) Threshold for excluding little difference in calculation
            Default is 0.015.

    Return: The same clip as clip1 with the frameprop named in "propname".
    '''

    assert isinstance(norm_order, int) and norm_order > 0
    assert isinstance(clip1, vs.VideoNode) and clip1.format.id == vs.GRAYS
    assert isinstance(clip2, vs.VideoNode) and clip2.format.id == vs.GRAYS

    diff_moment = core.std.Expr([clip1, clip2], [f"x y - abs dup {ex_thr} > swap {norm_order} pow 0 ?"])

    if crop_size > 0:
        diff_moment = core.std.CropRel(diff_moment, *([crop_size] * 4))

    diff_moment = core.std.PlaneStats(diff_moment)

    try:
        diff_moment = core.akarin.PropExpr(diff_moment, lambda: {propname: f"x.PlaneStatsAverage {1 / norm_order} pow"})
        return core.std.CopyFrameProps(clip1, diff_moment, propname)
    except Exception:
        def calc_norm(n: int, f: vs.VideoFrame):
            fout = f.copy()
            fout.props[propname] = f.props["PlaneStatsAverage"] ** (1 / norm_order)
            return fout
        return core.std.CopyFrameProps(clip1, core.std.ModifyFrame(clip=diff_moment, clips=diff_moment, selector=calc_norm), propname)


def getnative(
    clip: vs.VideoNode,
    rescalers: Union[rescale.Rescaler, List[rescale.Rescaler]] = [rescale.Bicubic(0, 0.5)],
    src_heights: Union[int, float, Sequence[int], Sequence[float]] = tuple(range(500, 1001)),
    base_height: int = None,
    crop_size: int = 5,
    rt_eval: bool = True,
    dark: bool = True,
    ex_thr: float = 0.015,
    filename: str = None,
    vertical_only: bool = False,
    stats_func: Optional[Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode]] = None,
    stats_prop: str = "PlaneDiffMeasure"
) -> vs.VideoNode:
    """Find the native resolution(s) of upscaled material (mostly anime)

    Modifyed from:
        [getnative 1.3.0](https://github.com/Infiziert90/getnative/tree/ea08405f34a23dc681ff38a45e840ca21379a14d)
        [descale_verify](https://github.com/himesaka-noa/descale-verifier/blob/master/descale_verify.py)
        [getfnative](https://github.com/YomikoR/GetFnative/blob/main/getfnative.py)

    The function has 3 modes: verify, multi heights and multi kernels.
    They can be enabled by passing multi-frame clip, multi rescalers and multi src_heights to the function.
    For more details see Examples section below.

    The result is generated after all frames have been evaluated, which can be done through vspipe or "benchmark" of vsedit.

    Args:
        clip: Input clip, vs.GRAYS.

        rescalers: (rescale.Rescaler []) Sequence of resizers to be evaluated. Should be wrapped as muf.rescale.Rescaler
            Default is [muf.rescaler.Bicubic(0, 0.5)].
                Functions in muf.rescale such as muf.rescale.Bicubic() might help you to get a Rescaler.

        src_heights: (int|float []) Sequence of heights to be evaluated.
            Default is [500 ... 1000].
                muf.arange(start[, stop, step]) might help you to construct such a sequence.

        base_height: (int) The real integer height before cropping. If not None, fractional resolution will be evaluated.
            Default is None.

        crop_size: (int) Range of pixels around the border to be excluded in calculation.
            Default is 5.

        rt_eval: (bool) Whether to build the processing graph in runtime to reduce overhead.
            Default is True.

        dark: (bool) Whether to use dark background in output png file.
            Default is True

        ex_thr: (float) Threshold for excluding little difference in calculation
            Default is 0.015.

        filename: (str) The filename of the output file.
            Default is None.

        vertical_only: (bool) Only scale the frame in the vertical direction.
            Default is False

        stats_func: (function) Function that computes the metric between the source image and the rescaled image.
            The value is stored in a frame property specified by "stats_prop".
            Default is None.

        stats_prop: (str) Property name that "stats_func" writes to.
            Default is "PlaneDiffMeasure".

    Examples:
        Assume that src is a one-plane GRAYS clip. You might get such a clip by

            src = src.std.ShufflePlanes(0, vs.GRAY).resize.Point(format=vs.GRAYS)

        Compare between different integer source heights:

            clip = src[1000]  # to get a single frame clip

            # evaluate 500, 501, 502, ..., 999, 1000 as source heights with bicubic(b=1/3, c=1/3) as kernel.
            res = muf.getnative(clip, rescalers=muf.rescale.Bicubic(1/3, 1/3), src_heights=muf.arange(500, 1001, 1))
            res.set_output()

        Compare between different fractional source heights:

            clip = src[1000]  # to get a single frame clip

            # evaluate 800.0, 800.1, 800.2, ..., 899.8, 899.9 as source heights with bicubic(b=1/3, c=1/3) as kernel
            # base_height here must be a interger larger than any of src_heights
            res = muf.getnative(clip, rescalers=muf.rescale.Bicubic(), src_heights=muf.arange(800, 900, 0.1), base_height=900)
            res.set_output()

        Compare between different descale kernels:

            clip = src[1000]  # to get a single frame clip

            # construct a list of Rescaler
            # https://github.com/LittlePox/getnative/tree/f2fef4a5ebbed3cf88e972c14693b75102a0ee29
            from muf import rescale
            rescalers = [
                rescale.Bilinear(), rescale.Bicubic(1/3, 1/3), rescale.Bicubic(0, 0.5), rescale.Bicubic(0.5, 0.5),
                rescale.Lanczos(2), rescale.Lanczos(3), rescale.Lanczos(4),
                rescale.Spline16(), rescale.Spline36()
            ]

            # evaluate 714.7 as source height with bilinear, bicubic(b=1/3, c=1/3), ..., spline36 as kernels
            res = muf.getnative(clip, rescalers=rescalers, src_heights=714.7, base_height=720)
            res.set_output()

        Verify if a source height and a kernel can descale the whole clip well:

            clip = src  # clip is a multi frame clip

            # evaluate 714.7 as source height with bicubic(b=1/3, c=1/3) as kernel for every frame in clip
            res = muf.getnative(clip, rescalers=muf.rescale.Bicubic(1/3, 1/3), src_heights=714.7, base_height=720)
            res.set_output()

        Customizing stats_func with measurediff:
            from functools import partial
            rescalers = muf.rescale.Bicubic(1/3, 1/3)
            src_heights = muf.arange(500, 1001, 1)
            stats_func = partial(muf.measurediff, norm_order=2)
            res = muf.getnative(clip, rescalers=rescalers, src_heights=src_heights, stats_func=stats_func)
            res.set_output()

        Customizing stats_func with SSIM (note that 1 means no diff):
            from functools import partial
            kwargs = dict(
                rescalers=muf.rescale.Bicubic(1/3, 1/3),
                src_heights=muf.arange(500, 1001, 1),
                stats_func=muf.SSIM,
                stats_prop="PlaneSSIM"
            )
            res = muf.getnative(clip, **kwargs)
            res.set_output()

    Requirments:
        descale, matplotlib
    """

    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    import numpy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from enum import Enum

    class Mode(Enum):
        MULTI_FRAME = 1
        MULTI_HEIGHT = 2
        MULTI_KERNEL = 3

    # check
    assert isinstance(clip, vs.VideoNode) and clip.format.id == vs.GRAYS
    assert isinstance(base_height, int) or base_height is None

    if not isinstance(rescalers, list):
        rescalers = [rescalers]
    for rescaler in rescalers:
        assert isinstance(rescaler, rescale.Rescaler)

    if isinstance(src_heights, int) or isinstance(src_heights, float):
        src_heights = (src_heights,)
    if not isinstance(src_heights, tuple):
        src_heights = tuple(src_heights)
    if base_height is not None:
        assert base_height > max(src_heights)

    if stats_func is None:
        stats_func = functools.partial(measurediff, ex_thr=ex_thr, crop_size=crop_size)

    if clip.num_frames > 1:
        mode = Mode.MULTI_FRAME
        assert len(src_heights) == 1 and len(rescalers) == 1, "1 src_height and 1 rescaler should be passed for verify mode."
    elif len(src_heights) > 1:
        mode = Mode.MULTI_HEIGHT
        assert clip.num_frames == 1 and len(rescalers) == 1, "1-frame clip and 1 rescaler should be passed for multi heights mode."
    elif len(rescalers) > 1:
        mode = Mode.MULTI_KERNEL
        assert clip.num_frames == 1 and len(src_heights) == 1, "1-frame clip and 1 src_height shoule be passed for multi kernels mode."

    def output_statistics(clip: vs.VideoNode, rescalers: List[rescale.Rescaler], src_heights: Sequence[int], mode: Mode, dark: bool) -> vs.VideoNode:
        data = [0] * clip.num_frames
        remaining_frames = [1] * clip.num_frames # mutable

        def func_core(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
            # add eps to avoid getting 0 diff, which later messes up the graph.
            data[n] = f.props[stats_prop] + 1e-9 # type: ignore

            nonlocal remaining_frames
            remaining_frames[n] = 0

            if sum(remaining_frames) == 0:
                create_plot(data, rescalers, src_heights, mode, dark)

            return clip

        return core.std.FrameEval(clip, functools.partial(func_core, clip=clip), clip)

    def create_plot(data: Sequence[float], rescalers: List[rescale.Rescaler], src_heights: Sequence[float], mode: Mode, dark: bool) -> None:
        def get_heights_ticks(data: Sequence[float], src_heights: Sequence[float]) -> Sequence[float]:
            interval = round((max(src_heights) - min(src_heights)) * 0.05)
            log10_data = [math.log10(v) for v in data]
            d2_log10_data = []
            valley_heights = []
            for i in range(1, len(data) - 1):
                if log10_data[i - 1] > log10_data[i] and log10_data[i + 1] > log10_data[i]:
                    d2_log10_data.append(log10_data[i - 1] + log10_data[i + 1] - 2 * log10_data[i])
                    valley_heights.append(src_heights[i])
            candidate_heights = [valley_heights[i] for _, i in sorted(zip(d2_log10_data, range(len(valley_heights))), reverse=True)]
            candidate_heights.append(src_heights[0])
            candidate_heights.append(src_heights[-1])
            ticks = []
            for height in candidate_heights:
                for tick in ticks:
                    if abs(height - tick) < interval:
                        break
                else:
                    ticks.append(height)
            return ticks
        def get_kernel_ticks(data: Sequence[float], kernels: List[str]) -> tuple[List[int], List[str]]:
            interval = round(len(kernels) * 0.05)
            log10_data = [math.log10(v) for v in data]
            d2_log10_data = []
            valley_kernels = []
            for i in range(1, len(data) - 1):
                if log10_data[i - 1] > log10_data[i] and log10_data[i + 1] > log10_data[i]:
                    d2_log10_data.append(log10_data[i - 1] + log10_data[i + 1] - 2 * log10_data[i])
                    valley_kernels.append(kernels[i])
            candidate_kernels = [valley_kernels[i] for _, i in sorted(zip(d2_log10_data, range(len(valley_kernels))), reverse=True)]
            candidate_kernels.append(kernels[0])
            candidate_kernels.append(kernels[-1])
            ticks = []
            ticklabels = []
            for kernel in candidate_kernels:
                for ticklabel in ticklabels:
                    if abs(kernels.index(kernel) - kernels.index(ticklabel)) < interval:
                        break
                else:
                    ticks.append(kernels.index(kernel))
                    ticklabels.append(kernel)
            return ticks, ticklabels
        if dark:
            plt.style.use("dark_background")
            fmt = ".w-"
        else:
            fmt = ".-"
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.set_tight_layout(True)
        if mode == Mode.MULTI_FRAME:
            save_filename = get_save_filename(f"verify_{rescalers[0].name}_{src_heights[0]}")
            ax.plot(range(len(data)), data, fmt)
            ax.set(xlabel="Frame", ylabel="Relative error", title=save_filename, yscale="log")
            with open(f"{save_filename}.txt", "w") as ftxt:
                import pprint
                pprint.pprint(list(enumerate(data)), stream=ftxt)
        if mode == Mode.MULTI_HEIGHT:
            save_filename = get_save_filename(f"height_{rescalers[0].name}")
            ax.plot(src_heights, data, fmt)
            ticks = get_heights_ticks(data, src_heights)
            ax.set(xlabel="Height", xticks=ticks, ylabel="Relative error", title=save_filename, yscale="log")
            with open(f"{save_filename}.txt", "w") as ftxt:
                import pprint
                pprint.pprint(list(zip(src_heights, data)), stream=ftxt)
        elif mode == Mode.MULTI_KERNEL:
            save_filename = get_save_filename(f"kernel_{src_heights[0]}")
            ax.plot(range(len(data)), data, fmt)
            ticks = list(range(len(data)))
            ticklabels = [rescaler.name for rescaler in rescalers]
            with open(f"{save_filename}.txt", "w") as ftxt:
                import pprint
                pprint.pprint(list(zip(ticklabels, data)), stream=ftxt)
            if len(data) >= 20:
                ticks, ticklabels = get_kernel_ticks(data, ticklabels)
            ticklabels = [ticklabel.replace('_', '\n') for ticklabel in ticklabels]
            ax.set(xlabel="Kernel", xticks=ticks, xticklabels=ticklabels, ylabel="Relative error", title=save_filename, yscale="log")
        fig.savefig(f"{save_filename}")
        plt.close()

    def get_save_filename(name: str):
        if filename is None:
            from datetime import datetime
            return f"{name}_{datetime.now().strftime('%H-%M-%S')}.png"
        else:
            return filename

    # process
    if mode == Mode.MULTI_FRAME:
        src_height = src_heights[0]
        rescaler = rescalers[0]
        if not vertical_only:
            rescaled = rescaler.rescale(clip, src_height, base_height)
        else:
            rescaled = rescaler.rescale_pro(clip, src_height=src_height, base_height=base_height)
    elif mode == Mode.MULTI_HEIGHT:
        rescaler = rescalers[0]
        if rt_eval:
            clip = core.std.Loop(clip, len(src_heights))
            if not vertical_only:
                rescaled = core.std.FrameEval(clip, lambda n, clip=clip: rescaler.rescale(clip, src_heights[n], base_height))  # type: ignore
            else:
                rescaled = core.std.FrameEval(clip, lambda n, clip=clip: rescaler.rescale_pro(clip, src_height = src_heights[n], base_height = base_height))  # type: ignore
        else:
            if not vertical_only:
                rescaled = core.std.Splice([rescaler.rescale(clip, src_height, base_height) for src_height in src_heights])  # type: ignore
            else:
                rescaled = core.std.Splice([rescaler.rescale_pro(clip, src_height = src_height, base_height = base_height) for src_height in src_heights])  # type: ignore
            clip = core.std.Loop(clip, len(src_heights))
    elif mode == Mode.MULTI_KERNEL:
        src_height = src_heights[0]
        if rt_eval:
            clip = core.std.Loop(clip, len(rescalers))
            if not vertical_only:
                rescaled = core.std.FrameEval(clip, lambda n, clip=clip: rescalers[n].rescale(clip, src_height, base_height))  # type: ignore
            else:
                rescaled = core.std.FrameEval(clip, lambda n, clip=clip: rescalers[n].rescale_pro(clip, src_height = src_height, base_height = base_height))  # type: ignore
        else:
            if not vertical_only:
                rescaled = core.std.Splice([rescaler.rescale(clip, src_height, base_height) for rescaler in rescalers])  # type: ignore
            else:
                rescaled = core.std.Splice([rescaler.rescale_pro(clip, src_height = src_height, base_height = base_height) for rescaler in rescalers])  # type: ignore
            clip = core.std.Loop(clip, len(rescalers))

    stats = stats_func(clip, rescaled)

    return output_statistics(stats, rescalers, src_heights, mode, dark)

# port from fmtconv by Firesledge
class ResampleKernel:
    @staticmethod
    def bilinear() -> Tuple[Callable[[float], float], int]:
        def contributions(x: float) -> float:
            return max(1 - abs(x), 0)

        return contributions, 1

    @staticmethod
    def bicubic(b: float = 1/3, c: float = 1/3) -> Tuple[Callable[[float], float], int]:
        p0 = (6 - 2 * b) / 6
        p2 = (-18 + 12 * b + 6 * c) / 6
        p3 = (12 - 9 * b - 6 * c) / 6
        q0 = (8 * b + 24 * c) / 6
        q1 = (-12 * b - 48 * c) / 6
        q2 = (6 * b + 30 * c) / 6
        q3 = (-b - 6 * c) / 6

        def contributions(x: float) -> float:
            x = abs(x)
            if x <= 1:
                return p0 + x * x * (p2 + x * p3)
            elif x <= 2:
                return q0 + x * (q1 + x * (q2 + x * q3))
            else:
                return 0.0

        return contributions, 2

    @staticmethod
    def _sinc_function(x: float) -> float:
        if x == 0:
            return 1
        else:
            xp = x * math.pi
            return math.sin(xp) / xp

    @classmethod
    def lanczos(cls, taps: int = 4) -> Tuple[Callable[[float], float], int]:
        assert taps >= 1

        def contributions(x: float) -> float:
            if abs(x) <= taps:
                return cls._sinc_function(x) * cls._sinc_function(x / taps)
            else:
                return 0.0

        return contributions, taps

    @staticmethod
    def spline16() -> Tuple[Callable[[float], float], int]:
        def contributions(x: float) -> float:
            x = abs(x)

            if x <= 1:
                return ((x - 9/5) * x - 1/5) * x + 1
            elif x <= 2:
                x -= 1
                return ((-1/3 * x + 4/5) * x - 7/15) * x
            else:
                return 0.0

        return contributions, 2

    @staticmethod
    def spline36() -> Tuple[Callable[[float], float], int]:
        def contributions(x: float) -> float:
            x = abs(x)

            if x <= 1:
                return ((13/11 * x - 453/209) * x - 3/209) * x + 1
            elif x <= 2:
                x -= 1
                return ((-6/11 * x + 270/209) * x - 156/209) * x
            elif x <= 3:
                x -= 2
                return ((1/11 * x - 45/209) * x + 26/209) * x
            else:
                return 0.0

        return contributions, 3

    @staticmethod
    def spline64() -> Tuple[Callable[[float], float], int]:
        def contributions(x: float) -> float:
            x = abs(x)

            if x <= 1:
                return ((49/41 * x - 6387/2911) * x - 3/2911) * x + 1
            elif x <= 2:
                x -= 1
                return ((-24/41 * x + 4032/2911) * x - 2328/2911) * x
            elif x <= 3:
                x -= 2
                return ((6/41 * x - 1008/2911) * x + 582/2911) * x
            elif x <= 4:
                x -= 3
                return ((-1/41 * x + 168/2911) * x - 97/2911) * x
            else:
                return 0.0

        return contributions, 4

    @staticmethod
    def gauss(p: float = 30.0, taps: int = 4) -> Tuple[Callable[[float], float], int]:
        assert taps >= 1

        p = max(1, min(p, 100)) / 10

        def contributions(x: float) -> float:
            if abs(x) <= taps:
                return 2 ** (-p * x * x)
            else:
                return 0.0

        return contributions, taps

    @staticmethod
    def spline(taps: int = 4) -> Tuple[Callable[[float], float], int]:
        assert taps >= 1

        y = [0.0] * (2 * taps + 1)
        y[taps] = 1.0

        f = [0.0] * (2 * taps)
        if taps > 1:
            f[taps - 2] = 6.0
            f[taps] = 6.0
        f[taps - 1] = -12.0

        w = [4.0]
        z = [f[0] / w[0]]
        for j in range(1, 2 * taps):
            w.append(4 - 1 / w[j - 1])
            z.append((f[j] - z[j - 1]) / w[j])

        x = [0.0] * (2 * taps + 1)
        for j in range(2 * taps - 1, 0, -1):
            x[j] = z[j - 1] - x[j + 1] / w[j - 1]

        coef = [float(taps)]
        for j in range(taps, 2 * taps):
            p = 4 * (j - taps)
            coef.extend([
                (x[j+1] - x[j]) / 6,
                x[j] / 2,
                (y[j+1] - y[j]) - (x[j+1] + 2 * x[j]) / 6,
                y[j]
            ])

        def contributions(x: float) -> float:
            x = abs(x)
            if (p := int(x)) < taps:
                r = x - p
                return functools.reduce(lambda x, y: x * r + y, coef[4*p+1:4*p+5])
            else:
                return 0.0

        return contributions, taps

    @classmethod
    def sinc(cls, taps: int = 4) -> Tuple[Callable[[float], float], int]:
        assert taps >= 1

        def contributions(x: float) -> float:
            if abs(x) <= taps:
                return cls._sinc_function(x)
            else:
                return 0.0

        return contributions, taps

    @classmethod
    def blackman(cls, taps: int = 3) -> Tuple[Callable[[float], float], int]:
        assert taps >= 1

        def compute_win_coef(x: float) -> float:
            w_x = x * (math.pi / taps)
            return 0.42 + 0.50 * math.cos(w_x) + 0.08 * math.cos(w_x * 2)

        def contributions(x: float) -> float:
            if abs(x) <= taps:
                return cls._sinc_function(x) * compute_win_coef(x)
            else:
                return 0.0

        return contributions, taps

    @classmethod
    def blackmanminlobe(cls, taps: int = 3) -> Tuple[Callable[[float], float], int]:
        assert taps >= 1

        def compute_win_coef(x: float) -> float:
            w_x = x * (math.pi / taps)
            return (
                0.355768 + 0.487396 * math.cos(w_x) +
                0.144232 * math.cos(w_x * 2) + 0.012604 * math.cos(w_x * 3))

        def contributions(x: float) -> float:
            if abs(x) <= taps:
                return cls._sinc_function(x) * compute_win_coef(x)
            else:
                return 0.0

        return contributions, taps


def _downsample_helper(
    kernel_func: Callable[[float], float],
    support: int,
    kernel_width: int,
    kernel_scale: float,
    down_scale: float,
    shift: float
) -> Tuple[List[float], float]:

    shift_div, shift_rem = divmod(shift, 1)

    def mod(x: float) -> float:
        return (x + support) % (2 * support) - support

    if down_scale % 2 == 0:
        weights = [kernel_func(mod((i + 0.5 - shift_rem) * kernel_scale - support)) for i in range(kernel_width)]
        impulse = list(itertools.accumulate(weights[1:-1], lambda s, x: 2 * x - s, initial=2*weights[0]))
    else:
        weights = [kernel_func(mod((i - shift_rem) * kernel_scale - support)) for i in range(1, kernel_width)]
        impulse = weights

    return impulse, shift_div


class ResampleArgs(TypedDict):
    sx: float
    sy: float
    kernel: str
    impulseh: List[float]
    impulsev: List[float]
    kovrspl: int


def get_downsample_args(
    down_scale: int,
    kernel: str = "bicubic",
    taps: Optional[int] = None,
    a1: Optional[float] = None,
    a2: Optional[float] = None,
    sx: float = 0.0,
    sy: float = 0.0,
    antialiasing: bool = True,
    custom_kernel: Optional[Tuple[Callable[[float], float], int]] = None
) -> ResampleArgs:
    """ utility for downsample() """

    kovrspl = down_scale

    if taps is None:
        taps = 4

    if custom_kernel is not None:
        kernel_func, support = custom_kernel
    elif kernel == "bilinear":
        kernel_func, support = ResampleKernel.bilinear()
    elif kernel == "bicubic":
        if a1 is None:
            a1 = 1 / 3
        if a2 is None:
            a2 = 1 / 3
        kernel_func, support = ResampleKernel.bicubic(b=a1, c=a2)
    elif kernel == "lanczos":
        kernel_func, support = ResampleKernel.lanczos(taps=taps)
    elif kernel == "gauss":
        if a1 is None:
            a1 = 30.0
        kernel_func, support = ResampleKernel.gauss(p=a1, taps=taps)
    elif kernel == "spline16":
        kernel_func, support = ResampleKernel.spline16()
    elif kernel == "spline36":
        kernel_func, support = ResampleKernel.spline36()
    elif kernel == "spline64":
        kernel_func, support = ResampleKernel.spline64()
    elif kernel == "spline":
        kernel_func, support = ResampleKernel.spline(taps=taps)
    elif kernel == "sinc":
        kernel_func, support = ResampleKernel.sinc(taps=taps)
    elif kernel == "blackman":
        kernel_func, support = ResampleKernel.blackman(taps=taps)
    elif kernel == "blackmanminlobe":
        kernel_func, support = ResampleKernel.blackmanminlobe(taps=taps)
    else:
        raise ValueError(f"Unknown kernel {kernel}")

    if antialiasing:
        kernel_width = down_scale * support * 2
        kernel_scale = 1.0 / down_scale
    else:
        kernel_width = support * 2
        kernel_scale = 1.0

    impulseh, fmtc_sx = _downsample_helper(kernel_func, support, kernel_width, kernel_scale, down_scale, sx)
    impulsev, fmtc_sy = _downsample_helper(kernel_func, support, kernel_width, kernel_scale, down_scale, sy)

    return ResampleArgs(sx=fmtc_sx, sy=fmtc_sy, kernel="impulse", impulseh=impulseh, impulsev=impulsev, kovrspl=kovrspl)


def downsample(
    clip: vs.VideoNode,
    down_scale: int,
    kernel: str = "bicubic",
    taps: Optional[int] = None,
    a1: Optional[float] = None,
    a2: Optional[float] = None,
    sx: float = 0.0,
    sy: float = 0.0,
    antialiasing: bool = True,
    custom_kernel: Optional[Tuple[Callable[[float], float], int]] = None,
    **resample_kwargs
) -> vs.VideoNode:

    """ Integer-factor downsampling using fmtc.resample(kernel="impulse")

    Args:
        clip: Input clip.

        down_scale: (int) Downsample factor.
            Must be greater than 1 and be a common factor of clip's resolutions.

        kernel: (str) Downsample kernel. Possible values:
            bilinear, bicubic, lanczos, gauss, spline16, spline36, spline64,
            spline, sinc, blackman, blackmanminlobe

            Default is bicubic.

        taps: (int) Number of sample points.
            Default is 4.

        a1, a2, sx, sy: (float) Please refer to documentation of fmtc.resample().
            https://github.com/EleonoreMizo/fmtconv/blob/master/doc/fmtconv.html

        antialiasing: (bool) Whether to perform antialiasing when downsampling.
            fmtc.resample() implements "antialiasing=True".

            Default is True.

        custom_kernel: (kernel function, support) Override previous specification of downsample kernel.

    Warning:
        Subsampling is not handled.

    Example:
        # customizing SSIM_downsample()
        down_scale = 2
        assert gray.width % down_scale == 0 and gray.height % down_scale == 0

        w = gray.width // down_scale
        h = gray.height // down_scale
        kwargs = muvsfunc.get_downsample_args(down_scale=down_scale, antialiasing=False)

        res = muvsfunc.SSIM_downsample(gray, w, h, use_fmtc=True, **kwargs)
    """

    funcName = "downsample"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0:
        raise NotImplementedError(f'{funcName}: Subsampling is not handled!')

    if not isinstance(down_scale, int):
        raise TypeError(f'{funcName}: "down_scale" must be an int!')

    if down_scale <= 1:
        raise ValueError(f'{funcName}: "down_scale" must be greater than 1!')

    if clip.width % down_scale != 0 or clip.height % down_scale != 0:
        raise ValueError(f'{funcName}: "down_scale" is not a factor of video dimensions!')

    w = clip.width // down_scale
    h = clip.height // down_scale

    kwargs = get_downsample_args(
        down_scale=down_scale,
        kernel=kernel,
        taps=taps,
        a1=a1,
        a2=a2,
        sx=sx,
        sy=sy,
        antialiasing=antialiasing,
        custom_kernel=custom_kernel
    )

    return core.fmtc.resample(clip, w, h, **kwargs, **resample_kwargs)


def SSFDeband(
    clip: vs.VideoNode,
    thr: Union[float, Sequence[float]] = 1,
    smooth_taps: Union[int, Sequence[int]] = 2,
    edge_taps: Union[int, Sequence[int]] = 3,
    stride: Union[int, Sequence[int]] = 3,
    planes: PlanesType = None,
    ref: Optional[vs.VideoNode] = None
) -> vs.VideoNode:
    """Selective sparse filter debanding in VapourSynth

    Deband using a selective sparse filter which combines smooth region detection and banding reduction.

    Each plane is processed separately.

    Output may be slightly different from muvsfunc_numpy.SSFDeband() due to rounding error and boundary handling.

    Args:
        clip: Input clip.

        thr: (float or a list of floats) Threshold in (0, 255) of edge detection.
            Default is 1.

        smooth_taps: (int or a list of ints) Taps of the sparse filter, the larger, the smoother.
            Default is 2.

        edge_taps: (int or a list of ints) Taps of the edge detector, the larger, smaller region will be smoothed.
            Default is 3.

        strides: (int or a list of ints) The stride of the edge detection and filtering.
            Default is 3.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "clip".

        ref: (clip). Externally provided clip for banding detection, must be of the same shape as "clip".
            Default is None.

    Ref:
        [1] Song, Q., Su, G. M., & Cosman, P. C. (2016, September).
            Hardware-efficient debanding and visual enhancement filter for inverse tone mapped high dynamic range images and videos.
            In Image Processing (ICIP), 2016 IEEE International Conference on (pp. 3299-3303). IEEE.

    """

    funcName = "SSFDeband"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if ref is not None:
        if not isinstance(ref, vs.VideoNode):
            raise TypeError(f'{funcName}: "ref" must be a clip!')

        if ref.width != clip.width or ref.height != clip.height:
            raise TypeError(f'{funcName}: "ref" must be of the same size as "clip"!')

    def to_list(x):
        if isinstance(x, abc.Sequence):
            return list(x) + [x[-1]] * 2
        else:
            return [x] * 3

    thr = to_list(thr)
    smooth_taps = to_list(smooth_taps)
    edge_taps = to_list(edge_taps)
    stride = to_list(stride)

    if clip.format.sample_type == vs.INTEGER:
        thr = [t * ((2 ** clip.format.bits_per_sample) - 1) / 255 for t in thr]
    else:
        thr = [t / 255 for t in thr]

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    def get_expr(thr: float, smooth_taps: int, edge_taps: int, stride: int, is_vertical: bool, smooth_var: str):
        if thr <= 0 or smooth_taps <= 1 or edge_taps <= 0 or stride == 0:
            return ""

        isclose = lambda x, thr=thr: f"x {x} - abs {thr} <"

        generate = lambda taps, stride=stride, var="x": (
            (f"{var}[0,{i * stride}]" if is_vertical else f"{var}[{i * stride},0]")
            for i in range(-taps, taps+1)
        )

        mask_expr = functools.reduce(lambda x, y: f"{x} {y} and", map(isclose, generate(taps=edge_taps)))
        smooth_expr = functools.reduce(
            lambda x, y: f"{x} {y} +",
            generate(taps=smooth_taps, var=smooth_var)
        ) + f" {2 * smooth_taps + 1} /"
        return f"{mask_expr} {smooth_expr} {smooth_var} ?"

    vrt_exprs = [(
            get_expr(
                thr=thr[plane],
                smooth_taps=smooth_taps[plane],
                edge_taps=edge_taps[plane],
                stride=stride[plane],
                is_vertical=True,
                smooth_var="y" if ref is not None else "x"
            )
            if plane in planes else ""
        ) for plane in range(clip.format.num_planes)
    ]
    vrt_deband = core.akarin.Expr(([ref] if ref is not None else []) + [clip], vrt_exprs)

    hrz_exprs = [(
            get_expr(
                thr=thr[plane],
                smooth_taps=smooth_taps[plane],
                edge_taps=edge_taps[plane],
                stride=stride[plane],
                is_vertical=False,
                smooth_var="y" if ref is not None else "x"
            )
            if plane in planes else ""
        ) for plane in range(clip.format.num_planes)
    ]
    hrz_deband = core.akarin.Expr(([ref] if ref is not None else []) + [vrt_deband], hrz_exprs)

    return hrz_deband


#####################
## Toon v0.82 edit ##
#####################
#
# function created by mf
#   support by Soulhunter ;-)
#   ported to masktools v2 and optimized by Didee (0.82)
#   added parameters and smaller changes by MOmonster (0.82 edited)
#
# toon v0.8 is the newest light-weight build of mf´s nice line darken function mf_toon
#
# Parameters:
#  str (float) - Strength of the line darken. Default is 1.0
#  l_thr (int) - Lower threshold for the linemask. Default is 2
#  u_thr (int) - Upper threshold for the linemask. Default is 12
#  blur (int)  - "blur" parameter of AWarpSharp2. Default is 2
#  depth (int) - "depth" parameter of AWarpSharp2. Default is 32
def haf_Toon(input, str=1.0, l_thr=2, u_thr=12, blur=2, depth=32):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('Toon: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('Toon: RGB format is not supported')

    neutral = 1 << (input.format.bits_per_sample - 1)
    peak = (1 << input.format.bits_per_sample) - 1
    multiple = peak / 255

    if input.format.color_family != vs.GRAY:
        input_orig = input
        input = mvf.GetPlane(input, 0)
    else:
        input_orig = None

    lthr = neutral + scale(l_thr, peak)
    lthr8 = lthr / multiple
    uthr = neutral + scale(u_thr, peak)
    uthr8 = uthr / multiple
    ludiff = u_thr - l_thr

    last = core.std.MakeDiff(input.std.Maximum().std.Minimum(), input)
    last = core.std.Expr([last, haf_Padding(last, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)], expr=['x y min'])
    expr = f'y {lthr} <= {neutral} y {uthr} >= x {uthr8} y {multiple} / - 128 * x {multiple} / y {multiple} / {lthr8} - * + {ludiff} / {multiple} * ? {neutral} - {str} * {neutral} + ?'
    last = core.std.MakeDiff(input, core.std.Expr([last, last.std.Maximum()], expr=[expr]))

    if input_orig is not None:
        last = core.std.ShufflePlanes([last, input_orig], planes=[0, 1, 2], colorfamily=input_orig.format.color_family)
    return last


################################################################################################
###                                                                                          ###
###                       LimitedSharpenFaster MOD : function LSFmod()                       ###
###                                                                                          ###
###                                Modded Version by LaTo INV.                               ###
###                                                                                          ###
###                                  v1.9 - 05 October 2009                                  ###
###                                                                                          ###
################################################################################################
###
### +--------------+
### | DEPENDENCIES |
### +--------------+
###
### -> RGVS
### -> CAS
###
###
###
### +---------+
### | GENERAL |
### +---------+
###
### strength [int]
### --------------
### Strength of the sharpening
###
### Smode [int: 1,2,3]
### ----------------------
### Sharpen mode:
###    =1 : Range sharpening
###    =2 : Nonlinear sharpening (corrected version)
###    =3 : Contrast Adaptive Sharpening
###
### Smethod [int: 1,2,3]
### --------------------
### Sharpen method: (only used in Smode=1,2)
###    =1 : 3x3 kernel
###    =2 : Min/Max
###    =3 : Min/Max + 3x3 kernel
###
### kernel [int: 11,12,19,20]
### -------------------------
### Kernel used in Smethod=1&3
### In strength order: + 19 > 12 >> 20 > 11 -
###
###
###
### +---------+
### | SPECIAL |
### +---------+
###
### preblur [int: 0,1,2,3]
### --------------------------------
### Mode to avoid noise sharpening & ringing:
###    =-1 : No preblur
###    = 0 : MinBlur(0)
###    = 1 : MinBlur(1)
###    = 2 : MinBlur(2)
###    = 3 : DFTTest
###
### secure [bool]
### -------------
### Mode to avoid banding & oil painting (or face wax) effect of sharpening
###
### source [clip]
### -------------
### If source is defined, LSFmod doesn't sharp more a denoised clip than this source clip
### In this mode, you can safely set Lmode=0 & PP=off
###    Usage:   denoised.LSFmod(source=source)
###    Example: last.FFT3DFilter().LSFmod(source=last,Lmode=0,soft=0)
###
###
###
### +----------------------+
### | NONLINEAR SHARPENING |
### +----------------------+
###
### Szrp [int]
### ----------
### Zero Point:
###    - differences below Szrp are amplified (overdrive sharpening)
###    - differences above Szrp are reduced   (reduced sharpening)
###
### Spwr [int]
### ----------
### Power: exponent for sharpener
###
### SdmpLo [int]
### ------------
### Damp Low: reduce sharpening for small changes [0:disable]
###
### SdmpHi [int]
### ------------
### Damp High: reduce sharpening for big changes [0:disable]
###
###
###
### +----------+
### | LIMITING |
### +----------+
###
### Lmode [int: ...,0,1,2,3,4]
### --------------------------
### Limit mode:
###    <0 : Limit with repair (ex: Lmode=-1 --> repair(1), Lmode=-5 --> repair(5)...)
###    =0 : No limit
###    =1 : Limit to over/undershoot
###    =2 : Limit to over/undershoot on edges and no limit on not-edges
###    =3 : Limit to zero on edges and to over/undershoot on not-edges
###    =4 : Limit to over/undershoot on edges and to over/undershoot2 on not-edges
###
### overshoot [int]
### ---------------
### Limit for pixels that get brighter during sharpening
###
### undershoot [int]
### ----------------
### Limit for pixels that get darker during sharpening
###
### overshoot2 [int]
### ----------------
### Same as overshoot, only for Lmode=4
###
### undershoot2 [int]
### -----------------
### Same as undershoot, only for Lmode=4
###
###
###
### +-----------------+
### | POST-PROCESSING |
### +-----------------+
###
### soft [int: -2,-1,0...100]
### -------------------------
### Soft the sharpening effect (-1 = old autocalculate, -2 = new autocalculate)
###
### soothe [bool]
### -------------
###    =True  : Enable soothe temporal stabilization
###    =False : Disable soothe temporal stabilization
###
### keep [int: 0...100]
### -------------------
### Minimum percent of the original sharpening to keep (only with soothe=True)
###
###
###
### +-------+
### | EDGES |
### +-------+
###
### edgemode [int: -1,0,1,2]
### ------------------------
###    =-1 : Show edgemask
###    = 0 : Sharpening all
###    = 1 : Sharpening only edges
###    = 2 : Sharpening only not-edges
###
### edgemaskHQ [bool]
### -----------------
###    =True  : Original edgemask
###    =False : Faster edgemask
###
###
###
### +------------+
### | UPSAMPLING |
### +------------+
###
### ss_x ; ss_y [float]
### -------------------
### Supersampling factor (reduce aliasing on edges)
###
### dest_x ; dest_y [int]
### ---------------------
### Output resolution after sharpening (avoid a resizing step)
###
###
###
### +----------+
### | SETTINGS |
### +----------+
###
### defaults [string: "old" or "slow" or "fast"]
### --------------------------------------------
###    = "old"  : Reset settings to original version (output will be THE SAME AS LSF)
###    = "slow" : Enable SLOW modded version settings
###    = "fast" : Enable FAST modded version settings
###  --> /!\ [default:"fast"]
###
###
### defaults="old" :  - strength    = 100
### ----------------  - Smode       = 1
###                   - Smethod     = Smode==1?2:1
###                   - kernel      = 11
###
###                   - preblur     = -1
###                   - secure      = false
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 2
###                   - SdmpLo      = strength/25
###                   - SdmpHi      = 0
###
###                   - Lmode       = 1
###                   - overshoot   = 1
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = 0
###                   - soothe      = false
###                   - keep        = 25
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = true
###
###                   - ss_x        = Smode==1?1.50:1.25
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
###
### defaults="slow" : - strength    = 100
### ----------------- - Smode       = 2
###                   - Smethod     = 3
###                   - kernel      = 11
###
###                   - preblur     = -1
###                   - secure      = true
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 4
###                   - SdmpLo      = 4
###                   - SdmpHi      = 48
###
###                   - Lmode       = 4
###                   - overshoot   = strength/100
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = -2
###                   - soothe      = true
###                   - keep        = 20
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = true
###
###                   - ss_x        = Smode==3?1.00:1.50
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
###
### defaults="fast" : - strength    = 80
### ----------------- - Smode       = 3
###                   - Smethod     = 2
###                   - kernel      = 11
###
###                   - preblur     = 0
###                   - secure      = true
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 4
###                   - SdmpLo      = 4
###                   - SdmpHi      = 48
###
###                   - Lmode       = 0
###                   - overshoot   = strength/100
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = 0
###                   - soothe      = false
###                   - keep        = 20
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = false
###
###                   - ss_x        = Smode==3?1.00:1.25
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
################################################################################################
def haf_LSFmod(input, strength=None, Smode=None, Smethod=None, kernel=11, preblur=None, secure=None, source=None, Szrp=16, Spwr=None, SdmpLo=None, SdmpHi=None, Lmode=None, overshoot=None, undershoot=None,
           overshoot2=None, undershoot2=None, soft=None, soothe=None, keep=None, edgemode=0, edgemaskHQ=None, ss_x=None, ss_y=None, dest_x=None, dest_y=None, defaults='fast'):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('LSFmod: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('LSFmod: RGB format is not supported')

    if source is not None and (not isinstance(source, vs.VideoNode) or source.format.id != input.format.id):
        raise vs.Error("LSFmod: 'source' must be the same format as input")

    isGray = (input.format.color_family == vs.GRAY)
    isInteger = (input.format.sample_type == vs.INTEGER)

    if isInteger:
        neutral = 1 << (input.format.bits_per_sample - 1)
        peak = (1 << input.format.bits_per_sample) - 1
        factor = 1 << (input.format.bits_per_sample - 8)
    else:
        neutral = 0.0
        peak = 1.0
        factor = 255.0

    ### DEFAULTS
    try:
        num = ['old', 'slow', 'fast'].index(defaults.lower())
    except:
        raise vs.Error('LSFmod: defaults must be "old" or "slow" or "fast"')

    ox = input.width
    oy = input.height

    if strength is None:
        strength = [100, 100, 80][num]
    if Smode is None:
        Smode = [1, 2, 3][num]
    if Smethod is None:
        Smethod = [2 if Smode == 1 else 1, 3, 2][num]
    if preblur is None:
        preblur = [-1, -1, 0][num]
    if secure is None:
        secure = [False, True, True][num]
    if Spwr is None:
        Spwr = [2, 4, 4][num]
    if SdmpLo is None:
        SdmpLo = [strength // 25, 4, 4][num]
    if SdmpHi is None:
        SdmpHi = [0, 48, 48][num]
    if Lmode is None:
        Lmode = [1, 4, 0][num]
    if overshoot is None:
        overshoot = [1, strength // 100, strength // 100][num]
    if undershoot is None:
        undershoot = overshoot
    if overshoot2 is None:
        overshoot2 = overshoot * 2
    if undershoot2 is None:
        undershoot2 = overshoot2
    if soft is None:
        soft = [0, -2, 0][num]
    if soothe is None:
        soothe = [False, True, False][num]
    if keep is None:
        keep = [25, 20, 20][num]
    if edgemaskHQ is None:
        edgemaskHQ = [True, True, False][num]
    if ss_x is None:
        ss_x = [1.5 if Smode == 1 else 1.25, 1.0 if Smode == 3 else 1.5, 1.0 if Smode == 3 else 1.25][num]
    if ss_y is None:
        ss_y = ss_x
    if dest_x is None:
        dest_x = ox
    if dest_y is None:
        dest_y = oy

    if kernel == 4:
        RemoveGrain = functools.partial(core.std.Median)
    elif kernel in [11, 12]:
        RemoveGrain = functools.partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif kernel == 19:
        RemoveGrain = functools.partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif kernel == 20:
        RemoveGrain = functools.partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        RemoveGrain = functools.partial(core.rgvs.RemoveGrain, mode=[kernel])

    if soft == -1:
        soft = math.sqrt(((ss_x + ss_y) / 2 - 1) * 100) * 10
    elif soft <= -2:
        soft = int((1 + 2 / (ss_x + ss_y)) * math.sqrt(strength))
    soft = min(soft, 100)

    xxs = haf_cround(ox * ss_x / 8) * 8
    yys = haf_cround(oy * ss_y / 8) * 8

    Str = strength / 100

    ### SHARP
    if ss_x > 1 or ss_y > 1:
        tmp = input.resize.Spline36(xxs, yys)
    else:
        tmp = input

    if not isGray:
        tmp_orig = tmp
        tmp = mvf.GetPlane(tmp, 0)

    if preblur <= -1:
        pre = tmp
    elif preblur >= 3:
        expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=scale(16, peak), j=scale(75, peak), peak=peak)
        pre = core.std.MaskedMerge(tmp.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0]), tmp, tmp.std.Expr(expr=[expr]))
    else:
        pre = haf_MinBlur(tmp, r=preblur)

    dark_limit = pre.std.Minimum()
    bright_limit = pre.std.Maximum()

    if Smode < 3:
        if Smethod <= 1:
            method = RemoveGrain(pre)
        elif Smethod == 2:
            method = core.std.Merge(dark_limit, bright_limit)
        else:
            method = RemoveGrain(core.std.Merge(dark_limit, bright_limit))

        if secure:
            method = core.std.Expr([method, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale(1, peak))])

        if preblur > -1:
            method = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, method))

        if Smode <= 1:
            normsharp = core.std.Expr([tmp, method], expr=[f'x x y - {Str} * +'])
        else:
            tmpScaled = tmp.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'], format=tmp.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            methodScaled = method.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'], format=method.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            expr = f'x y = x x x y - abs {Szrp} / {1 / Spwr} pow {Szrp} * {Str} * x y - dup abs / * x y - dup * {Szrp * Szrp} {SdmpLo} + * x y - dup * {SdmpLo} + {Szrp * Szrp} * / * 1 {SdmpHi} 0 = 0 {(Szrp / SdmpHi) ** 4} ? + 1 {SdmpHi} 0 = 0 x y - abs {SdmpHi} / 4 pow ? + / * + ? {factor if isInteger else 1 / factor} *'
            normsharp = core.std.Expr([tmpScaled, methodScaled], expr=[expr], format=tmp.format)
    else:
        normsharp = pre.cas.CAS(sharpness=min(Str, 1))

        if secure:
            normsharp = core.std.Expr([normsharp, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale(1, peak))])

        if preblur > -1:
            normsharp = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, normsharp))

    ### LIMIT
    normal = haf_Clamp(normsharp, bright_limit, dark_limit, scale(overshoot, peak), scale(undershoot, peak))
    second = haf_Clamp(normsharp, bright_limit, dark_limit, scale(overshoot2, peak), scale(undershoot2, peak))
    zero = haf_Clamp(normsharp, bright_limit, dark_limit, 0, 0)

    if edgemaskHQ:
        edge = tmp.std.Sobel(scale=2)
    else:
        edge = core.std.Expr([tmp.std.Maximum(), tmp.std.Minimum()], expr=['x y -'])
    edge = edge.std.Expr(expr=[f'x {1 / factor if isInteger else factor} * {128 if edgemaskHQ else 32} / 0.86 pow 255 * {factor if isInteger else 1 / factor} *'])

    if Lmode < 0:
        limit1 = core.rgvs.Repair(normsharp, tmp, mode=[abs(Lmode)])
    elif Lmode == 0:
        limit1 = normsharp
    elif Lmode == 1:
        limit1 = normal
    elif Lmode == 2:
        limit1 = core.std.MaskedMerge(normsharp, normal, edge.std.Inflate())
    elif Lmode == 3:
        limit1 = core.std.MaskedMerge(normal, zero, edge.std.Inflate())
    else:
        limit1 = core.std.MaskedMerge(second, normal, edge.std.Inflate())

    if edgemode <= 0:
        limit2 = limit1
    elif edgemode == 1:
        limit2 = core.std.MaskedMerge(tmp, limit1, edge.std.Inflate().std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))
    else:
        limit2 = core.std.MaskedMerge(limit1, tmp, edge.std.Inflate().std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))

    ### SOFT
    if soft == 0:
        PP1 = limit2
    else:
        sharpdiff = core.std.MakeDiff(tmp, limit2)
        sharpdiff = core.std.Expr([sharpdiff, sharpdiff.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])], expr=[f'x {neutral} - abs y {neutral} - abs > y {soft} * x {100 - soft} * + 100 / x ?'])
        PP1 = core.std.MakeDiff(tmp, sharpdiff)

    ### SOOTHE
    if soothe:
        diff = core.std.MakeDiff(tmp, PP1)
        diff = core.std.Expr([diff, diff.focus2.TemporalSoften2(1, 255 << (input.format.bits_per_sample - 8), 0, 32, 2)],
                             expr=[f'x {neutral} - y {neutral} - * 0 < x {neutral} - 100 / {keep} * {neutral} + x {neutral} - abs y {neutral} - abs > x {keep} * y {100 - keep} * + 100 / x ? ?'])
        PP2 = core.std.MakeDiff(tmp, diff)
    else:
        PP2 = PP1

    ### OUTPUT
    if dest_x != ox or dest_y != oy:
        if not isGray:
            PP2 = core.std.ShufflePlanes([PP2, tmp_orig], planes=[0, 1, 2], colorfamily=input.format.color_family)
        out = PP2.resize.Spline36(dest_x, dest_y)
    elif ss_x > 1 or ss_y > 1:
        out = PP2.resize.Spline36(dest_x, dest_y)
        if not isGray:
            out = core.std.ShufflePlanes([out, input], planes=[0, 1, 2], colorfamily=input.format.color_family)
    elif not isGray:
        out = core.std.ShufflePlanes([PP2, input], planes=[0, 1, 2], colorfamily=input.format.color_family)
    else:
        out = PP2

    if edgemode <= -1:
        return edge.resize.Spline36(dest_x, dest_y, format=input.format)
    elif source is not None:
        if dest_x != ox or dest_y != oy:
            src = source.resize.Spline36(dest_x, dest_y)
            In = input.resize.Spline36(dest_x, dest_y)
        else:
            src = source
            In = input

        shrpD = core.std.MakeDiff(In, out, planes=[0])
        expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
        shrpL = core.std.Expr([core.rgvs.Repair(shrpD, core.std.MakeDiff(In, src, planes=[0]), mode=[1] if isGray else [1, 0]), shrpD], expr=[expr] if isGray else [expr, ''])
        return core.std.MakeDiff(In, shrpL, planes=[0])
    else:
        return out


def haf_ChangeFPS(clip: vs.VideoNode, fpsnum: int, fpsden: int = 1) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('ChangeFPS: this is not a clip')

    factor = fractions.Fraction(fpsnum, fpsden) / clip.fps

    def frame_adjuster(n):
        real_n = math.floor(n / factor)
        one_frame_clip = clip[real_n] * (len(clip) + 100)
        return one_frame_clip

    attribute_clip = clip.std.BlankClip(length=math.floor(len(clip) * factor), fpsnum=fpsnum, fpsden=fpsden)
    return attribute_clip.std.FrameEval(eval=frame_adjuster)


def haf_Clamp(clip, bright_limit, dark_limit, overshoot=0, undershoot=0, planes=None):
    if not (isinstance(clip, vs.VideoNode) and isinstance(bright_limit, vs.VideoNode) and isinstance(dark_limit, vs.VideoNode)):
        raise vs.Error('Clamp: this is not a clip')

    if bright_limit.format.id != clip.format.id or dark_limit.format.id != clip.format.id:
        raise vs.Error('Clamp: clips must have the same format')

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    expr = f'x y {overshoot} + > y {overshoot} + x ? z {undershoot} - < z {undershoot} - x y {overshoot} + > y {overshoot} + x ? ?'
    return core.std.Expr([clip, bright_limit, dark_limit], expr=[expr if i in planes else '' for i in range(clip.format.num_planes)])


def haf_TemporalSoften(clip, radius=4, luma_threshold=4, chroma_threshold=8, scenechange=15, mode=2):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('TemporalSoften: This is not a clip')

    if scenechange:
        clip = haf_set_scenechange(clip, scenechange)
    return core.focus2.TemporalSoften2(clip, radius, luma_threshold, chroma_threshold, scenechange)


def haf_set_scenechange(clip, thresh=15):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('set_scenechange: This is not a clip')

    def set_props(n, f):
        fout = f[0].copy()
        fout.props._SceneChangePrev = f[1].props._SceneChangePrev
        fout.props._SceneChangeNext = f[1].props._SceneChangeNext
        return fout

    sc = clip

    if clip.format.color_family == vs.RGB:
        sc = core.resize.Bicubic(clip, format=vs.GRAY16, matrix_s='709')
        if sc.format.bits_per_sample != clip.format.bits_per_sample:
            sc = core.fmtc.bitdepth(sc, bits=clip.format.bits_per_sample, dmode=1)

    sc = core.scd.Detect(sc, thresh)

    if clip.format.color_family == vs.RGB:
        sc = core.std.ModifyFrame(clip, clips=[clip, sc], selector=set_props)

    return sc


def haf_Padding(clip, left=0, right=0, top=0, bottom=0):
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    return clip.resize.Point(clip.width + left + right, clip.height + top + bottom, src_left=-left, src_top=-top, src_width=clip.width + left + right, src_height=clip.height + top + bottom)


def haf_SCDetect(clip, threshold=None):
    def copy_property(n, f):
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
        fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
        return fout

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')

    sc = clip
    if clip.format.color_family == vs.RGB:
        sc = clip.resize.Bicubic(format=vs.GRAY8, matrix_s='709')
    sc = sc.misc.SCDetect(threshold=threshold)

    if clip.format.color_family == vs.RGB:
        sc = clip.std.ModifyFrame(clips=[clip, sc], selector=copy_property)
    return sc


def haf_Weave(clip, tff):
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Weave: this is not a clip')

    return clip.std.DoubleWeave(tff=tff)[::2]


# MinBlur   by Didée (http://avisynth.nl/index.php/MinBlur)
# Nifty Gauss/Median combination
def haf_MinBlur(clp, r=1, planes=None):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: this is not a clip')

    if planes is None:
        planes = list(range(clp.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if r <= 0:
        RG11 = sbr(clp, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 1:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 2:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        RG4 = clp.ctmf.CTMF(radius=2, planes=planes)
    else:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        if clp.format.bits_per_sample == 16:
            s16 = clp
            RG4 = clp.fmtc.bitdepth(bits=12, planes=planes, dmode=1).ctmf.CTMF(radius=3, planes=planes).fmtc.bitdepth(bits=16, planes=planes)
            RG4 = mvf.LimitFilter(s16, RG4, thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    expr = 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
    return core.std.Expr([clp, RG11, RG4], expr=[expr if i in planes else '' for i in range(clp.format.num_planes)])


# make a highpass on a blur's difference (well, kind of that)
def sbr(c, r=1, planes=None):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbr: this is not a clip')

    neutral = 1 << (c.format.bits_per_sample - 1) if c.format.sample_type == vs.INTEGER else 0.0

    if planes is None:
        planes = list(range(c.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if r <= 1:
        RG11 = c.std.Convolution(matrix=matrix1, planes=planes)
    elif r == 2:
        RG11 = c.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
    else:
        RG11 = c.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    if r <= 1:
        RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes)
    elif r == 2:
        RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
    else:
        RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)

    expr = f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?'
    RG11DD = core.std.Expr([RG11D, RG11DS], expr=[expr if i in planes else '' for i in range(c.format.num_planes)])
    return core.std.MakeDiff(c, RG11DD, planes=planes)



########################################
## cretindesalpes' functions:

# Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)
# By courtesy of cretindesalpes. (http://forum.doom9.org/showthread.php?p=1548318#post1548318)
def haf_DitherLumaRebuild(src, s0=2.0, c=0.0625, chroma=True):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('DitherLumaRebuild: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('DitherLumaRebuild: RGB format is not supported')

    isGray = (src.format.color_family == vs.GRAY)
    isInteger = (src.format.sample_type == vs.INTEGER)

    shift = src.format.bits_per_sample - 8
    neutral = 128 << shift if isInteger else 0.0

    k = (s0 - 1) * c
    t = f'x {16 << shift if isInteger else 16 / 255} - {219 << shift if isInteger else 219 / 255} / 0 max 1 min'
    e = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + {256 << shift if isInteger else 256 / 255} *'
    return src.std.Expr(expr=[e] if isGray else [e, f'x {neutral} - 128 * 112 / {neutral} +' if chroma else ''])


#=============================================================================
#   mt_expand_multi
#   mt_inpand_multi
#
#   Calls mt_expand or mt_inpand multiple times in order to grow or shrink
#   the mask from the desired width and height.
#
#   Parameters:
#   - sw   : Growing/shrinking shape width. 0 is allowed. Default: 1
#   - sh   : Growing/shrinking shape height. 0 is allowed. Default: 1
#   - mode : "rectangle" (default), "ellipse" or "losange". Replaces the
#       mt_xxpand mode. Ellipses are actually combinations of
#       rectangles and losanges and look more like octogons.
#       Losanges are truncated (not scaled) when sw and sh are not
#       equal.
#   Other parameters are the same as mt_xxpand.
#=============================================================================
def haf_mt_expand_multi(src, mode='rectangle', planes=None, sw=1, sh=1):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_expand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = haf_mt_expand_multi(src.std.Maximum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src


def haf_mt_inpand_multi(src, mode='rectangle', planes=None, sw=1, sh=1):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inpand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = haf_mt_inpand_multi(src.std.Minimum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src


def haf_mt_inflate_multi(src: vs.VideoNode, planes: Optional[PlanesType] = None, radius: int = 1) -> vs.VideoNode:
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inflate_multi: this is not a clip')

    for i in range(radius):
        src = core.std.Inflate(src, planes=planes)
    return src


def haf_mt_deflate_multi(src: vs.VideoNode, planes: Optional[PlanesType] = None, radius: int = 1) -> vs.VideoNode:
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_deflate_multi: this is not a clip')

    for _ in range(radius):
        src = core.std.Deflate(src, planes=planes)
    return src


def haf_cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)


def haf_m4(x: int) -> int:
    return 16 if x < 16 else haf_cround(x / 4) * 4


def pyramid(
    clip: vs.VideoNode,
    num_levels: int = 11,
    scale: float = 0.5,
    sigma: float = 1.0,
    resampler: typing.Callable[[vs.VideoNode, int, int], vs.VideoNode] = core.resize.Bilinear
) -> typing.Tuple[typing.List[vs.VideoNode], typing.List[vs.VideoNode]]:

    gaussian_pyramids = [clip]
    laplacian_pyramids = []

    for _ in range(num_levels):
        down = resampler(clip.tcanny.TCanny(mode=-1, sigma=sigma), int(clip.width * scale), int(clip.height * scale))
        gaussian_pyramids.append(down)

        up = resampler(down, clip.width, clip.height).tcanny.TCanny(mode=-1, sigma=sigma)
        laplacian_pyramids.append(core.std.MakeDiff(clip, up))

        clip = down

    return gaussian_pyramids, laplacian_pyramids


def pyramid_texture_filter(
    clip: vs.VideoNode,
    sigma_s: float = 5.0,
    sigma_r: float = 0.05,
    num_levels: int = 11,
    sigma_g: float = 1.0,
    scale: float = 0.8,
    resampler: typing.Callable[[vs.VideoNode, int, int], vs.VideoNode] = core.resize.Bilinear
) -> vs.VideoNode:
    """ Pyramid Texture Filtering

    Pyramid texture filtering is a simple but effective technique to smooth out textures while preserving the prominent structures.
    It is built upon a key observation that
    the coarsest level in a Gaussian pyramid often naturally eliminates textures and summarizes the main image structures.
    This idea is used for texture filtering,
    which is to progressively upsample the very low-resolution coarsest Gaussian pyramid level
    to a full-resolution texture smoothing result with well-preserved structures,
    under the guidance of each fine-scale Gaussian pyramid level and its associated Laplacian pyramid level.
    The authors claim that it is effective to separate structure from texture of different scales, local contrasts, and forms,
    without degrading structures or introducing visual artifacts.

    Personally this filter creates too much aliasing and still cannot efficiently differentiate textures from structures.

    Args:
        clip: RGB/YUV/Gray, 8..16 bit integer.

        sigma_s: (float) sigmaS of bilateral filter.
            Default is 5.0.

        sigma_r: (float) sigmaR of bilateral filter.
            Default is 0.05.

        num_levels: (int) Number of pyramid levels.
            Default is 11.

        scale: (float) Scaling factor to build pyramid.
            Default is 0.8.

        sigma_g: (float) Sigma of gaussian filter.
            Default is 1.0.

        resampler: (Callable[[vs.VideoNode, int, int] -> vs.VideoNode]) Resampler to build pyramid.
            Default is Bilinear.

    Ref:
        [1] Zhang, Q., Jiang, H., Nie, Y., & Zheng, W. S. (2023). Pyramid Texture Filtering. In ACM SIGGRAPH 2023 Conference Proceedings.
        [2] https://github.com/RewindL/pyramid_texture_filtering
    """

    funcName = "pyramid_texture_filter"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    gaussian_pyramids, laplacian_pyramids = pyramid(clip, num_levels=num_levels, scale=scale, resampler=resampler, sigma=sigma_g)

    r = gaussian_pyramids[-1]
    for i in range(len(laplacian_pyramids) - 1, -1, -1):
        adaptive_sigma_s = sigma_s * scale ** i

        # upsample
        r_up = resampler(r, gaussian_pyramids[i].width, gaussian_pyramids[i].height)
        r_hat = core.bilateral.Bilateral(r_up, ref=gaussian_pyramids[i], sigmaS=adaptive_sigma_s, sigmaR=sigma_r)

        # laplacian
        r_laplacian = core.std.MergeDiff(r_hat, laplacian_pyramids[i])
        r_out = core.bilateral.Bilateral(r_laplacian, ref=r_hat, sigmaS=adaptive_sigma_s, sigmaR=sigma_r)

        # enhancement
        r_refine = core.bilateral.Bilateral(r_out, sigmaS=adaptive_sigma_s, sigmaR=sigma_r)
        r = r_refine

    return r


def pixels_per_degree(
    distance: float = 0.7,
    monitor_width: float = 0.7,
    resolution: float = 3840
) -> float:
    return distance / monitor_width * math.pi * (resolution / 180)


def flip(
    reference: vs.VideoNode,
    test: vs.VideoNode,
    pixels_per_degree: float = pixels_per_degree(distance=0.7, monitor_width=0.7, resolution=3840),
    map_type: int = 0
) -> vs.VideoNode:
    """ ꟻLIP Image Difference Evaluator

    ꟻLIP is a full-reference image difference algorithm. Given two input RGBS images in sRGB space,
    it returns a GRAYS image indicating the magnitude of the perceived difference between two images at each pixel.

    It takes into account perceptual difference in the YCxCz color space and also edges and points in images.

    Example:
        # monitor of width 0.8 meter, resolution 3840 in width for an observer at distance 0.7 meter
        diff = muf.flip(src, flt, muf.pixels_per_degree(distance=0.7, monitor_width=0.8, resolution=3840))

    Args:
        reference, test: Input clip. Must be of the same resolution and of RGBS color format.

        pixels_per_degree: (float) Number of pixels per degree of the viewing environment.

        map_type: (int, 0~2) Type of the output map.
            0: Original ꟻLIP map.
            1: Color-only map.
            2: Features-only (points and edges) map.

    Ref:
        [1] Andersson, P., Nilsson, J., Akenine-Möller, T., Oskarsson, M., Åström, K., & Fairchild, M. D. (2020).
            FLIP: A Difference Evaluator for Alternating Images.
            Proc. ACM Comput. Graph. Interact. Tech., 3(2), 15-1.
        [2] https://github.com/NVlabs/flip
    """

    funcName = "flip"

    if not isinstance(reference, vs.VideoNode) or reference.format.id != vs.RGBS:
        raise TypeError(f'{funcName}: "reference" must be an RGBS clip!')

    if not isinstance(test, vs.VideoNode) or test.format.id != vs.RGBS:
        raise TypeError(f'{funcName}: "test" must be a clip!')

    if reference.width != test.width or reference.height != test.height:
        raise TypeError(f'{funcName}: "reference" and "test" must be of the same resolution!')

    def srgb2ycxcz(clip: vs.VideoNode) -> typing.Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
        r = core.std.ShufflePlanes(clip, [0], vs.GRAY)
        g = core.std.ShufflePlanes(clip, [1], vs.GRAY)
        b = core.std.ShufflePlanes(clip, [2], vs.GRAY)

        # linear rgb -> xyz
        A1 = [
            [10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794],
            [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
            [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382]
        ]

        # xyz -> YCxCz
        A2 = [
            [0, 116, 0],
            [500 * 1.052156925, -500, 0],
            [0, 200, -200 * 0.918357670]
        ] # type: list[list[float]]
        B2 = [-16, 0, 0]

        return tuple(
            core.akarin.Expr(
                [r, g, b],
                f"x 0.04045 > x 0.055 + 1.055 / 2.4 pow x 12.92 / ? {math.fsum(A2[j][i] * A1[i][0] for i in range(3))} * "
                f"y 0.04045 > y 0.055 + 1.055 / 2.4 pow y 12.92 / ? {math.fsum(A2[j][i] * A1[i][1] for i in range(3))} * + "
                f"z 0.04045 > z 0.055 + 1.055 / 2.4 pow z 12.92 / ? {math.fsum(A2[j][i] * A1[i][2] for i in range(3))} * + "
                f"{B2[j]} +"
            )
            for j in range(3)
        ) # type: ignore

    def ycxcz2pre_lab(y: vs.VideoNode, cx: vs.VideoNode, cz: vs.VideoNode) -> typing.Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
        # ycxcz -> xyz
        A1 = [
            [1 / 116, 1 / 500, 0],
            [1 / 116, 0, 0],
            [1 / 116, 0, -1/200]
        ]
        B1 = [
            16 / 116,
            16 / 116,
            16 / 116
        ]

        # xyz -> linrgb -> clamp
        A2 = [
            [3.241003275, -1.537398934, -0.498615861],
            [-0.969224334, 1.875930071, 0.041554224],
            [0.055639423, -0.204011202, 1.057148933]
        ]
        linrgb = tuple(
            core.akarin.Expr(
                [y, cx, cz],
                f"x {math.fsum(A2[j][i] * A1[i][0] for i in range(3))} * "
                f"y {math.fsum(A2[j][i] * A1[i][1] for i in range(3))} * + "
                f"z {math.fsum(A2[j][i] * A1[i][2] for i in range(3))} * + "
                f"{math.fsum(A2[j][i] * B1[i] for i in range(3))} + 0 1 clip"
            )
            for j in range(3)
        )

        # linrgb -> xyz
        A3 = [
            [10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794],
            [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
            [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382]
        ]

        # linrgb -> xyz -> pre-lab
        return tuple(
            core.akarin.Expr(
                linrgb,
                f"x {A3[i][0]} * "
                f"y {A3[i][1]} * + "
                f"z {A3[i][2]} * + T! "
                f"T@ {6 ** 3 / 29 ** 3} > T@ {1/3} pow T@ {29 ** 3 / (3 * 6 ** 3)} * {4 / 29} + ?"
            )
            for i in range(3)
        ) # type: ignore

    def get_filter(
        pixels_per_degree: float
    ) -> typing.Tuple[typing.List[float], typing.List[float], typing.List[float]]:
        a1 = [1, 1, 34.1]
        a2 = [0, 0, 13.5]
        b1 = [0.0047, 0.0053, 0.04]
        b2 = [1e-5, 1e-5, 0.025]

        max_scale_parameter = max(*b1, *b2)
        r = math.ceil(3 * math.sqrt(max_scale_parameter / 2) / math.pi * pixels_per_degree)
        zs = [
            (x / pixels_per_degree) ** 2 + (y / pixels_per_degree) ** 2
            for y in range(-r, r + 1)
            for x in range(-r, r + 1)
        ]
        ss = [
            [
                a1[i] * math.sqrt(math.pi / b1[i]) * math.exp(-math.pi**2 * z / b1[i]) +
                a2[i] * math.sqrt(math.pi / b2[i]) * math.exp(-math.pi**2 * z / b2[i])
                for z in zs
            ]
            for i in range(3)
        ]
        sums = [math.fsum(s) for s in ss]
        ss = [[s / sums[i] if s / sums[i] > 1e-5 else 0 for s in ss[i]] for i in range(len(ss))]

        return ss # type: ignore

    def calc_color_error(
        l1: vs.VideoNode, a1: vs.VideoNode, b1: vs.VideoNode,
        l2: vs.VideoNode, a2: vs.VideoNode, b2: vs.VideoNode
    ) -> vs.VideoNode:

        qc = 0.7
        cmax = 203.30165581011028 ** qc
        pc = 0.4
        pt = 0.95
        pccmax = pc * cmax

        return core.akarin.Expr(
            [l1, a1, b1, l2, a2, b2],
            "116 y * 16 - L1! "
            "x y - 5 * L1@ * A1! "
            "y z - 2 * L1@ * B1! "
            "116 b * 16 - L2! "
            "a b - 5 * L2@ * A2! "
            "b c - 2 * L2@ * B2! "
            f"L1@ L2@ - abs A1@ A2@ - dup * B1@ B2@ - dup * + sqrt + {qc} pow HYAB! "
            f"HYAB@ {pccmax} < {pt / pccmax} HYAB@ * {pt} HYAB@ {pccmax} - {cmax - pccmax} / {1 - pt} * + ?"
        )

    def convolution(clip: vs.VideoNode, matrix: typing.Sequence[float]) -> vs.VideoNode:
        radius = (math.isqrt(len(matrix)) - 1) // 2
        return core.akarin.Expr(
            clip,
            ' '.join(
                f"x[{x},{y}]:c {matrix[(y + radius) * (2 * radius + 1) + (x + radius)]} * {'+' if y > -radius or x > -radius else ''}"
                for y in range(-radius, radius + 1)
                for x in range(-radius, radius + 1)
            )
        )

    def output(clip: vs.VideoNode) -> vs.VideoNode:
        return (
            clip
            .std.SetFrameProp(prop="_Matrix", intval=2)
            .std.SetFrameProp(prop="_Primaries", intval=1)
            .std.SetFrameProp(prop="_Transfer", intval=1)
            .std.SetFrameProp(prop="_ColorRange", intval=0)
        )

    src = reference
    src_ycxcz = srgb2ycxcz(src)
    src_y = src_ycxcz[0]

    dst = test
    dst_ycxcz = srgb2ycxcz(dst)
    dst_y = dst_ycxcz[0]

    if map_type == 0 or map_type == 1:
        filters = get_filter(pixels_per_degree=pixels_per_degree)

        filtered_src_ycxcz = [
            convolution(c, matrix=filters[i])
            for i, c in enumerate(src_ycxcz)
        ]
        src_prelab = ycxcz2pre_lab(*filtered_src_ycxcz)

        filtered_dst_ycxcz = [
            convolution(c, matrix=filters[i])
            for i, c in enumerate(dst_ycxcz)
        ]
        dst_prelab = ycxcz2pre_lab(*filtered_dst_ycxcz)

        color_error = calc_color_error(*src_prelab, *dst_prelab)

        if map_type == 1:
            return output(color_error)

    if map_type == 0 or map_type == 2:
        qf = 0.5
        w = 0.082
        sd = 0.5 * w * pixels_per_degree
        radius = math.ceil(3 * sd)

        flatten = lambda matrix: [matrix[y][x] for x in range(len(matrix[0])) for y in range(len(matrix))]
        transpose = lambda matrix: [[matrix[y][x] for y in range(len(matrix))] for x in range(len(matrix[0]))]

        x_vec = y_vec = list(range(-radius, radius + 1))
        Gx1 = [[-x * math.exp(-(x ** 2 + y ** 2) / (2 * sd ** 2)) for x in x_vec] for y in y_vec]
        negative_weights_sum = -math.fsum(v for v in flatten(Gx1) if v < 0)
        positive_weights_sum = math.fsum(v for v in flatten(Gx1) if v > 0)
        Gx1 = [[(v / negative_weights_sum if v < 0 else v / positive_weights_sum) for v in t] for t in Gx1]

        Gx2 = [[(x ** 2 / sd ** 2 - 1) * math.exp(-(x ** 2 + y ** 2) / (2 * sd ** 2)) for x in x_vec] for y in y_vec]
        negative_weights_sum = -math.fsum(v for v in flatten(Gx2) if v < 0)
        positive_weights_sum = math.fsum(v for v in flatten(Gx2) if v > 0)
        Gx2 = [[(v / negative_weights_sum if v < 0 else v / positive_weights_sum) for v in t] for t in Gx2]

        src_y = core.akarin.Expr([src_y], "x 16 + 116 /")
        src_y1 = convolution(src_y, flatten(Gx1))
        src_y2 = convolution(src_y, flatten(transpose(Gx1)))
        src_y3 = convolution(src_y, flatten(Gx2))
        src_y4 = convolution(src_y, flatten(transpose(Gx2)))
        dst_y = core.akarin.Expr([dst_y], "x 16 + 116 /")
        dst_y1 = convolution(dst_y, flatten(Gx1))
        dst_y2 = convolution(dst_y, flatten(transpose(Gx1)))
        dst_y3 = convolution(dst_y, flatten(Gx2))
        dst_y4 = convolution(dst_y, flatten(transpose(Gx2)))

        if map_type == 0:
            return output(core.akarin.Expr(
                [src_y1, src_y2, src_y3, src_y4, dst_y1, dst_y2, dst_y3, dst_y4, color_error],
                "f 1 x dup * y dup * + sqrt b dup * c dup * + sqrt - abs "
                f"z dup * a dup * + sqrt d dup * e dup * + sqrt - abs max {math.sqrt(2)} / {qf} pow - pow"
            ))
        else:
            return output(core.akarin.Expr(
                [src_y1, src_y2, src_y3, src_y4, dst_y1, dst_y2, dst_y3, dst_y4],
                "x dup * y dup * + sqrt b dup * c dup * + sqrt - abs "
                f"z dup * a dup * + sqrt d dup * e dup * + sqrt - abs max {math.sqrt(2)} / {qf} pow"
            ))
    else:
        raise ValueError(f'{funcName}: "map_type" must be 0, 1 or 2!')


def expr_join(iterable: typing.Iterable[str], op: str) -> str:
    iterator = iter(iterable)
    ret = next(iterator)
    for item in iterator:
        ret = f"{ret} {item} {op}"
    return ret


def temporal_dft(clip: vs.VideoNode, radius: int = 1) -> typing.List[vs.VideoNode]:
    """ Temporal Discrete Fourier Transform

    Performs temporal dft on real input data.

    With (n := 2 * radius + 1), dft is defined as
        y[k]: complex = sum(x[k] * exp(-2 * pi * j * k / n) for k in range(n)) / sqrt(n)
    i.e. orthogonal dft. Output in interleaved complex format in float.

    Similarly, inverse dft is similarly defined as
        z[k]: float = sum(y[k] * exp(2 * pi * j * k / n) for k in range(n)) / sqrt(n)

    Example (frequency thresholding):
        # y1_real, y1_imag, y2_real, y2_imag, y3_real, y3_imag
        dfts = temporal_dft(clip)

        for i in range(len(dfts)):
            dfts[i] = core.akarin.Expr('x {thr} < 0 x ?')

        outputs = temporal_idft(dfts)

        output = outputs[len(outputs) // 2]

    Args:
        clip: Input clip.

        radius: (int) Temporal radius. Must be positive.
            Default is 1.
    """

    funcName = "temporal_dft"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "reference" must be a clip!')

    if radius <= 0:
        raise ValueError(f'{funcName}: "radius" must be positive!')

    dft_width = 2 * radius + 1

    def shift(clip: vs.VideoNode, delta: int) -> vs.VideoNode:
        if delta == 0:
            return clip
        elif delta < 0:
            return core.std.Splice([clip[0]] * -delta + [clip[:delta]])
        else:
            return core.std.Splice([clip[delta:]] + [clip[-1]] * delta)

    def coeff(i: int, j: int) -> float:
        pos = -2 * i * (j // 2) / dft_width * math.pi
        if j % 2 == 0: # real part
            return math.cos(pos) / math.sqrt(dft_width)
        else: # imaginary part
            return math.sin(pos) / math.sqrt(dft_width)

    clips = [shift(clip, i) for i in range(-radius, radius + 1)]

    if clip.format.sample_type == vs.FLOAT:
        norm = 1
        format = clip.format
    else:
        norm = (2 ** clip.format.bits_per_sample) - 1
        format = clip.format.replace(core=core, sample_type=vs.FLOAT, bits_per_sample=32)

    dfts = [
        core.akarin.Expr(
            clips,
            expr_join((f"src{i} {coeff(i, j) / norm} *" for i in range(dft_width)), '+'),
            format=format.id
        )
        for j in range(2 * dft_width)
    ]

    return dfts


def temporal_idft(clips: typing.Sequence[vs.VideoNode]) -> typing.List[vs.VideoNode]:
    """ Temporal Inverse Discrete Fourier Transform

    Check `temporal_dft()` for details.
    """

    dft_width = len(clips) // 2

    def coeff(i: int, j: int) -> float:
        pos = 2 * (i // 2) * j / dft_width * math.pi
        if i % 2 == 0: # real part
            return math.cos(pos) / math.sqrt(dft_width)
        else: # imaginary part
            return -math.sin(pos) / math.sqrt(dft_width)

    idfts = [
        core.akarin.Expr(
            clips,
            expr_join((f"src{i} {coeff(i, j)} *" for i in range(2 * dft_width)), '+')
        )
        for j in range(dft_width)
    ]

    return idfts


def srestore(
    source: vs.VideoNode,
    frate: Optional[numbers.Real] = None,
    omode: int = 6,
    speed: Optional[int] = None,
    mode: int = 2,
    thresh: int = 16,
    dclip: Optional[vs.VideoNode] = None
) -> vs.VideoNode:

    """ srestore v2.7e
    srestore with serialized execution by explicit node processing dependency

    modified from havsfunc's srestore function
    https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/e236281cd8c1dd6b1b0cc906844944b79b1b52fa/havsfunc.py#L1899-L2227
    """

    if not isinstance(source, vs.VideoNode):
        raise vs.Error('srestore: this is not a clip')

    if source.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    if dclip is None:
        dclip = source
    elif not isinstance(dclip, vs.VideoNode):
        raise vs.Error("srestore: 'dclip' is not a clip")
    elif dclip.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    bits = source.format.bits_per_sample
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    ###### parameters & other necessary vars ######
    srad = math.sqrt(abs(speed)) * 4 if speed is not None and abs(speed) >= 1 else 12
    irate = source.fps_num / source.fps_den
    bsize = 16 if speed is not None and speed > 0 else 32
    bom = isinstance(omode, str)
    thr = abs(thresh) + 0.01

    if bom or abs(omode - 3) < 2.5:
        frfac = 1
    elif frate is not None:
        if frate * 5 < irate or frate > irate:
            frfac = 1
        else:
            frfac = abs(frate) / irate
    elif haf_cround(irate * 10010) % 30000 == 0:
        frfac = 1001 / 2400
    else:
        frfac = 480 / 1001

    if abs(frfac * 1001 - haf_cround(frfac * 1001)) < 0.01:
        numr = haf_cround(frfac * 1001)
    elif abs(1001 / frfac - haf_cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = haf_cround(frfac * 9000)
    if (
        frate is not None and
        abs(irate * numr / haf_cround(numr / frfac) - frate) > abs(irate * haf_cround(frate * 100) / haf_cround(irate * 100) - frate)
    ):
        numr = haf_cround(frate * 100)
    denm = haf_cround(numr / frfac)

    ###### source preparation & lut ######
    if abs(mode) >= 2 and not bom:
        mec = core.std.Merge(source, source.std.Trim(first=1), weight=[0, 0.5])
        mec = core.std.Merge(mec, source.std.Trim(first=1), weight=[0.5, 0])

    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)
    dclip = dclip.resize.Point(
        dclip.width if srad == 4 else int(dclip.width / 2 / srad + 4) * 4,
        dclip.height if srad == 4 else int(dclip.height / 2 / srad + 4) * 4
    )
    dclip = dclip.std.Trim(first=2)
    if mode < 0:
        dclip = core.std.StackVertical([
            core.std.StackHorizontal([mvf.GetPlane(dclip, 1), mvf.GetPlane(dclip, 2)]),
            mvf.GetPlane(dclip, 0)
        ])
    else:
        dclip = mvf.GetPlane(dclip, 0)
    if bom:
        dclip = dclip.std.Expr(expr=['x 0.5 * 64 +'])

    expr1 = 'x 128 - y 128 - * 0 > x 128 - abs y 128 - abs < x 128 - 128 x - * y 128 - 128 y - * ? x y + 256 - dup * ? 0.25 * 128 +'
    expr2 = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
    diff = core.std.MakeDiff(dclip, dclip.std.Trim(first=1))
    if not bom:
        bclp = core.std.Expr([diff, diff.std.Trim(first=1)], expr=[expr1]).resize.Bilinear(bsize, bsize)
    else:
        bclp = core.std.Expr([
            diff.std.Trim(first=1),
            core.std.MergeDiff(diff, diff.std.Trim(first=2))
        ], expr=[expr2])
        bclp = bclp.resize.Bilinear(bsize, bsize)
    dclp = diff.std.Trim(first=1).std.Lut(function=lambda x: max(haf_cround(abs(x - 128) ** 1.1 - 1), 0))
    dclp = dclp.resize.Bilinear(bsize, bsize)

    ###### postprocessing ######
    if bom:
        sourceDuplicate = source.std.DuplicateFrames(frames=[0])
        sourceTrim1 = source.std.Trim(first=1)
        sourceTrim2 = source.std.Trim(first=2)

        unblend1 = core.std.Expr([sourceDuplicate, source], expr=['x -1 * y 2 * +'])
        unblend2 = core.std.Expr([sourceTrim1, sourceTrim2], expr=['x 2 * y -1 * +'])

        qmask1 = core.std.MakeDiff(
            unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend1,
            planes=[0]
        )
        qmask2 = core.std.MakeDiff(
            unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend2,
            planes=[0]
        )
        diffm = core.std.MakeDiff(sourceDuplicate, source, planes=[0]).std.Maximum(planes=[0])
        bmask = core.std.Expr([qmask1, qmask2], expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        expr = (
            'x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?'
            .format(i=scale(4, bits), peak=peak, j=scale(200, bits), k=scale(28, bits))
        )
        dmask = core.std.Expr([diffm, diffm.std.Trim(first=2)], expr=[expr, ''])
        pmask = core.std.Expr([dmask, bmask], expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])

        matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

        omode = omode.lower()
        if omode == 'pp0':
            fin = core.std.Expr([sourceDuplicate, source, sourceTrim1, sourceTrim2], expr=['x -0.5 * y + z + a -0.5 * +'])
        elif omode == 'pp1':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                dmask.std.Convolution(matrix=matrix, planes=[0]).std.Expr(expr=['', repr(neutral)])
            )
        elif omode == 'pp2':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                bmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True
            )
        elif omode == 'pp3':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                pmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True
            )
            fin = fin.std.Convolution(matrix=matrix, planes=[1, 2])
        else:
            raise vs.Error('srestore: unexpected value for omode')

    def srestore_inside(n: int, f: List[vs.VideoFrame], real_n: int) -> vs.VideoNode:
        n = real_n

        if n == 0:
            ###### initialise variables ######
            lfr = -100
            offs = 0
            ldet = -100
            lpos = 0
            d32 = d21 = d10 = d01 = d12 = d23 = d34 = None
            m42 = m31 = m20 = m11 = m02 = m13 = m24 = None
            bp2 = bp1 = bn0 = bn1 = bn2 = bn3 = None
            cp2 = cp1 = cn0 = cn1 = cn2 = cn3 = None
        else:
            state = f[3].props

            lfr = state["_lfr"]
            offs = state["_offs"]
            ldet = state["_ldet"]
            lpos = state["_lpos"]
            d32 = state.get("_d32")
            d21 = state.get("_d21")
            d10 = state.get("_d10")
            d01 = state.get("_d01")
            d12 = state.get("_d12")
            d23 = state.get("_d23")
            d34 = state.get("_d34")
            m42 = state.get("_m42")
            m31 = state.get("_m31")
            m20 = state.get("_m20")
            m11 = state.get("_m11")
            m02 = state.get("_m02")
            m13 = state.get("_m13")
            m24 = state.get("_m24")
            bp2 = state.get("_bp2")
            bp1 = state.get("_bp1")
            bn0 = state.get("_bn0")
            bn1 = state.get("_bn1")
            bn2 = state.get("_bn2")
            bn3 = state.get("_bn3")
            cp2 = state.get("_cp2")
            cp1 = state.get("_cp1")
            cn0 = state.get("_cn0")
            cn1 = state.get("_cn1")
            cn2 = state.get("_cn2")
            cn3 = state.get("_cn3")

        ### preparation ###
        jmp = lfr + 1 == n
        cfo = ((n % denm) * numr * 2 + denm + numr) % (2 * denm) - denm
        bfo = cfo > -numr and cfo <= numr
        lfr = n
        if bfo:
            if offs <= -4 * numr:
                offs = offs + 2 * denm
            elif offs >= 4 * numr:
                offs = offs - 2 * denm
        pos = 0 if frfac == 1 else -haf_cround((cfo + offs) / (2 * numr)) if bfo else lpos
        cof = cfo + offs + 2 * numr * pos
        ldet = -1 if n + pos == ldet else n + pos

        ### diff value shifting ###
        d_v = f[1].props['PlaneStatsMax'] + 0.015625
        if jmp:
            d43 = d32
            d32 = d21
            d21 = d10
            d10 = d01
            d01 = d12
            d12 = d23
            d23 = d34
        else:
            d43 = d32 = d21 = d10 = d01 = d12 = d23 = d_v
        d34 = d_v

        m_v = f[2].props['PlaneStatsDiff'] * 255 + 0.015625 if not bom and abs(omode) > 5 else 1
        if jmp:
            m53 = m42
            m42 = m31
            m31 = m20
            m20 = m11
            m11 = m02
            m02 = m13
            m13 = m24
        else:
            m53 = m42 = m31 = m20 = m11 = m02 = m13 = m_v
        m24 = m_v

        ### get blend and clear values ###
        b_v = 128 - f[0].props['PlaneStatsMin']
        if b_v < 1:
            b_v = 0.125
        c_v = f[0].props['PlaneStatsMax'] - 128
        if c_v < 1:
            c_v = 0.125

        ### blend value shifting ###
        if jmp:
            bp3 = bp2
            bp2 = bp1
            bp1 = bn0
            bn0 = bn1
            bn1 = bn2
            bn2 = bn3
        else:
            bp3 = b_v - c_v if bom else b_v
            bp2 = bp1 = bn0 = bn1 = bn2 = bp3
        bn3 = b_v - c_v if bom else b_v

        ### clear value shifting ###
        if jmp:
            cp3 = cp2
            cp2 = cp1
            cp1 = cn0
            cn0 = cn1
            cn1 = cn2
            cn2 = cn3
        else:
            cp3 = cp2 = cp1 = cn0 = cn1 = cn2 = c_v
        cn3 = c_v

        ### used detection values ###
        bb = [bp3, bp2, bp1, bn0, bn1][pos + 2]
        bc = [bp2, bp1, bn0, bn1, bn2][pos + 2]
        bn = [bp1, bn0, bn1, bn2, bn3][pos + 2]

        cb = [cp3, cp2, cp1, cn0, cn1][pos + 2]
        cc = [cp2, cp1, cn0, cn1, cn2][pos + 2]
        cn = [cp1, cn0, cn1, cn2, cn3][pos + 2]

        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]
        dn2 = [d01, d12, d23, d34, d34][pos + 2]

        mb1 = [m53, m42, m31, m20, m11][pos + 2]
        mb = [m42, m31, m20, m11, m02][pos + 2]
        mc = [m31, m20, m11, m02, m13][pos + 2]
        mn = [m20, m11, m02, m13, m24][pos + 2]
        mn1 = [m11, m02, m13, m24, 0.01][pos + 2]

        ### basic calculation ###
        bbool = 0.8 * bc * cb > bb * cc and 0.8 * bc * cn > bn * cc and bc * bc > cc
        blend = (
            bbool and
            bc * 5 > cc and
            dbc + dcn > 1.5 * thr and
            (dbb < 7 * dbc or dbb < 8 * dcn) and
            (dnn < 8 * dcn or dnn < 7 * dbc) and
            (
                mb < mb1 and mb < mc or
                mn < mn1 and mn < mc or
                (dbb + dnn) * 4 < dbc + dcn or
                (bb * cc * 5 < bc * cb or mb > thr) and (bn * cc * 5 < bc * cn or mn > thr) and bc > thr
            )
        )
        clear = (
            dbb + dbc > thr and
            dcn + dnn > thr and
            (bc < 2 * bb or bc < 2 * bn) and
            (dbb + dnn) * 2 > dbc + dcn and
            (
                mc < 0.96 * mb and mc < 0.96 * mn and (bb * 2 > cb or bn * 2 > cn) and cc > cb and cc > cn or
                frfac > 0.45 and frfac < 0.55 and 0.8 * mc > mb1 and 0.8 * mc > mn1 and mb > 0.8 * mn and mn > 0.8 * mb
            )
        )
        highd = dcn > 5 * dbc and dcn > 5 * dnn and dcn > thr and dbc < thr and dnn < thr
        lowd = (
            dcn * 5 < dbc and
            dcn * 5 < dnn and
            dbc > thr and
            dnn > thr and
            dcn < thr and
            frfac > 0.35 and
            (frfac < 0.51 or dcn * 5 < dbb)
        )
        res = (
            d43 < thr and
            d32 < thr and
            d21 < thr and
            d10 < thr and
            d01 < thr and
            d12 < thr and
            d23 < thr and
            d34 < thr or

            dbc * 4 < dbb and
            dcn * 4 < dbb and
            dnn * 4 < dbb and
            dn2 * 4 < dbb or

            dcn * 4 < dbc and
            dnn * 4 < dbc and
            dn2 * 4 < dbc
        )

        ### offset calculation ###
        if blend:
            odm = denm
        elif clear:
            odm = 0
        elif highd:
            odm = denm - numr
        elif lowd:
            odm = 2 * denm - numr
        else:
            odm = cof
        odm += haf_cround((cof - odm) / (2 * denm)) * 2 * denm

        if blend:
            odr = denm - numr
        elif clear or highd:
            odr = numr
        elif frfac < 0.5:
            odr = 2 * numr
        else:
            odr = 2 * (denm - numr)
        odr *= 0.9

        if ldet >= 0:
            if cof > odm + odr:
                if cof - offs - odm - odr > denm and res:
                    cof = odm + 2 * denm - odr
                else:
                    cof = odm + odr
            elif cof < odm - odr:
                if offs > denm and res:
                    cof = odm - 2 * denm + odr
                else:
                    cof = odm - odr
            elif offs < -1.15 * denm and res:
                cof += 2 * denm
            elif offs > 1.25 * denm and res:
                cof -= 2 * denm

        offs = 0 if frfac == 1 else cof - cfo - 2 * numr * pos
        lpos = pos
        if frfac == 1:
            opos = 0
        else:
            opos = -haf_cround((cfo + offs + (denm if bfo and offs <= -4 * numr else 0)) / (2 * numr))
        pos = min(max(opos, -2), 2)

        ### frame output calculation - resync - dup ###
        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]

        ### dup_hq - merge ###
        if opos != pos or abs(mode) < 2 or abs(mode) == 3:
            dup = 0
        elif (
            dcn * 5 < dbc and dnn * 5 < dbc and (dcn < 1.25 * thr or bn < bc and pos == lpos) or
            (dcn * dcn < dbc or dcn * 5 < dbc) and bn < bc and pos == lpos and dnn < 0.9 * dbc or
            dnn * 9 < dbc and dcn * 3 < dbc
        ):
            dup = 1
        elif (
            (dbc * dbc < dcn or dbc * 5 < dcn) and bb < bc and pos == lpos and dbb < 0.9 * dcn or
            dbb * 9 < dcn and dbc * 3 < dcn or
            dbb * 5 < dcn and dbc * 5 < dcn and (dbc < 1.25 * thr or bb < bc and pos == lpos)
        ):
            dup = -1
        else:
            dup = 0

        mer = (
            not bom and
            opos == pos and
            dup == 0 and
            abs(mode) > 2 and
            (
                dbc * 8 < dcn or
                dbc * 8 < dbb or
                dcn * 8 < dbc or
                dcn * 8 < dnn or
                dbc * 2 < thr or
                dcn * 2 < thr or
                dnn * 9 < dbc and dcn * 3 < dbc or
                dbb * 9 < dcn and dbc * 3 < dcn
            )
        )

        ### deblend - doubleblend removal - postprocessing ###
        add = (
            bp1 * cn2 > bn2 * cp1 * (1 + thr * 0.01) and
            bn0 * cn2 > bn2 * cn0 * (1 + thr * 0.01) and
            cn2 * bn1 > cn1 * bn2 * (1 + thr * 0.01)
        )
        if bom:
            if bn0 > bp2 and bn0 >= bp1 and bn0 > bn1 and bn0 > bn2 and cn0 < 125:
                if d12 * d12 < d10 or d12 * 9 < d10:
                    dup = 1
                elif d10 * d10 < d12 or d10 * 9 < d12:
                    dup = 0
                else:
                    dup = 4
            elif bp1 > bp3 and bp1 >= bp2 and bp1 > bn0 and bp1 > bn1:
                dup = 1
            else:
                dup = 0
        elif dup == 0:
            if omode > 0 and omode < 5:
                if not bbool:
                    dup = 0
                elif omode == 4 and bp1 * cn1 < bn1 * cp1 or omode == 3 and d10 < d01 or omode == 1:
                    dup = -1
                else:
                    dup = 1
            elif omode == 5:
                if (
                    bp1 * cp2 > bp2 * cp1 * (1 + thr * 0.01) and
                    bn0 * cp2 > bp2 * cn0 * (1 + thr * 0.01) and
                    cp2 * bn1 > cn1 * bp2 * (1 + thr * 0.01) and
                    (not add or cp2 * bn2 > cn2 * bp2)
                ):
                    dup = -2
                elif add:
                    dup = 2
                elif bn0 * cp1 > bp1 * cn0 and (bn0 * cn1 < bn1 * cn0 or cp1 * bn1 > cn1 * bp1):
                    dup = -1
                elif bn0 * cn1 > bn1 * cn0:
                    dup = 1
                else:
                    dup = 0
            else:
                dup = 0

        ### output clip ###
        if dup == 4:
            ret = fin
        else:
            oclp = mec if mer and dup == 0 else source
            opos += dup - (1 if dup == 0 and mer and dbc < dcn else 0)
            if opos < 0:
                ret = oclp.std.DuplicateFrames(frames=[0] * -opos)
            else:
                ret = oclp.std.Trim(first=opos)

        ret = ret[n]

        temp_kwargs = dict(
            lfr=lfr, offs=offs, ldet=ldet, lpos=lpos,
            d32=d32, d21=d21, d10=d10, d01=d01, d12=d12, d23=d23, d34=d34,
            m42=m42, m31=m31, m20=m20, m11=m11, m02=m02, m13=m13, m24=m24,
            bp2=bp2, bp1=bp1, bn0=bn0, bn1=bn1, bn2=bn2, bn3=bn3,
            cp2=cp2, cp1=cp1, cn0=cn0, cn1=cn1, cn2=cn2, cn3=cn3
        )
        state_kwargs = {f"_{k}": v for k, v in temp_kwargs.items() if v is not None}

        if hasattr(core.std, "SetFrameProps"):
            return core.std.SetFrameProps(ret, **state_kwargs)
        else:
            for k, v in state_kwargs.items():
                ret = core.std.SetFrameProp(ret, prop=k, intval=v)
            return ret

    ###### evaluation call & output calculation ######
    bclpYStats = bclp.std.PlaneStats()
    dclpYStats = dclp.std.PlaneStats()
    dclipYStats = core.std.PlaneStats(dclip, dclip.std.Trim(first=2))

    # https://github.com/vapoursynth/vapoursynth/blob/55e7d0e989359c23782fc1e0d4aa1c0c35838a80/src/core/vsapi.cpp#L151-L152
    def get_frame(clip: vs.VideoNode, n: int) -> vs.VideoNode:
        return clip[min(n, clip.num_frames - 1)]

    last_frames: List[vs.VideoNode] = []
    state: vs.VideoNode

    for n in range(source.num_frames):
        prop_src = [
            get_frame(bclpYStats, n),
            get_frame(dclpYStats, n),
            get_frame(dclipYStats, n)
        ]

        if n > 0:
            prop_src.append(state)

        state = source[n].std.FrameEval(
            eval=functools.partial(srestore_inside, real_n=n),
            prop_src=prop_src
        )
        last_frames.append(state)

    last = core.std.Splice(last_frames)

    ###### final decimation ######
    return haf_ChangeFPS(last, source.fps_num * numr, source.fps_den * denm)
