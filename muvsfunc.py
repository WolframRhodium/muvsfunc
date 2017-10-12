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
    scale (using the old expression in havsfunc)
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
    Wiener2
'''

import vapoursynth as vs
import havsfunc as haf
import mvsfunc as mvf
import functools
import math

def LDMerge(flt_h, flt_v, src, mrad=0, show=0, planes=None, convknl=1, conv_div=None, calc_mode=0, power=1.0):
    """A filter to merge two filtered clip based on gradient direction map from source clip.

    Args:
        flt_h, flt_v: Two filtered clip.

        src: Source clip. Must matc the filtered clip.

        mrad: (int) Expanding of gradient direction map. Default is 0.

        show: (bint) Whether to output gradient direction map. Default is False.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the first clip, "flt_h".

        convknl: (0 or 1) Which convolution kernel is used to generate gradient direction map. Default is 1.

        conv_div: (int) Divisor in convolution filter. Default is the max value in convolution kernel.

        calc_mode: (0 or 1) Which method is used to calculate line direction map. Default is 0.

        power: (float) Power coefficient in "calc_mode=0".

    Example:
        # Fast Anti-aliasing
        horizontal = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='h')
        vertical = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='v')
        blur_src = core.tcanny.TCanny(clip, mode=-1, planes=[0]) # Eliminate noise
        antialiasing = muf.LDMerge(horizontal, vertical, blur_src, mrad=1, planes=[0])

    """

    core = vs.get_core()
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
        hmap = haf.mt_expand_multi(hmap, sw=0, sh=mrad, planes=planes)
        vmap = haf.mt_expand_multi(vmap, sw=mrad, sh=0, planes=planes)
    elif mrad < 0:
        hmap = haf.mt_inpand_multi(hmap, sw=0, sh=-mrad, planes=planes)
        vmap = haf.mt_inpand_multi(vmap, sw=-mrad, sh=0, planes=planes)
    
    ldexpr = '{peak} 1 x 0.0001 + y 0.0001 + / {power} pow + /'.format(peak=(1 << bits) - 1, power=power) if calc_mode == 0 else 'y 0.0001 + x 0.0001 + dup * y 0.0001 + dup * + 2 * sqrt / {peak} *'.format(peak=(1 << bits) - 1)
    ldmap = core.std.Expr([hmap, vmap], [(ldexpr if i in planes else '') for i in range(src.format.num_planes)])

    if show == 0:
        return core.std.MaskedMerge(flt_h, flt_v, ldmap, planes=planes)
    elif show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap


def Compare(src, flt, power=1.5, chroma=False, mode=2):
    """A filter to check the difference of source clip and filtered clip.

    Args:
        src: Source clip.

        flt: Filtered clip.

        power: (float) The variable in the processing kernel which controls the "strength" to increase difference. Default is 1.5.

        chroma: (bint) Whether to process chroma. Default is False.

        mode: (1 or 2) Different processing kernel. 1: non-linear; 2: linear.

    """

    core = vs.get_core()
    funcName = 'Compare'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"src\" must be a YUV clip!')
    if not isinstance(flt, vs.VideoNode):
        raise TypeError(funcName + ': \"flt\" must be a clip!')
    if src.format.id != flt.format.id:
        raise TypeError(funcName + ': \"flt\" must be of the same format as \"src\"!')
    if src.width != flt.width or src.height != flt.height:
        raise TypeError(funcName + ': \"flt\" must be of the same size as \"src\"!')
    if mode not in [1, 2]:
        raise TypeError(funcName + ': \"mode\" must be in [1, 2]!')

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


def Compare2(clip1, clip2, props_list=None):
    """Simple function to compare the format between two clips.

    TypeError will be raised when the two formats are not identical.
    Otherwise, None is returned.

    Args:
        clip1, clip2: Input.
        props_list: (list) A list containing the format to be compared. If it is none, all the formats will be compared.
            Default is None.

    """

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

    return


def ExInpand(input, mrad=0, mode='rectangle', planes=None):
    """A filter to use std.Maximum()/std.Minimum() and their mix conveniently.

    Args:
        input: Source clip.

        mrad: (int []) How many times to use std.Maximum()/std.Minimum(). Default is 0.
            Positive value indicates to use std.Maximum().
            Negative value indicates to use std.Minimum().
            Values can be formed into a list, or a list of lists.

            Example:
                mrad=[2, -1] is equvalant to clip.std.Maximum().std.Maximum().std.Minimum()
                mrad=[[2, 1], [2, -1]] is equivalant to
                    haf.mt_expand_multi(clip, sw=2, sh=1).std.Maximum().std.Maximum().std.Minimum()

        mode: (0:"rectangle", 1:"losange" or 2:"ellipse", int or string). Default is "rectangle"
            The shape of the kernel.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    core = vs.get_core()
    funcName = 'ExInpand'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if isinstance(mrad, int):
        mrad = [mrad]
    if isinstance(mode, str) or isinstance(mode, int):
        mode = [mode]
    
    if not isinstance(mode, list):
        raise TypeError(funcName + ': \"mode\" must be an int, a string, a list of ints, a list of strings or a list of mixing ints and strings!')

    # internel function
    def ExInpand_process(input, mode=None, planes=None, mrad=None):
        if isinstance(mode, int):
            mode = ['rectangle', 'losange', 'ellipse'][mode]
        if isinstance(mode, str):
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
            return haf.mt_expand_multi(input, mode=mode, planes=planes, sw=sw, sh=sh)
        else:
            return haf.mt_inpand_multi(input, mode=mode, planes=planes, sw=-sw, sh=-sh)

    # process
    if isinstance(mrad, list):
        if len(mode) < len(mrad):
            mode_length = len(mode)
            for i in range(mode_length, len(mrad)):
                mode.append(mode[mode_length - 1])

        for i in range(len(mrad)):
            if isinstance(mrad[i], list):
                if len(mrad[i]) != 1 and len(mrad[i]) != 2:
                    raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')
                for n in mrad[i]:
                    if not isinstance(n, int):
                        raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')
                if len(mrad[i]) == 1:
                    mrad[i].append(mrad[i][0])
            elif not isinstance(mrad[i], int):
                raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')
            clip = ExInpand_process(input, mode=mode[i], planes=planes, mrad=mrad[i])
    else:
        raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')

    return clip


def InDeflate(input, msmooth=0, planes=None):
    """A filter to use std.Inflate()/std.Deflate() and their mix conveniently.

    Args:
        input: Source clip.

        msmooth: (int []) How many times to use std.Inflate()/std.Deflate(). Default is 0.
            The behaviour is the same as "mode" in ExInpand(). 

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    core = vs.get_core()
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
    def InDeflate_process(input, planes=None, radius=None):
        if radius > 0:
            return haf.mt_inflate_multi(input, planes=planes, radius=radius)
        else:
            return haf.mt_deflate_multi(input, planes=planes, radius=-radius)

    # process
    if isinstance(msmooth, list):
        for m in msmooth:
            if not isinstance(m, int):
                raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')
            else:
                clip = InDeflate_process(input, planes=planes, radius=m)
    else:
        raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')
    
    return clip


def MultiRemoveGrain(input, mode=0, loop=1):
    """A filter to use rgvs.RemoveGrain() and their mix conveniently.

    Args:
        input: Source clip.
        mode: (int []) "mode" in rgvs.RemoveGrain(). Default is 0.
            Can be a list, the logic is similar to "mode" in ExInpand().

            Example: mode=[4, 11, 11] is equivalant to clip.rgvs.RemoveGrain(4).rgvs.RemoveGrain(11).rgvs.RemoveGrain(11)

        loop: (int) How many times the "mode" loops.

    """

    core = vs.get_core()
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


def GradFun3(src, thr=None, radius=None, elast=None, mask=None, mode=None, ampo=None, ampn=None,
             pat=None, dyn=None, lsb=None, staticnoise=None, smode=None, thr_det=None,
             debug=None, thrc=None, radiusc=None, elastc=None, planes=None, ref=None):
    """GradFun3 by Firesledge v0.1.1

    Ported by Muonium  2016/6/18
    Ported from Dither_tools v1.27.2 (http://avisynth.nl/index.php/Dither_tools)
    Internal calculation precision is always 16 bits.

    Read the document of Avisynth version for more details.

    Notes:
        1. In this function I try to keep the original look of GradFun3 in Avisynth.
            It should be better to use Frechdachs's GradFun3 in his fvsfunc.py (https://github.com/Irrational-Encoding-Wizardry/fvsfunc) which is more novel and powerful.

        2. current smode=1 or 2 only support small "radius" (<=9).

    Removed parameters list: 
        "dthr", "wmin", "thr_edg", "subspl", "lsb_in"
    Parameters "y", "u", "v" are changed into "planes"

    """

    core = vs.get_core()
    funcName = 'GradFun3'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcName + ': \"src\" must be YUV, GRAY or YCOCG color family!')

    if thr is None:
        thr = 0.35
    else:
        raise TypeError(funcName + ': \"thr\" must be an int or a float!')

    if smode is None:
        smode = 1
    elif smode not in [0, 1, 2, 3]:
        raise ValueError(funcName + ': \"smode\" must be in [0, 1, 2, 3]!')

    if radius is None:
        radius = (16 if src.width > 1024 or src.height > 576 else 12) if (smode == 1 or smode == 2) else 9
    elif isinstance(radius, int):
        if radius <= 0:
            raise ValueError(funcName + ': \"radius\" must be strictly positive.')
    else:
        raise TypeError(funcName + ': \"radius\" must be an int!')

    if elast is None:
        elast = 3.0
    elif isinstance(elast, int) or isinstance(elast, float):
        if elast < 1:
            raise ValueError(funcName + ': Valid range of \"elast\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elast\" must be an int or a float!')

    if mask is None:
        mask = 2
    elif not isinstance(mask, int):
        raise TypeError(funcName + ': \"mask\" must be an int!')

    if lsb is None:
        lsb = False

    if thr_det is None:
        thr_det = 2 + round(max(thr - 0.35, 0) / 0.3)
    elif isinstance(thr_det, int) or isinstance(thr_det, float):
        if thr_det <= 0.0:
            raise ValueError(funcName + '" \"thr_det\" must be strictly positive!')
    else:
        raise TypeError(funcName + ': \"mask\" must be an int or a float!')

    if debug is None:
        debug = False
    elif not isinstance(debug, bool) and debug not in [0, 1]:
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
    elif isinstance(elastc, int) or isinstance(elastc, float):
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
    if ref.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcName + ': \"ref\" must be YUV, GRAY or YCOCG color family!')
    if src.width != ref.width or src.height != ref.height:
        raise TypeError(funcName + ': \"ref\" must be of the same size as \"src\"!')

    bits = src.format.bits_per_sample
    src_16 = core.fmtc.bitdepth(src, bits=16, planes=planes) if bits < 16 else src
    src_8 = core.fmtc.bitdepth(src, bits=8, dmode=1, planes=[0]) if bits != 8 else src
    ref_16 = core.fmtc.bitdepth(ref, bits=16, planes=planes) if ref.format.bits_per_sample < 16 else ref

    # Main debanding
    chroma_flag = (thrc != thr or radiusc != radius or
                   elastc != elast) and 0 in planes and (1 in planes or 2 in planes)

    if chroma_flag:
        planes2 = [0] if 0 in planes else []
    else:
        planes2 = planes

    if not planes2:
        raise ValueError(funcName + ': no plane is processed!')

    flt_y = GF3_smooth(src_16, ref_16, smode, radius, thr, elast, planes2)
    if chroma_flag:
        if 0 in planes2:
            planes2.remove(0)
        flt_c = GF3_smooth(src_16, ref_16, smode, radiusc, thrc, elastc, planes2)
        flt = core.std.ShufflePlanes([flt_y, flt_c], list(range(src.format.num_planes)), src.format.color_family)
    else:
        flt = flt_y

    # Edge/detail mask
    td_lo = max(thr_det * 0.75, 1.0)
    td_hi = max(thr_det, 1.0)
    mexpr = 'x {tl} - {th} {tl} - / 255 *'.format(tl=td_lo - 0.0001, th=td_hi + 0.0001)

    if mask > 0:
        dmask = mvf.GetPlane(src_8, 0)
        dmask = Build_gf3_range_mask(dmask)
        dmask = core.std.Expr([dmask], [mexpr])
        dmask = core.rgvs.RemoveGrain([dmask], [22])
        if mask > 1:
            dmask = core.rgvs.RemoveGrain([dmask], [11])
            if mask > 2:
                dmask = core.std.Convolution(dmask, matrix=[1]*9)
        dmask = core.fmtc.bitdepth(dmask, bits=16, fulls=True, fulld=True)
        res_16 = core.std.MaskedMerge(flt, src_16, dmask, planes=planes, first_plane=True)
    else:
        res_16 = flt

    # Dithering
    result = res_16 if lsb else core.fmtc.bitdepth(res_16, bits=bits, planes=planes, dmode=mode, ampo=ampo,
                                                   ampn=ampn, dyn=dyn, staticnoise=staticnoise, patsize=pat)

    if debug:
        last = dmask
        if not lsb:
            last = core.fmtc.bitdepth(last, bits=8, fulls=True, fulld=True)
    else:
        last = result

    return last


def GF3_smooth(src_16, ref_16, smode, radius, thr, elast, planes):
    funcName = 'GradFun3'

    if smode == 0:
        return GF3_smoothgrad_multistage(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 1:
        return GF3_dfttest(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 2:
        return GF3_bilateral_multistage(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 3:
        return GF3_smoothgrad_multistage_3(src_16, radius, thr, elast, planes)
    else:
        raise ValueError(funcName + ': wrong smode value!')


def GF3_smoothgrad_multistage(src, ref, radius, thr, elast, planes):
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


def GF3_smoothgrad_multistage_3(src, radius, thr, elast, planes):
    core = vs.get_core()

    ref = SmoothGrad(src, radius=radius // 3, thr=thr * 0.8, elast=elast)
    last = BoxFilter(src, radius=radius, planes=planes)
    last = BoxFilter(last, radius=radius, planes=planes)
    last = mvf.LimitFilter(last, src, thr=thr * 0.6, elast=elast, ref=ref, planes=planes)
    return last


def GF3_dfttest(src, ref, radius, thr, elast, planes):
    core = vs.get_core()

    hrad = max(radius * 3 // 4, 1)
    last = core.dfttest.DFTTest(src, sigma=hrad * thr * thr * 32, sbsize=hrad * 4,
                                sosize=hrad * 3, tbsize=1, planes=planes)
    last = mvf.LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)

    return last


def GF3_bilateral_multistage(src, ref, radius, thr, elast, planes):
    core = vs.get_core()

    last = core.bilateral.Bilateral(src, ref=ref, sigmaS=radius / 2, sigmaR=thr / 255, planes=planes, algorithm=0) # The use of "thr" may be wrong

    last = mvf.LimitFilter(last, src, thr=thr, elast=elast, planes=planes)

    return last


def Build_gf3_range_mask(src, radius=1):
    core = vs.get_core()

    last = src

    if radius > 1:
        ma = haf.mt_expand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        mi = haf.mt_inpand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        last = core.std.Expr([ma, mi], ['x y -'])
    else:
        bits = src.format.bits_per_sample
        black = 0
        white = (1 << bits) - 1
        maxi = core.std.Maximum(last, [0])
        mini = core.std.Minimum(last, [0])
        exp = "x y -"
        exp2 = "x {thY1} < {black} x ? {thY2} > {white} x ?".format(thY1=0, thY2=255, black=black, white=white)
        last = core.std.Expr([maxi,mini],[exp])
        last = core.std.Expr([last], [exp2])

    return last


def AnimeMask(input, shift=0, expr=None, mode=1, resample_args=None):
    """A filter to generate edge/ringing mask for anime based on gradient operator.

    For Anime's ringing mask, it's recommended to set "shift" between 0.5 to 1.0.

    Args:
        input: Source clip. Only the First plane will be processed.

        shift: (float, -1.5 ~ 1.5) Location of mask. Default is 0.

        expr: (string) Subsequent processing in std.Expr(). Default is "".

        mode: (-1 or 1) Different kernel. Typically, -1 is for edge, 1 is for ringing. Default is 1.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict. 
            Default is dict(kernel='bicubic').

    """

    core = vs.get_core()
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

    if resample_args is None:
        resample_args = dict(kernel='bicubic')
    
    bits = input.format.bits_per_sample
    
    fmtc_args = dict(fulls=True, fulld=True)
    mask1 = core.std.Convolution(input, [0, 0, 0, 0, 2, -1, 0, -1, 0], saturate=True).fmtc.resample(sx=shift, sy=shift, **fmtc_args, **resample_args)
    mask2 = core.std.Convolution(input, [0, -1, 0, -1, 2, 0, 0, 0, 0], saturate=True).fmtc.resample(sx=-shift, sy=-shift, **fmtc_args, **resample_args)
    mask3 = core.std.Convolution(input, [0, -1, 0, 0, 2, -1, 0, 0, 0], saturate=True).fmtc.resample(sx=shift, sy=-shift, **fmtc_args, **resample_args)
    mask4 = core.std.Convolution(input, [0, 0, 0, -1, 2, 0, 0, -1, 0], saturate=True).fmtc.resample(sx=-shift, sy=shift, **fmtc_args, **resample_args)

    calc_expr = 'x x * y y * + z z * + a a * + sqrt '

    if isinstance(expr, str):
        calc_expr += expr

    mask = core.std.Expr([mask1, mask2, mask3, mask4], [calc_expr])

    if bits != mask.format.bits_per_sample:
        mask = core.fmtc.bitdepth(mask, bits=bits, fulls=True, fulld=True, dmode=1)

    return mask


def AnimeMask2(input, r=1.2, expr=None, mode=1):
    """Yet another filter to generate edge/ringing mask for anime.

    More specifically, it's an approximate Difference of Gaussians filter based on resampling kernel.

    Args:
        input: Source clip. Only the First plane will be processed.

        r: (float, positive) Radius of resampling coefficient. Default is 1.2.

        expr: (string) Subsequent processing in std.Expr(). Default is "".

        mode: (-1 or 1) Different kernel. Typically, -1 is for edge, 1 is for ringing. Default is 1.

    """

    core = vs.get_core()
    funcName = 'AnimeMask2'
    
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    w = input.width
    h = input.height
    bits = input.format.bits_per_sample

    if mode not in [-1, 1]:
        raise ValueError(funcName + ': \'mode\' have not a correct value! [-1 or 1]')

    smooth = core.fmtc.resample(input, haf.m4(w / r), haf.m4(h / r), kernel='bicubic').fmtc.resample(w, h, kernel='bicubic', a1=1, a2=0)
    smoother = core.fmtc.resample(input, haf.m4(w / r), haf.m4(h / r), kernel='bicubic').fmtc.resample(w, h, kernel='bicubic', a1=1.5, a2=-0.25)

    calc_expr = 'x y - ' if mode == 1 else 'y x - '

    if isinstance(expr, str):
        calc_expr += expr

    mask = core.std.Expr([smooth, smoother], [calc_expr])

    if bits != mask.format.bits_per_sample:
        mask = core.fmtc.bitdepth(mask, bits=bits, fulls=True, fulld=True, dmode=1)

    return mask


def PolygonExInpand(input, shift=0, shape=0, mixmode=0, noncentral=False, step=1, amp=1, fmtc_args=None, resample_args=None):
    """A filter to process mask based on resampling kernel.

    Args:
        input: Source clip. Only the First plane will be processed.

        shift: (float) How far to expand/inpand. Default is 0.

        shape: (int, 0:losange, 1:square, 2:octagon) The shape of expand/inpand kernel. Default is 0.

        mixmode: (int, 0:max, 1:arithmetic mean, 2:quadratic mean)
            Method used to calculate the mix of different mask. Default is 0.

        noncentral: (bint) Whether to calculate center pixel in mix process.

        step: (float) How far each step of expand/inpand. Default is 1.

        amp: (float) Linear multiple to strengthen the final mask. Default is 1.

        fmtc_args: (dict) Additional parameters passed to core.fmtc.resample and core.fmtc.bitdepth in the form of dict.
            Default is {}.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict. Controls which kernel is used to shift the mask.
            Default is dict(kernel='bilinear').

    """

    core = vs.get_core()
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
    elif shift == 0:
        return input

    if fmtc_args is None:
        fmtc_args = {}

    if resample_args is None:
        resample_args = dict(kernel='bilinear')

    bits = input.format.bits_per_sample

    mask5 = input

    while shift > 0:
        step = min(step, shift)
        shift = shift - step

        ortho = [step, step * (1<< input.format.subsampling_h)]
        inv_ortho = [-step, -step * (1<< input.format.subsampling_h)]
        dia = [math.sqrt(step / 2), math.sqrt(step / 2) * (1 << input.format.subsampling_h)]
        inv_dia = [-math.sqrt(step / 2), -math.sqrt(step / 2) * (1 << input.format.subsampling_h)]
        
        # shift
        if shape == 0 or shape == 2:
            mask2 = core.fmtc.resample(mask5, sx=0, sy=ortho, **fmtc_args, **resample_args)
            mask4 = core.fmtc.resample(mask5, sx=ortho, sy=0, **fmtc_args, **resample_args)
            mask6 = core.fmtc.resample(mask5, sx=inv_ortho, sy=0, **fmtc_args, **resample_args)
            mask8 = core.fmtc.resample(mask5, sx=0, sy=inv_ortho, **fmtc_args, **resample_args)

        if shape == 1 or shape == 2:
            mask1 = core.fmtc.resample(mask5, sx=dia, sy=dia, **fmtc_args, **resample_args)
            mask3 = core.fmtc.resample(mask5, sx=inv_dia, sy=dia, **fmtc_args, **resample_args)
            mask7 = core.fmtc.resample(mask5, sx=dia, sy=inv_dia, **fmtc_args, **resample_args)
            mask9 = core.fmtc.resample(mask5, sx=inv_dia, sy=inv_dia, **fmtc_args, **resample_args)

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
                mask5 = core.std.Expr([mask2, mask4, mask5, mask6, mask8] if shape == 0 else [mask1, mask3, mask5, mask7, mask9], [expr])
            else: # shape == 2
                expr = expr_list[mixmode + 3] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9], [expr])

    if bits != mask5.format.bits_per_sample:
        mask5 = core.fmtc.bitdepth(mask5, bits=bits, dmode=1, **fmtc_args)

    return core.std.Invert(mask5) if invert else mask5


def Luma(input, plane=0, power=4):
    """std.Lut() implementation of Luma() in Histogram() filter.

    Args:
        input: Source clip. Only the First plane will be processed.

        plane: (int) Which plane to be processed. Default is 0.

        power: (int) Coefficient in processing. Default is 4.

    """

    core = vs.get_core()
    funcName = 'Luma'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')
    
    if (input.format.sample_type != vs.INTEGER):
        raise TypeError(funcName + ': \"input\" must be of integer format!')
        
    bits = input.format.bits_per_sample
    peak = (1 << bits) - 1
    
    clip = mvf.GetPlane(input, plane)
    

    def calc_luma(x):
        p = x << power
        return (peak - (p & peak)) if (p & (peak + 1)) else (p & peak)
    
    return core.std.Lut(clip, function=calc_luma)


def ediaa(a):
    """Suggested by Mystery Keeper in "Denoise of tv-anime" thread

    Read the document of Avisynth version for more details.

    """

    core = vs.get_core()
    funcName = 'ediaa'
    
    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    bits = a.format.bits_per_sample
    
    last = core.eedi2.EEDI2(a, field=1).std.Transpose()
    last = core.eedi2.EEDI2(last, field=1).std.Transpose()
    last = core.fmtc.resample(last, a.width, a.height, [-0.5, -0.5 * (1 << a.format.subsampling_w)], [-0.5, -0.5 * (1 << a.format.subsampling_h)], kernel='spline36')

    if last.format.bits_per_sample == bits:
        return last
    else:
        return core.fmtc.bitdepth(last, bits=bits)


def nnedi3aa(a):
    """Using nnedi3 (Emulgator):

    Read the document of Avisynth version for more details.

    """

    core = vs.get_core()
    funcName = 'nnedi3aa'
    
    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    bits = a.format.bits_per_sample
    
    last = core.nnedi3.nnedi3(a, field=1, dh=True).std.Transpose()
    last = core.nnedi3.nnedi3(last, field=1, dh=True).std.Transpose()
    last = core.fmtc.resample(last, a.width, a.height, [-0.5, -0.5 * (1 << a.format.subsampling_w)], [-0.5, -0.5 * (1 << a.format.subsampling_h)], kernel='spline36')

    if last.format.bits_per_sample == bits:
        return last
    else:
        return core.fmtc.bitdepth(last, bits=bits)


def maa(input):
    """Anti-aliasing with edge masking by martino, mask using "sobel" taken from Kintaro's useless filterscripts and modded by thetoof for spline36

    Read the document of Avisynth version for more details.

    """

    core = vs.get_core()
    funcName = 'maa'
    
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')
    
    w = input.width
    h = input.height
    bits = input.format.bits_per_sample
    
    if input.format.color_family != vs.GRAY:
        input_src = input
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
        return core.std.ShufflePlanes([last, input_src], planes=list(range(input_src.format.num_planes)), colorfamily=input_src.format.color_family)


def SharpAAMcmod(orig, dark=0.2, thin=10, sharp=150, smooth=-1, stabilize=False, tradius=2, aapel=1, aaov=None, aablk=None, aatype='nnedi3'):
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

        stabilize: (bint) Use post stabilization with Motion Compensation. Default is False.

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

    core = vs.get_core()
    funcName = 'SharpAAMcmod'
    
    if not isinstance(orig, vs.VideoNode):
        raise TypeError(funcName + ': \"orig\" must be a clip!')
    
    w = orig.width
    h = orig.height
    bits = orig.format.bits_per_sample
    
    if orig.format.color_family != vs.GRAY:
        orig_src = orig
        orig = mvf.GetPlane(orig, 0)
    else:
        orig_src = None
    
    if aaov is None:
        aaov = 8 if w > 1100 else 4
    
    if aablk is None:
        aablk = 16 if w > 1100 else 8
    
    m = core.std.Expr([core.std.Convolution(orig, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False), core.std.Convolution(orig, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)], ['x y max {neutral8} / 0.86 pow {peak8} *'.format(neutral8=scale(128, bits), peak8=scale(255, bits))])

    if thin == 0 and dark == 0:
        preaa = orig
    elif thin == 0:
        preaa = haf.Toon(orig, str=dark)
    elif dark == 0:
        preaa = core.warp.AWarpSharp2(orig, depth=thin)
    else:
        preaa = haf.Toon(orig, str=dark).warp.AWarpSharp2(depth=thin)
    
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
        postsh = haf.LSFmod(aa, strength=sharp, overshoot=1, soft=smooth, edgemode=1)
    
    merged = core.std.MaskedMerge(orig, postsh, m)
    
    if stabilize:
        sD = core.std.MakeDiff(orig, merged)
    
        origsuper = haf.DitherLumaRebuild(orig, s0=1).mv.Super(pel=aapel)
        sDsuper = core.mv.Super(sD, pel=aapel)
    
        fv3 = core.mv.Analyse(origsuper, isb=False, delta=3, overlap=aaov, blksize=aablk) if tradius == 3 else None
        fv2 = core.mv.Analyse(origsuper, isb=False, delta=2, overlap=aaov, blksize=aablk) if tradius >= 2 else None
        fv1 = core.mv.Analyse(origsuper, isb=False, delta=1, overlap=aaov, blksize=aablk) if tradius >= 1 else None
        bv1 = core.mv.Analyse(origsuper, isb=True, delta=1, overlap=aaov, blksize=aablk) if tradius >= 1 else None
        bv2 = core.mv.Analyse(origsuper, isb=True, delta=2, overlap=aaov, blksize=aablk) if tradius >= 2 else None
        bv3 = core.mv.Analyse(origsuper, isb=True, delta=3, overlap=aaov, blksize=aablk) if tradius == 3 else None
        
        if tradius == 1:
            sDD = core.mv.Degrain1(sD, sDsuper, bv1, fv1)
        elif tradius == 2:
            sDD = core.mv.Degrain2(sD, sDsuper, bv1, fv1, bv2, fv2)
        elif tradius == 3:
            sDD = core.mv.Degrain3(sD, sDsuper, bv1, fv1, bv2, fv2, bv3, fv3)
        else:
            raise ValueError(funcName + ': valid values of \"tradius\" are 1, 2 and 3!')
        
        reduc = 0.4
        sDD = core.std.Expr([sD, sDD], ['x {neutral} - abs y {neutral} - abs < x y ?'.format(neutral=scale(128, bits))]).std.Merge(sDD, 1.0 - reduc)
    
        last = core.std.MakeDiff(orig, sDD)
    else:
        last = merged
    
    if orig_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, orig_src], planes=list(range(orig_src.format.num_planes)), colorfamily=orig_src.format.color_family)


def TEdge(input, min=0, max=65535, planes=None, rshift=0):
    """Detect edge using the kernel like TEdgeMask(type=2).

    Ported from https://github.com/chikuzen/GenericFilters/blob/2044dc6c25a1b402aae443754d7a46217a2fddbf/src/convolution/tedge.c

    Args:
        input: Source clip.

        min: (int) If output pixel value is lower than this, it will be zero. Default is 0.

        max: (int) If output pixel value is same or higher than this, it will be maximum value of the format. Default is 65535.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

        rshift: (int) Shift the output values to right by this count before clamp. Default is 0.

    """

    core = vs.get_core()
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


def Sort(input, order=1, planes=None, mode='max'):
    """Simple filter to get nth large value in 3x3.

    Args:
        input: Source clip.

        order: (int) The order of value to get in 3x3 neighbourhood. Default is 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

        mode: ("max" or "min") How to measure order. Default is "max".

    """

    core = vs.get_core()
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


def Soothe_mod(input, source, keep=24, radius=1, scenechange=32, use_misc=True):
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

    core = vs.get_core()
    funcName = 'Soothe_mod'
    
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')
    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    if not isinstance(source, vs.VideoNode):
        raise TypeError(funcName + ': \"source\" must be a clip!')
    if source.format.color_family != vs.GRAY:
        source_src = source
        source = mvf.GetPlane(source, 0)
    else:
        source_src = None

    if input.format.id != source.format.id:
        raise TypeError(funcName + ': \"source\" must be of the same format as \"input\"!')
    if input.width != source.width or input.height != source.height:
        raise TypeError(funcName + ': \"source\" must be of the same size as \"input\"!')
    
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
        if 'TemporalSoften' in dir(haf):
            diff2 = haf.TemporalSoften(diff, radius, scale(255, bits), 0, scenechange)
        else:
            raise NameError(funcName + ': \"TemporalSoften\" has been deprecated from the latest havsfunc. If you would like to use it, copy the old function in https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/0f5e6c5c2f1e825caf17f6b7de6edd4a0e13d27d/havsfunc.py#L4300 and function set_scenechange() in the following line 4320 to havsfunc in your disk.')

    expr = 'x {neutral} - y {neutral} - * 0 < x {neutral} - {KP} * {neutral} + x {neutral} - abs y {neutral} - abs > x {KP} * y {iKP} * + x ? ?'.format(neutral=scale(128, bits), KP=keep/100, iKP=1-keep/100)
    diff3 = core.std.Expr([diff, diff2], [expr])
    
    last = core.std.MakeDiff(source, diff3)

    if source_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, source_src], planes=list(range(source_src.format.num_planes)), colorfamily=source_src.format.color_family)


def TemporalSoften(input, radius=4, scenechange=15):
    """TemporalSoften filter without thresholding using Miscellaneous filters.

    There will be slight difference in result compare to havsfunc.TemporalSoften().
    It seems that this Misc-filter-based TemporalSoften is slower than the one in havsfunc.

    Read the document of Avisynth version for more details.

    """

    core = vs.get_core()
    funcName = 'TemporalSoften'
    
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if scenechange:
        if 'SCDetect' in dir(haf):
            input = haf.SCDetect(input, scenechange / 255)
        elif 'set_scenechange' in dir(haf):
            input = haf.set_scenechange(input, scenechange)
        else:
            raise AttributeError('module \"havsfunc\" has no attribute \"SCDetect\"!')

    return core.misc.AverageFrames(input, [1 for i in range(2 * radius + 1)], scenechange=scenechange)


def FixTelecinedFades(input, mode=0, threshold=[0.0], color=[0.0], full=None, planes=None):
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

    core = vs.get_core()
    funcName = 'FixTelecinedFades'

    # set parameters
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    bits = input.format.bits_per_sample
    isFloat = input.format.sample_type == vs.FLOAT

    if isinstance(mode, int):
        mode = [mode]
    elif not isinstance(mode, list):
        raise TypeError(funcName + ': \"mode\" must be an int!')
    if len(mode) < input.format.num_planes:
        if len(mode) == 0:
            mode = [0]
        modeLength = len(mode)
        for i in range(input.format.num_planes - modeLength):
            mode.append(mode[modeLength - 1])
    for i in mode:
        if i not in [0, 1, 2]:
            raise ValueError(funcName + ': valid values of \"mode\" are 0, 1 or 2!')

    if isinstance(threshold, int) or isinstance(threshold, float):
        threshold = [threshold]
    if not isinstance(threshold, list):
        raise TypeError(funcName + ': \"threshold\" must be a list!')
    if len(threshold) < input.format.num_planes:
        if len(threshold) == 0:
            threshold = [0.0]
        thresholdLength = len(threshold)
        for i in range(input.format.num_planes - thresholdLength):
            threshold.append(threshold[thresholdLength - 1])
    if isFloat:
        for i in range(len(threshold)):
            threshold[i] = abs(threshold[i]) / 255
    else:
        for i in range(len(threshold)):
            threshold[i] = abs(threshold[i]) * ((1 << bits) - 1) / 255

    if isinstance(color, int) or isinstance(color, float):
        color = [color]
    if not isinstance(color, list):
        raise TypeError(funcName + ': \"color\" must be a list!')
    if len(color) < input.format.num_planes:
        if len(color) == 0:
            color = [0.0]
        colorLength = len(color)
        for i in range(input.format.num_planes - colorLength):
            color.append(color[colorLength - 1])
    if isFloat:
        for i in range(len(color)):
            color[i] = color[i] / 255
    else:
        for i in range(len(color)):
            color[i] = abs(color[i]) * ((1 << bits) - 1) / 255
    
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
    def GetExpr(scale, color, threshold):
        if color != 0:
            flt = 'x {color} - {scale} * {color} +'.format(scale=scale, color=color)
        else:
            flt = 'x {scale} *'.format(scale=scale)
        return flt if threshold == 0 else '{flt} x - abs {threshold} > {flt} x ?'.format(flt=flt, threshold=threshold)
    

    def Adjust(n, f, clip, core, mode, threshold, color):
        separated = core.std.SeparateFields(clip, tff=True)
        topField = core.std.SelectEvery(separated, 2, [0])
        bottomField = core.std.SelectEvery(separated, 2, [1])
        
        topAvg = f[0].props.PlaneStatsAverage
        bottomAvg = f[1].props.PlaneStatsAverage
        
        if color != 0:
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
    adjustedPlanes = {}
    for i in range(input.format.num_planes):
        if i in planes:
            inputPlane = mvf.GetPlane(input, i)
            topFieldPlanes[i] = mvf.GetPlane(topField, i).std.PlaneStats()
            bottomFieldPlanes[i] = mvf.GetPlane(bottomField, i).std.PlaneStats()
            adjustedPlanes[i] = core.std.FrameEval(inputPlane, functools.partial(Adjust, clip=inputPlane, core=core, mode=mode[i], threshold=threshold[i], color=color[i]), prop_src=[topFieldPlanes[i], bottomFieldPlanes[i]])
        else:
            adjustedPlanes[i] = None

    adjusted = core.std.ShufflePlanes([(adjustedPlanes[i] if i in planes else input_src) for i in range(input.format.num_planes)], [(0 if i in planes else i) for i in range(input.format.num_planes)], input.format.color_family)
    if not full and not isFloat:
        adjusted = core.fmtc.bitdepth(adjusted, fulls=True, fulld=False, planes=planes)
        adjusted = core.std.ShufflePlanes([(adjusted if i in planes else input_src) for i in range(input.format.num_planes)], list(range(input.format.num_planes)), input.format.color_family)
    return adjusted


def TCannyHelper(input, t_h=8.0, t_l=1.0, plane=0, returnAll=False, **canny_args):
    """A helper function for tcanny.TCanny(mode=0)

    Strong edge detected by "t_h" will be highlighted in white, and weak edge detected by "t_l" will be highlighted in gray.

    Args:
        input: Source clip. Can be 8-16 bits integer or 32 bits floating point based.

        t_h: (float) TCanny's high gradient magnitude threshold for hysteresis. Default is 8.0.

        t_l: (float) TCanny's low gradient magnitude threshold for hysteresis. Default is 1.0.

        plane: (int) Which plane to be processed. Default is 0.

        returnAll: (bint) Whether to return a tuple containing every 4 temporary clips(strongEdge, weakEdge, view, tcannyOutput) or just "view" clip.
            Default is False.

        canny_args: (dict) Additional parameters passed to core.tcanny.TCanny (except "mode" and "planes") in the form of keyword arguments.

    """

    core = vs.get_core()
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


def MergeChroma(clip1, clip2, weight=1.0):
    """A function that merges the chroma from one videoclip into another. Ported from Avisynth's equivalent.

    There is an optional weighting, so a percentage between the two clips can be specified.

    Args:
        clip1: The clip that has the chroma pixels merged into (the base clip).

        clip2: The clip from which the chroma pixel data is taken (the overlay clip).

        weight: (float) Defines how much influence the new clip should have. Range is 0.0–1.0.

    """

    core = vs.get_core()
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


def firniture(clip, width, height, kernel='binomial7', taps=None, gamma=False, **resample_args):
    '''5 new interpolation kernels (via fmtconv)
    
    Proposed by *.mp4 guy (https://forum.doom9.org/showthread.php?t=166080)
    
    Args:
        clip: Source clip.

        width, height: (int) New picture width and height in pixels.

        kernel: (string) Default is "binomial7".
            "binomial5", "binomial7": A binomial windowed sinc filter with 5 or 7 taps. Should have the least ringing of any available interpolator, except perhaps "noaliasnoring4".
            "maxflat5", "maxflat8": 5 or 8 tap interpolation that is maximally flat in the passband. In English, these filters have a sharp and relatively neutral look, but can have ringing and aliasing problems.
            "noalias4": A 4 tap filter hand designed to be free of aliasing while having acceptable ringing and blurring characteristics. Not always a good choice, but sometimes very useful.
            "noaliasnoring4": Derived from the "noalias4" kernel, but modified to have reduced ringing. Other attributes are slightly worse.

        taps: (int) Default is the last num in "kernel".
            "taps" in fmtc.resample. This parameter is now mostly superfluous. It has been retained so that you can truncate the kernels to shorter taps then they would normally use.

        gamma: (bool) Default is False.
            Set to true to turn on gamma correction for the y channel.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of keyword arguments.

    Examples:
        clip = muvsfunc.firniture(clip, 720, 400, kernel="noalias4", gamma=False)

    '''

    import nnedi3_resample as nnrs
    
    core = vs.get_core()
    funcName = 'firniture'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')
    
    impulseCoefficents = dict(
        binomial5=[8, 0, -589, 0, 11203, 0, -93355, 0, 606836, 1048576, 606836, 0, -93355, 0, 11203, 0, -589, 0, 8],
        binomial7=[146, 0, -20294, 0, 744006, 0, -11528384, 0, 94148472, 0, -487836876, 0, 2551884458, 4294967296, 2551884458, 0, -487836876, 0, 94148472, 0, -11528384, 0, 744006, 0, -20294, 0, 146],
        maxflat5=[-259, 1524, -487, -12192, 17356, 42672, -105427, -85344, 559764, 1048576, 559764, -85344, -105427, 42672, 17356, -12192, -487, 1524, -259],
        maxflat8=[2, -26, 166, -573, 912, 412, 1524, -589, -12192, 17356, 42672, -105427, -85344, 606836, 1048576, 606836, -85344, -105427, 42672, 17356, -12192, -589, 1524, 412, 912, -573, 166, -26, 2],
        noalias4=[-1, 2, 4, -6, -17, 7, 59, 96, 59, 7, -17, -6, 4, 2, -1],
        noaliasnoring4=[-1, 8, 40, -114, -512, 360, 3245, 5664, 3245, 360, -512, -114, 40, 8, -1]
        )
    
    if taps is None:
        taps = int(kernel[-1])
    
    if clip.format.bits_per_sample != 16:
        clip = mvf.Depth(clip, 16)
    
    if gamma:
        clip = nnrs.GammaToLinear(clip)
    
    clip = core.fmtc.resample(clip, width, height, kernel='impulse', impulse=impulseCoefficents[kernel], kovrspl=2, taps=taps, **resample_args)
    
    if gamma:
        clip = nnrs.LinearToGamma(clip)
    
    return clip


def BoxFilter(input, radius=16, radius_v=None, planes=None, fmtc_conv=0, radius_thr=None, resample_args=None, keep_bits=True, depth_args=None):
    '''Box filter
    
    Performs a box filtering on the input clip. Box filtering consists in averaging all the pixels in a square area whose center is the output pixel. You can approximate a large gaussian filtering by cascading a few box filters.
    
    Args:
        input: Input clip to be filtered.

        radius, radius_v: (int) Size of the averaged square. The size is (radius*2-1) * (radius*2-1). If "radius_v" is None, it will be set to "radius".
            Default is 16.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fmtc_conv: (0~2) Whether to use fmtc.resample for convolution.
            It's recommended to input clip without chroma subsampling when using fmtc.resample, otherwise the output may be incorrect.
            0: False. 1: True (except both "radius" and "radius_v" is strictly smaller than 4). 2: Auto, determined by radius_thr (exclusive).
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
    
    core = vs.get_core()
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
            raise NotImplementedError(funcName + ': Please update your VapourSynth. BoxBlur on float sample has not yet been implemented on current version.')
        elif radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2, cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
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
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2, cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                if keep_bits and input.format.bits_per_sample != flt.format.bits_per_sample:
                    flt = mvf.Depth(flt, depth=input.format.bits_per_sample, **depth_args)
                return flt

            elif core.std.get_functions().__contains__('BoxBlur'):
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur was not found
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input


def SmoothGrad(input, radius=9, thr=0.25, ref=None, elast=3.0, planes=None, **limit_filter_args):
    '''Avisynth's SmoothGrad
    
    SmoothGrad smooths the low gradients or flat areas of a 16-bit clip. It proceeds by applying a huge blur filter and comparing the result with the input data for each pixel.
    If the difference is below the specified threshold, the filtered version is taken into account, otherwise the input pixel remains unchanged.
    
    Args:
        input: Input clip to be filtered.

        radius: (int) Size of the averaged square. Its width is radius*2-1. Range is 2–9.

        thr: (float) Threshold between reference data and filtered data, on an 8-bit scale.

        ref: Reference clip for the filter output comparison. Specify here the input clip when you cascade several SmoothGrad calls.
            When undefined, the input clip is taken as reference.

        elast: (float) To avoid artifacts, the threshold has some kind of elasticity.
            Value differences falling over this threshold are gradually attenuated, up to thr * elast > 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        limit_filter_args: (dict) Additional arguments passed to mvf.LimitFilter in the form of keyword arguments.

    '''
        
    core = vs.get_core()
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


def DeFilter(input, fun, iter=10, planes=None, **fun_args):
    '''Zero-order reverse filter (arXiv:1704.04037)

    Args:
        input: Input clip to be reversed.

        fun: The function of how the input clip is filtered.

        iter: (int) Number of iterations. Default is 10.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fun_args: (dict) Additional arguments passed to "fun" in the form of keyword arguments. Alternative to functools.partial.

    '''

    core = vs.get_core()
    funcName = 'DeFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # initialization
    flt = input
    calc_expr = 'x y + z -'

    # iteration
    for i in range(iter):
        flt = core.std.Expr([flt, input, fun(flt, **fun_args)], [(calc_expr if i in planes else '') for i in range(input.format.num_planes)])
    
    return flt


def scale(val, bits):
    '''The old scale function in havsfunc.
    
    '''

    return val * ((1 << bits) - 1) // 255


def ColorBarsHD(clip=None, width=1288, height=720):
    '''Avisynth's ColorBarsHD()

    It produces a video clip containing SMPTE color bars (Rec. ITU-R BT.709 / arib std b28 v1.0) scaled to any image size.
    By default, a 1288×720, YV24, TV range, 29.97 fps, 1 frame clip is produced.

    Requirment:
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

    core = vs.get_core()
    funcName = 'ColorBarsHD'
    from mt_lutspa import lutspa
    
    if clip is not None and not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    c = round(width * 3 / 28)
    d = round((width - c * 7) / 2)
    
    p4 = round(height / 4)
    p23 = round(height / 12)
    p1 = height - p23 * 2 - p4

    blkclip_args = dict(format=vs.YUV444P8, length=1, fpsnum=30000, fpsden=1001)
    
    pattern1_colors = dict(Gray40=[104, 128, 128], White75=[180, 128, 128], Yellow=[168, 44, 136], Cyan=[145, 147, 44], Green=[134, 63, 52], Magenta=[63, 193, 204], Red=[51, 109, 212], Blue=[28, 212, 120])
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
    Y_Ramp = lutspa(Y_Ramp_tmp, mode='absolute', y_expr='220 x * {c} 7 * / 16 +'.format(c=c), chroma='copy')
    Y_Ramp = core.resize.Point(Y_Ramp, c*7, p23)
    Red100 = core.std.BlankClip(clip, d, p23, color=pattern3_colors['Red100'], **blkclip_args)
    pattern3 = core.std.StackHorizontal([Yellow100, Y_Ramp, Red100])
    
    pattern4_colors = dict(Gray15=[49, 128, 128], Black0=[16, 128, 128], White100=[235, 128, 128], Black_neg2=[12, 128, 128], Black_pos2=[20, 128, 128], Black_pos4=[25, 128, 128])
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
    pattern4 = core.std.StackHorizontal([Gray15, Black0_1, White100, Black0_2, Black_neg2, Black0_3, Black_pos2, Black0_4, Black_pos4, Black0_5, Gray15])
    
    #pattern = core.std.StackVertical([pattern1, pattern2, pattern3, pattern4])
    #return pattern1, pattern2, pattern3, pattern4
    pattern = core.std.StackVertical([pattern1, pattern2, pattern3, pattern4])
    return pattern


def SeeSaw(clp, denoised=None, NRlimit=2, NRlimit2=None, Sstr=1.5, Slimit=None, Spower=4, SdampLo=None, SdampHi=24, Szp=18, bias=49, Smode=None, sootheT=49, sootheS=0, ssx=1.0, ssy=None):
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

    Usage: (in Avisynth)
        a = TheNoisySource
        b = a.YourPreferredDenoising()
        SeeSaw( a, b, [parameters] )

    """

    core = vs.get_core()
    funcName = 'SeeSaw'
    
    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
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
    
    Szrp = Szp / pow(Sstr, 0.25) / pow((ssx + ssy) / 2, 0.5)
    SdampLo = SdampLo / pow(Sstr, 0.25) / pow((ssx + ssy) / 2, 0.5)

    ox = clp.width
    oy = clp.height
    xss = haf.m4(ox * ssx)
    yss = haf.m4(oy * ssy)
    NRL = scale(NRlimit, bits)
    NRL2 = scale(NRlimit2, bits)
    NRLL = scale(round(NRlimit2 * 100 / bias - 1), bits)
    SLIM = scale(abs(Slimit), bits)
    multiple = scale(1, bits)
    neutral = scale(128, bits)
    peak = scale(255, bits)

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
        denoised = mvf.GetPlane(denoised)
    
    NRdiff = core.std.MakeDiff(clp, denoised)

    tameexpr = 'x {NRLL} + y < x {NRL2} + x {NRLL} - y > x {NRL2} - x {BIAS1} * y {BIAS2} * + 100 / ? ?'.format(NRLL=NRLL, NRL2=NRL2, BIAS1=bias, BIAS2=100-bias)
    tame = core.std.Expr([clp, denoised], [tameexpr])
    
    head = SeeSaw_sharpen2(tame, Sstr, Spower, Szp, SdampLo, SdampHi, 4)

    if ssx == 1 and ssy == 1:
        last = core.rgvs.Repair(SeeSaw_sharpen2(tame, Sstr, Spower, Szp, SdampLo, SdampHi, Smode), head, [1])
    else:
        last = core.rgvs.Repair(SeeSaw_sharpen2(tame.fmtc.resample(xss, yss, kernel='lanczos').fmtc.bitdepth(bits=bits), Sstr, Spower, Szp, SdampLo, SdampHi, Smode), head.fmtc.resample(xss, yss, kernel='bicubic', a1=-0.2, a2=0.6).fmtc.bitdepth(bits=bits), [1]).fmtc.resample(ox, oy, kernel='lanczos').fmtc.bitdepth(bits=bits)
        
    last = SeeSaw_SootheSS(last, tame, sootheT, sootheS)
    sharpdiff = core.std.MakeDiff(tame, last)

    if NRlimit == 0:
        last = clp
    else:
        NRdiff = core.std.MakeDiff(clp, denoised)
        last = core.std.Expr([clp, NRdiff], ['y {neutral} {NRL} + > x {NRL} - y {neutral} {NRL} - < x {NRL} + x y {neutral} - - ? ?'.format(neutral=neutral, NRL=NRL)])
    
    if Slimit >= 0:
        limitexpr = 'y {neutral} {SLIM} + > x {SLIM} - y {neutral} {SLIM} - < x {SLIM} + x y {neutral} - - ? ?'.format(neutral=neutral, SLIM=SLIM)
        last = core.std.Expr([last, sharpdiff], [limitexpr])
    else:
        limitexpr = 'y {neutral} = x x y {neutral} - abs {multiple} / 1 {SLIM} / pow {multiple} * y {neutral} - y {neutral} - abs / * - ?'.format(neutral=neutral, SLIM=SLIM, multiple=multiple)
        last = core.std.Expr([last, sharpdiff], [limitexpr])

    return last if isGray else core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)


def SeeSaw_sharpen2(clp, strength, power, zp, lodmp, hidmp, rgmode):
    """Modified sharpening function from SeeSaw()

    Only the first plane (luma) will be processed.

    """

    core = vs.get_core()
    funcName = 'SeeSaw_sharpen2'
    
    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')
    
    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample
    multiple = scale(1, bits)
    neutral = scale(128, bits)
    peak = scale(255, bits)

    power = int(power)
    if power <= 0:
        raise ValueError(funcName + ': Power must be integer value 1 or more')
    power = 1 / power

    # copied from havsfunc
    def get_lut1(x):
        if x == neutral:
            return x
        else:
            tmp1 = abs(x - neutral) / multiple
            tmp2 = tmp1 ** 2
            tmp3 = zp ** 2
            return min(max(math.floor(neutral + (tmp1 / zp) ** power * zp * (strength * multiple) * (1 if x > neutral else -1) * (tmp2 * (tmp3 + lodmp) / ((tmp2 + lodmp) * tmp3)) * ((1 + (0 if hidmp == 0 else (zp / hidmp) ** 4)) / (1 + (0 if hidmp == 0 else (tmp1 / hidmp) ** 4))) + 0.5), 0), peak)
            
    method = clp.rgvs.RemoveGrain([rgmode] if isGray else [rgmode, 0])
    sharpdiff = core.std.MakeDiff(clp, method, [0]).std.Lut(function=get_lut1, planes=[0])
    return core.std.MergeDiff(clp, sharpdiff, [0])


def SeeSaw_SootheSS(sharp, orig, sootheT=25, sootheS=0):
    """Soothe() function to stabilze sharpening from SeeSaw()

    Only the first plane (luma) will be processed.

    """

    core = vs.get_core()
    funcName = 'SeeSaw_SootheSS'
    
    if not isinstance(sharp, vs.VideoNode) or sharp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
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
        orig = mvf.GetPlane(orig)
    
    expr1 = 'x {neutral} < y {neutral} < xor x {neutral} - 100 / {SSPT} * {neutral} + x {neutral} - abs y {neutral} - abs > x {SSPT} * y {i} * + 100 / x ? ?'.format(neutral=neutral, SSPT=SSPT, i=100-SSPT)
    expr2 = 'x {neutral} < y {neutral} < xor x {neutral} - 100 / {ST} * {neutral} + x {neutral} - abs y {neutral} - abs > x {ST} * y {i} * + 100 / x ? ?'.format(neutral=neutral, ST=ST, i=100-ST)

    if sootheS != 0:
        last = core.std.Expr([last, core.std.Convolution(last, [1]*9)], [expr1])
    if sootheT != 0:
        last = core.std.Expr([last, TemporalSoften(last, 1, 0)], [expr2])
    if sootheT <= -1:
        last = core.std.Expr([last, TemporalSoften(last, 1, 0)], [expr2])

    last = core.std.MakeDiff(orig, last, [0])
    return last if isGray else core.std.ShufflePlanes([last, orig_src], list(range(orig_src.format.num_planes)), orig_src.format.color_family)


def abcxyz(clp, rad=3.0, ss=1.5):
    """Avisynth's abcxyz()

    Reduces halo artifacts that can occur when sharpening.

    Author: Didée (http://avisynth.nl/images/Abcxyz_MT2.avsi)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rad: (float) Radius for halo removal. Default is 3.0.

        ss: (float) Radius for supersampling / ss=1.0 -> no supersampling. Range: 1.0 - ???. Default is 1.5

    """

    core = vs.get_core()
    funcName = 'abcxyz'
    
    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    ox = clp.width
    oy = clp.height
    
    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample
    
    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)

    x = core.resize.Bicubic(clp, haf.m4(ox/rad), haf.m4(oy/rad)).resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
    y = core.std.Expr([clp, x], ['x {a} + y < x {a} + x {b} - y > x {b} - y ? ? x y - abs * x {c} x y - abs - * + {c} /'.format(a=scale(8, bits), b=scale(24, bits), c=scale(32, bits))])

    z1 = core.rgvs.Repair(clp, y, [1])

    if ss != 1:
        maxbig = core.std.Maximum(y).resize.Bicubic(haf.m4(ox*ss), haf.m4(oy*ss))
        minbig = core.std.Minimum(y).resize.Bicubic(haf.m4(ox*ss), haf.m4(oy*ss))
        z2 = core.resize.Lanczos(clp, haf.m4(ox*ss), haf.m4(oy*ss))
        z2 = core.std.Expr([z2, maxbig, minbig], ['x y min z max']).resize.Lanczos(ox, oy)
        z1 = z2  # for simplicity
    
    if not isGray:
        z1 = core.std.ShufflePlanes([z1, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)
    
    return z1


def Sharpen(clip, amountH=1.0, amountV=None, planes=None):
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

    core = vs.get_core()
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


def Blur(clip, amountH=1.0, amountV=None, planes=None):
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

    core = vs.get_core()
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


def BlindDeHalo3(clp, rx=3.0, ry=3.0, strength=125, lodamp=0, hidamp=0, sharpness=0, tweaker=0, PPmode=0, PPlimit=None, interlaced=False):
    """Avisynth's BlindDeHalo3() version: 3_MT2

    This script removes the light & dark halos from too strong "Edge Enhancement".

    Author: Didée (https://forum.doom9.org/attachment.php?attachmentid=5599&d=1143030001)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rx, ry: (float) The radii to use for the [quasi-] Gaussian blur, on which the halo removal is based. Default is 3.0.

        strength: (int) The overall strength of the halo removal effect. Default is 125.

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

    core = vs.get_core()
    funcName = 'BlindDeHalo3'
    
    if not isinstance(clp, vs.VideoNode):
        raise TypeError(funcName + ': \"clp\" is not a clip!')

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
    TWK_HLIGHT = 'x y - abs {i} < {neutral} {TWK} {neutral} {TWK} - {TWK} {neutral} / * + {TWK0} {TWK} {LD} + / * {neutral} {TWK} - {j} / dup * {neutral} {TWK} - {j} / dup * {HD} + / * {neutral} + ?'.format(i=scale(1, bits), neutral=neutral, TWK=TWK, TWK0=TWK0, LD=LD, j=scale(20, bits), HD=HD)

    i = clp if not interlaced else core.std.SeparateFields(clp, tff=True)
    oxi = i.width
    oyi = i.height
    sm = core.resize.Bicubic(i, haf.m4(oxi/rx), haf.m4(oyi/ry))
    mm = core.std.Expr([sm.std.Maximum(), sm.std.Minimum()], ['x y - 4 *']).std.Maximum().std.Deflate().std.Convolution([1]*9).std.Inflate().resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0).std.Inflate()
    sm = core.resize.Bicubic(sm, oxi, oyi, filter_param_a=1, filter_param_b=0)
    smd = core.std.Expr([Sharpen(i, tweaker), sm], [TWK_HLIGHT])
    if sharpness != 0:
        smd = Blur(smd, sharpness)
    clean = core.std.Expr([i, smd], ['x y {neutral} - -'.format(neutral=neutral)])
    clean = core.std.MaskedMerge(i, clean, mm)

    if PPmode != 0:
        LL = scale(PPlimit, bits)
        LIM = 'x {LL} + y < x {LL} + x {LL} - y > x {LL} - y ? ?'.format(LL=LL)

        base = i if PPmode < 0 else clean
        small = core.resize.Bicubic(base, haf.m4(oxi / math.sqrt(rx * 1.5)), haf.m4(oyi / math.sqrt(ry * 1.5)))
        ex1 = Blur(small.std.Maximum(), 0.5)
        in1 = Blur(small.std.Minimum(), 0.5)
        hull = core.std.Expr([ex1.std.Maximum().rgvs.RemoveGrain(11), ex1, in1, in1.std.Minimum().rgvs.RemoveGrain(11)], ['x y - {i} - 5 * z a - {i} - 5 * max'.format(i=scale(1, bits))]).resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0)

        if abs(PPmode) == 1:
            postclean = core.std.MaskedMerge(base, small.resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0), hull)
        elif abs(PPmode) == 2:
            postclean = core.std.MaskedMerge(base, base.rgvs.RemoveGrain(19), hull)
        elif abs(PPmode) == 3:
            postclean = core.std.MaskedMerge(base, base.std.Median(), hull)
        else:
            raise ValueError(funcName + ': \"PPmode\" must be in [-3 ... 3]!')
    else:
        postclean = clean

    if PPlimit != 0:
        postclean = core.std.Expr([base, postclean], [LIM])

    last = haf.Weave(postclean, tff=True) if interlaced else postclean

    if not isGray:
        last = core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return last


def dfttestMC(input, pp=None, mc=2, mdg=False, planes=None, sigma=None, sbsize=None, sosize=None, tbsize=None, mdgSAD=None, thSAD=None, thSCD1=None, thSCD2=None, pel=None, blksize=None, search=None, searchparam=None, overlap=2, dct=None, **dfttest_params):
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

        dfttest's sigma, sbsize, sosize and tbsize re supported. Extra dfttest parameters may be passed via "dfttest_params".
        pel, thSCD, thSAD, blksize, overlap, dct, search, and searchparam are also supported.

        sigma is the main control of dfttest strength.
        tbsize should not be set higher than mc * 2 + 1.

    """

    core = vs.get_core()
    funcName = 'dfttestMC'

    if not isinstance(input, vs.VideoNode) or input.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"input\" must be a Gray or YUV clip!')

    if pp is not None:
        if not isinstance(pp, vs.VideoNode):
            raise TypeError(funcName + ': \"pp\" must be a clip!')
        if input.format.id != pp.format.id:
            raise TypeError(funcName + ': \"pp\" must be of the same format as \"input\"!')
        if input.width != pp.width or input.height != pp.height:
            raise TypeError(funcName + ': \"pp\" must be of the same size as \"input\"!')

    # Set default options. Most external parameters are passed valueless.
    if dfttest_params is None:
        dfttest_params = {}

    mc = min(mc, 5)

    # Set chroma parameters.
    if planes is None:
        planes = list(range(input.format.num_planes))
    elif not isinstance(planes, dict):
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
    pp_enabled = pp is not None
    pp_super = haf.DitherLumaRebuild(pp if pp_enabled else input, s0=1, chroma=chroma).mv.Super(pel=pel, chroma=chroma)
    super = haf.DitherLumaRebuild(input, s0=1, chroma=chroma).mv.Super(pel=pel, chroma=chroma) if pp_enabled else pp_super

    # Motion vector search.
    analysis_args = dict(chroma=chroma, search=search, searchparam=searchparam, overlap=overlap, blksize=blksize, dct=dct)
    bvec = []
    fvec = []

    for i in range(1, mc+1):
        bvec.append(core.mv.Analyse(pp_super, delta=i, isb=True, **analysis_args))
        fvec.append(core.mv.Analyse(pp_super, delta=i, isb=False, **analysis_args))

    # Optional MDegrain.
    if mdg:
        degrain_args = dict(thsad=mdgSAD, plane=plane, thscd1=thSCD1, thscd2=thSCD2)
        if mc >= 3:
            degrained = core.mv.Degrain3(input, super, bvec[0], fvec[0], bvec[1], fvec[1], bvec[2], fvec[2], **degrain_args)
        elif mc == 2:
            degrained = core.mv.Degrain2(input, super, bvec[0], fvec[0], bvec[1], fvec[1], **degrain_args)
        elif mc == 1:
            degrained = core.mv.Degrain1(input, super, bvec[0], fvec[0], **degrain_args)
        else:
            degrained = input
    else:
        degrained = input

    # Motion Compensation.
    degrained_super = haf.DitherLumaRebuild(degrained, s0=1, chroma=chroma).mv.Super(pel=pel, levels=1, chroma=chroma) if mdg else super
    compensate_args = dict(thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)
    bclip = []
    fclip = []
    for i in range(1, mc+1):
        bclip.append(core.mv.Compensate(degrained, degrained_super, bvec[i-1], **compensate_args))
        fclip.append(core.mv.Compensate(degrained, degrained_super, fvec[i-1], **compensate_args))

    # Create compensated clip.
    fclip.reverse()
    interleaved = core.std.Interleave(fclip + [degrained] + bclip) if mc >= 1 else degrained

    # Perform dfttest.
    filtered = core.dfttest.DFTTest(interleaved, sigma=sigma, sbsize=sbsize, sosize=sosize, tbsize=tbsize, **dfttest_params)

    return core.std.SelectEvery(filtered, mc * 2 + 1, mc) if mc > 1 else filtered


def TurnLeft(clip):
    """Avisynth's internel function TurnLeft()"""
    core = vs.get_core()

    return core.std.Transpose(clip).std.FlipVertical()


def TurnRight(clip):
    """Avisynth's internel function TurnRight()"""
    core = vs.get_core()

    return core.std.FlipVertical(clip).std.Transpose()


def BalanceBorders(c, cTop=0, cBottom=0, cLeft=0, cRight=0, thresh=128, blur=999):
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
        cTop, cBotteom, cLeft, cRight: (int) The number of variable pixels on each side.
            There will not be anything very terrible if you specify values that are greater than the minimum required in your case,
            but to achieve a good result, "it is better not to" ...
            Range: 2~inf for RGB input. For YUV or YCbCr input, the minimum accepted value varies depending on chroma subsampling.
                For YV24, the it's also 2~inf. For YV12, the it's 4~inf. Default is 0.
        thresh: (int) Threshold of acceptable changes for local color matching in 8 bit scale.
            Range: 0~128. Recommend: [0~16 or 128]. Default is 128.
        blur: (int) Degree of blur for local color matching. 
            Smaller values give a more accurate color match,
            larger values give a more accurate picture transfer.
            Range: 1~inf. Recommend: [1~20 or 999]. Default is 999.

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

    core = vs.get_core()
    funcName = 'BalanceBorders'

    if not isinstance(c, vs.VideoNode):
        raise TypeError(funcName + ': \"c\" must be a clip!')

    if c.format.sample_type != vs.INTEGER:
        raise TypeError(funcname+': \"c\" must be integer format!')

    if blur <= 0:
        raise ValueError(funcName + ': \'blur\' have not a correct value! (0 ~ inf]')

    if thresh <= 0:
        raise ValueError(funcName + ': \'thresh\' have not a correct value! (0 ~ inf]')

    last = c

    if cTop > 0:
        last = BalanceTopBorder(last, cTop, thresh, blur)

    last = TurnRight(last)

    if cLeft > 0:
        last = BalanceTopBorder(last, cLeft, thresh, blur)

    last = TurnRight(last)

    if cBottom > 0:
        last = BalanceTopBorder(last, cBottom, thresh, blur)

    last = TurnRight(last)

    if cRight > 0:
        last = BalanceTopBorder(last, cRight, thresh, blur)

    last = TurnRight(last)

    return last


def BalanceTopBorder(c, cTop, thresh, blur):
    """BalanceBorders()'s helper function"""
    core = vs.get_core()

    cWidth = c.width
    cHeight = c.height
    cTop = min(cTop, cHeight - 1)
    blurWidth = max(4, math.floor(cWidth / blur))
    
    c2 = mvf.PointPower(c, 1, 1)

    last = core.std.CropRel(c2, 0, 0, cTop*2, (cHeight - cTop - 1) * 2)
    last = core.resize.Point(last, cWidth * 2, cTop * 2)
    last = core.resize.Bilinear(last, blurWidth * 2, cTop * 2)
    last = core.std.Convolution(last, [1, 1, 1], mode='h')
    last = core.resize.Bilinear(last, cWidth * 2, cTop * 2)
    referenceBlur = last

    original = core.std.CropRel(c2, 0, 0, 0, (cHeight - cTop) * 2)

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

    return core.std.StackVertical([last, core.std.CropRel(c2, 0, 0, cTop * 2, 0)]).resize.Point(cWidth, cHeight)


def DisplayHistogram(clip, factor=None):
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

    core = vs.get_core()
    funcName = 'DisplayHistogram'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if clip.format.sample_type != vs.INTEGER or clip.format.bits_per_sample > 16 or clip.format.color_family != vs.YUV:
        raise TypeError(funcname+': \"clip\" must be 8..16 integer YUV format!')

    histogram_v = core.hist.Classic(clip)

    clip_8 = mvf.Depth(clip, 8)
    levels = core.hist.Levels(clip_8, factor=factor).std.CropRel(left=clip.width, right=0, top=0, bottom=clip.height - 256)
    if clip.format.bits_per_sample != 8:
        levels = mvf.Depth(levels, clip.format.bits_per_sample)
    histogram_h = TurnLeft(core.hist.Classic(clip.std.Transpose()).std.CropRel(left=clip.height))

    bottom = core.std.StackHorizontal([histogram_h, levels])

    return core.std.StackVertical([histogram_v, bottom])


def GuidedFilter(input, guidance=None, radius=4, regulation=0.01, regulation_mode=0, use_gauss=False, fast=None, subsampling_ratio=4, use_fmtc1=False, kernel1='point', kernel1_args=None, use_fmtc2=False, kernel2='bilinear', kernel2_args=None, **depth_args):
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
            If box filter is used, the range of radius is 1 ~ 12(fast=False) or 1 ~ 12*subsampling_ratio in VapourSynth R38 or older because of the limitation of std.Convolution().
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

        use_gauss: Whether to use gaussian guided filter [1]. This replaces mean filter with gaussian filter.
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
            This method reduces the time complexity from O(N) to O(N^2) for a subsampling ratio s.
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
            Default is None.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is None.

    Ref:
        [1] He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE transactions on pattern analysis and machine intelligence, 35(6), 1397-1409.
        [2] He, K., & Sun, J. (2015). Fast guided filter. arXiv preprint arXiv:1505.00996.
        [3] Li, Z., Zheng, J., Zhu, Z., Yao, W., & Wu, S. (2015). Weighted guided image filtering. IEEE Transactions on Image Processing, 24(1), 120-129.
        [4] Kou, F., Chen, W., Wen, C., & Li, Z. (2015). Gradient domain guided image filtering. IEEE Transactions on Image Processing, 24(11), 4528-4539.

    """

    core = vs.get_core()
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
        down_w = round(width / s + 0.5)
        down_h = round(height / s + 0.5)
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, **kernel1_args) if guidance is not None else p
        else: # use zimg
            p = eval('core.resize.{kernel}(p, {w}, {h}, **kernel1_args)'.format(kernel=kernel1.capitalize(), w=down_w, h=down_h))
            I = eval('core.resize.{kernel}(I, {w}, {h}, **kernel1_args)'.format(kernel=kernel1.capitalize(), w=down_w, h=down_h)) if guidance is not None else p

        r = round(r / s + 0.5)

    # Select the shape of the kernel. As the width of BoxFilter in this module is (radius*2-1) rather than (radius*2+1), radius should be increased by one.
    Filter = functools.partial(core.tcanny.TCanny, sigma=r/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=r+1)
    Filter_r1 = functools.partial(core.tcanny.TCanny, sigma=1/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=1+1)


    # Edge-Aware Weighting, equation (5) in [3], or equation (9) in [4].
    def FLT(n, f, clip, core, eps0):
        frameMean = f.props.PlaneStatsAverage

        return core.std.Expr(clip, ['x {eps0} + {avg} *'.format(avg=frameMean, eps0=eps0)])


    # Compute the optimal value of a of Gradient Domain Guided Image Filter, equation (12) in [4]
    def FLT2(n, f, cov_Ip, weight_in, weight, var_I, core, eps):
        frameMean = f.props.PlaneStatsAverage
        frameMin = f.props.PlaneStatsMin

        alpha = frameMean
        kk = -4 / (frameMin - alpha)

        return core.std.Expr([cov_Ip, weight_in, weight, var_I], ['x {eps} 1 1 1 {kk} y {alpha} - * exp + / - * z / + a {eps} z / + /'.format(eps=eps, kk=kk, alpha=alpha)])
    
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

        denominator = core.std.PlaneStats(denominator, plane=[0])
        weight = core.std.FrameEval(denominator, functools.partial(FLT, clip=weight_in, core=core, eps0=eps0), prop_src=[denominator]) # equation (5) in [3], or equation (9) in [4]
    
        if regulation_mode == 1: # Weighted Guided Image Filter
            a = core.std.Expr([cov_Ip, var_I, weight], ['x y {eps} z / + /'.format(eps=eps)])
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.PlaneStats(weight_in, plane=[0])
            a = core.std.FrameEval(weight, functools.partial(FLT2, cov_Ip=cov_Ip, weight_in=weight_in, weight=weight, var_I=var_I, core=core, eps=eps), prop_src=[weight_in])
    else: # regulation_mode == 0, Original Guided Filter
        a = core.std.Expr([cov_Ip, var_I], ['x y {} + /'.format(eps)])

    b = core.std.Expr([mean_p, a, mean_I], ['x y z * -'])
    
    mean_a = Filter(a)
    mean_b = Filter(b)
    
    # Fast guided filter's upsampling
    if fast:
        if use_fmtc2:
            mean_a = core.fmtc.resample(mean_a, width, height, kernel=kernel2, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, **kernel2_args)
        else: # use zimg
            mean_a = eval('core.resize.{kernel}(mean_a, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_b = eval('core.resize.{kernel}(mean_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))

    # Linear translation
    q = core.std.Expr([mean_a, I_src, mean_b], ['x y * z +'])
    
    # Final bitdepth conversion
    return mvf.Depth(q, depth=bits, sample=sampleType, **depth_args)


def GuidedFilterColor(input, guidance, radius=4, regulation=0.01, use_gauss=False, fast=None, subsampling_ratio=4, use_fmtc1=False, kernel1='point', kernel1_args=None, use_fmtc2=False, kernel2='bilinear', kernel2_args=None, **depth_args):
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

    """

    core = vs.get_core()
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
        down_w = round(width / s + 0.5)
        down_h = round(height / s + 0.5)
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, **kernel1_args)
        else: # use zimg
            p = eval('core.resize.{kernel}(p, {w}, {h}, **kernel1_args)'.format(kernel=kernel1.capitalize(), w=down_w, h=down_h))
            I = eval('core.resize.{kernel}(I, {w}, {h}, **kernel1_args)'.format(kernel=kernel1.capitalize(), w=down_w, h=down_h)) if guidance is not None else p

        r = round(r / s + 0.5)

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
        if use_fmtc2:
            mean_a_r = core.fmtc.resample(mean_a_r, width, height, kernel=kernel2, **kernel2_args)
            mean_a_g = core.fmtc.resample(mean_a_g, width, height, kernel=kernel2, **kernel2_args)
            mean_a_b = core.fmtc.resample(mean_a_b, width, height, kernel=kernel2, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, **kernel2_args)
        else: # use zimg
            mean_a_r = eval('core.resize.{kernel}(mean_a_r, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_a_g = eval('core.resize.{kernel}(mean_a_g, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_a_b = eval('core.resize.{kernel}(mean_a_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_b = eval('core.resize.{kernel}(mean_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))

    # Linear translation
    q = core.std.Expr([mean_a_r, I_src_r, mean_a_g, I_src_g, mean_a_b, I_src_b, mean_b], ['x y * z a * + b c * + d +'])
    
    # Final bitdepth conversion
    return mvf.Depth(q, depth=bits, sample=sampleType, **depth_args)


def Wiener2(input, radius_v=3, radius_h=None, noise=None, **depth_args):
    """2-D adaptive noise-removal filtering. (wiener2 from MATLAB)

    Wiener2 lowpass filters an intensity image that has been degraded by constant power additive noise.
    Wiener2 uses a pixel-wise adaptive Wiener method based on statistics estimated from a local neighborhood of each pixel.

    Estimate of the additive noise power will not be returned.

    Args:
        input: Input clip. Only the first plane will be processed.
        radius_v, radius_h: (int) Size of neighborhoods to estimate the local image mean and standard deviation. The size is (radius_v*2-1) * (radius_h*2-1).
            If "radius_h" is None, it will be set to "radius_v".
            Default is 3.
        noise: (float) Variance of addictive noise. If it is not given, average of all the local estimated variances will be used.
            Default is None.
        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is None.

    Ref:
        [1] Lim, J. S. (1990). Two-dimensional signal and image processing. Englewood Cliffs, NJ, Prentice Hall, 1990, 710 p, p. 538, equations 9.26, 9.27, and 9.29.
        [2] 2-D adaptive noise-removal filtering - MATLAB wiener2 - MathWorks (https://www.mathworks.com/help/images/ref/wiener2.html)

    """

    core = vs.get_core()
    funcName = 'Wiener2'

    if not isinstance(input, vs.VideoNode) or input.format.num_planes > 1:
        raise TypeError(funcName + ': \"input\" must be a gray-scale/single channel clip!')

    bits = input.format.bits_per_sample
    sampleType = input.format.sample_type

    if radius_h is None:
        radius_h = radius_v

    input32 = mvf.Depth(input, depth=32, sample=vs.FLOAT, **depth_args)

    localMean = BoxFilter(input32, radius_h+1, radius_v+1)
    localVar = BoxFilter(core.std.Expr([input32], ['x dup *']), radius_h+1, radius_v+1)
    localVar = core.std.Expr([localVar, localMean], ['x y dup * -'])

    if noise is None:
        localVarStats = core.std.PlaneStats(localVar, plane=[0])

        def FLT(n, f, clip, core, localMean, localVar):
            noise = f.props.PlaneStatsAverage

            return core.std.Expr([clip, localMean, localVar], ['y z {noise} - 0 max z {noise} max / x y - * +'.format(noise=noise)])

        flt = core.std.FrameEval(input32, functools.partial(FLT, clip=input32, core=core, localMean=localMean, localVar=localVar), prop_src=[localVarStats])
    else:
        flt = core.std.Expr([input32, localMean, localVar], ['y z {noise} - 0 max z {noise} max / x y - * +'.format(noise=noise)])

    return mvf.Depth(flt, depth=bits, sample=sampleType, **depth_args)
