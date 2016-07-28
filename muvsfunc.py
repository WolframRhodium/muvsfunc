import vapoursynth as vs
import havsfunc as haf
import mvsfunc as mvf

def LDMerge(flt_h, flt_v, src, mrad=0, power=1.0, show=0, planes=None):
    core = vs.get_core()
    funcName = 'LDMerge'
    
    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
         
    if not isinstance(flt_h, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_h\" must be a clip!')
    if src.format.id != flt_h.format.id:
        raise ValueError(funcName + ': clip \"flt_h\" must be of the same format as the src clip!')
    if src.width != flt_h.width or src.height != flt_h.height:
        raise ValueError(funcName + ': clip \"flt_h\" must be of the same size as the src clip!')
    
    if not isinstance(flt_v, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_v\" must be a clip!')
    if src.format.id != flt_v.format.id:
        raise ValueError(funcName + ': clip \"flt_v\" must be of the same format as the src clip!')
    if src.width != flt_v.width or src.height != flt_v.height:
        raise ValueError(funcName + ': clip \"flt_v\" must be of the same size as the src clip!')   
        
    if not isinstance(mrad, int):
        raise TypeError(funcName + '\"mrad\" must be an int!')
    
    if not isinstance(power, float) and not isinstance(power, int):
        raise TypeError(funcName + '\"power\" must be a float or an int!')

    if not isinstance(show, int):
        raise TypeError(funcName + '\"show\" must be an int!')
    if show not in list(range(0, 4)):
        raise ValueError(funcName + '\"show\" must be in [0, 1, 2, 3]!')
    
    if planes is None:
        planes = list(range(flt_h.format.num_planes))

    bits = flt_h.format.bits_per_sample
    isGray = flt_h.format.color_family == vs.GRAY
    
    hmap = core.std.Convolution(src, matrix=[-1, -1, -1, 2, 2, 2, -1, -1, -1], saturate=False, planes=planes)
    vmap = core.std.Convolution(src, matrix=[-1, 2, -1, -1, 2, -1, -1, 2, -1], saturate=False, planes=planes)
    if mrad > 0:
        hmap = haf.mt_expand_multi(hmap, sw=0, sh=mrad, planes=planes)
        vmap = haf.mt_expand_multi(vmap, sw=mrad, sh=0, planes=planes)
    elif mrad < 0:
        hmap = haf.mt_inpand_multi(hmap, sw=0, sh=-mrad, planes=planes)
        vmap = haf.mt_inpand_multi(vmap, sw=-mrad, sh=0, planes=planes)
    
    ldexpr = '{peak} 1 x 0.0001 + y 0.0001 + / {power} pow + /'.format(peak=(1 << bits) - 1, power=power)
    ldmap = core.std.Expr([hmap, vmap], [ldexpr] if isGray else [ldexpr if 0 in planes else '', ldexpr if 1 in planes else '', ldexpr if 2 in planes else ''])

    if show == 0:
        return core.std.MaskedMerge(flt_h, flt_v, ldmap, planes=planes)
    elif show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap

def Compare(src, flt, power=1.5, chroma=False, mode=1):
    core = vs.get_core()
    funcName = 'Compare'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"src\" must be a YUV clip!')
    if not isinstance(flt, vs.VideoNode):
        raise TypeError(funcName + ': \"flt\" must be a clip!')
    if src.format.id != flt.format.id:
        raise TypeError(funcName + ': \"src\" and \"flt\" must have the same format')
    if mode not in [1, 2]:
        raise ValueError(funcName + ': \"mode\" must be in [1, 2]!')

    isGray = src.format.color_family == vs.GRAY
    bits = src.format.bits_per_sample

    expr = {}
    expr[1] = 'y x - abs 1 + {power} pow 1 -'.format(power=power)
    expr[2] = 'y x - {scale} * {neutral} +'.format(scale=32768 / (65536 ** (1 / power) - 1), neutral=32768)

    chroma = chroma or isGray

    if bits < 16:
        src = core.fmtc.bitdepth(src, bits=16)
        flt = core.fmtc.bitdepth(flt, bits=16)
        diff = core.std.Expr([src, flt], [expr[mode]] if chroma else [expr[mode], '{neutral}'.format(neutral=32768)])
        diff = core.fmtc.bitdepth(diff, bits=bits, dmode=1, fulls=True, fulld=True)
    else:
        diff = core.std.Expr([src, flt], [expr[mode]] if chroma else [expr[mode], '{neutral}'.format(neutral=32768)])

    return diff

def MaskProcess(clip, mrad=0, msmooth=0, mblur=0, mode='rectangle', planes=None):
    core = vs.get_core()
    funcName = 'MaskProcess'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if planes is None:
        planes = list(range(clip.format.num_planes))
    
    # internel functions
    def ExInpand(clip, mode=None, planes=None, mrad=None):
        if isinstance(mode, int):
            if mode == 0:
                mode = 'rectangle'
            elif mode == 1:
                mode = 'losange'
            elif mode == 2:
                mode = 'ellipse'
        if isinstance(mode, str):
            mode = mode.lower()
            if mode not in ['rectangle', 'losange', 'ellipse']:
                raise ValueError(funcName + ': \"mode\" must be an int in [0, 2] or a specific string in [\"rectangle\", \"losange\", \"ellipse\"]!')
        else:
            raise TypeError(funcName + ': \"mode\" must be an int in [0, 2] or a specific string in [\"rectangle\", \"losange\", \"ellipse\"]!')
        
        if isinstance(mrad, int):
            sw = mrad
            sh = mrad
        else:
            sw = mrad[0]
            sh = mrad[1]
        
        if sw * sh < 0:
            raise TypeError(funcName + ': \"mrad\" at a time must be both positive or negative!')
        
        if sw > 0 or sh > 0:
            return haf.mt_expand_multi(clip, mode=mode, planes=planes, sw=sw, sh=sh)
        else:
            return haf.mt_inpand_multi(clip, mode=mode, planes=planes, sw=-sw, sh=-sh)

    def InDeflate(clip, planes=None, radius=None):
        if radius > 0:
            return haf.mt_inflate_multi(clip, planes=planes, radius=radius)
        else:
            return haf.mt_deflate_multi(clip, planes=planes, radius=-radius)

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if isinstance(mrad, int):
        mrad = [mrad]
    if isinstance(mode, str) or isinstance(mode, int):
        mode = [mode]
    
    if not isinstance(mode, list):
        raise TypeError(funcName + ': \"mode\" must be an int, a string, a list of ints, a list of strings or a list of mixing ints and strings!')
                
    # ExInpand
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
            clip = ExInpand(clip, mode=mode[i], planes=planes, mrad=mrad[i])
    else:
        raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')

    # InDeflate
    if isinstance(msmooth, int):
        clip = InDeflate(clip, planes=planes, radius=msmooth)
    elif isinstance(msmooth, list):
        for m in msmooth:
            if not isinstance(m, int):
                raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')
            else:
                clip = InDeflate(clip, planes=planes, radius=m)
    else:
        raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')

    clip = MultiRemoveGrain(clip, mode=mblur, loop=1)

    return clip

def MultiRemoveGrain(clip, mode=0, loop=1):
    core = vs.get_core()
    funcName = 'MultiRemoveGrain'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if isinstance(mode, int):
        mode = [mode]

    if not isinstance(loop, int):
        raise TypeError(funcName + ': \"loop\" must be an int!')
    if loop < 0:
        raise ValueError(funcName + ': \"loop\" must be positive value!')

    if isinstance(mode, list):
        for i in range(loop):
            for m in mode:
                clip = core.rgvs.RemoveGrain(clip, mode=m)
    else:
        raise TypeError(funcName + ': \"mode\" must be an int, a list of ints or a list of a list of ints!')

    return clip

### GradFun3 by Firesledge v0.0.1
### Port by Muonium  2016/6/18
### Port from Dither_tools v1.27.2 (http://avisynth.nl/index.php/Dither_tools)
### Currently only smode=1 and smode=2 is implemented in VapourSynth.
### Internal calculation precision is always 16 bits
### Removed parameters list: 
###     "dthr", "wmin", "thr_edg", "subspl", "lsb_in"
### Parameters "y", "u", "v" are changed into "planes"
def GradFun3(src, thr=None, radius=None, elast=None, mask=None, mode=None, ampo=None, ampn=None,
             pat=None, dyn=None, lsb=None, staticnoise=None, smode=None, thr_det=None,
             debug=None, thrc=None, radiusc=None, elastc=None, planes=None, ref=None):
    core = vs.get_core()
    funcName = 'GradFun3'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcName + ': \"clip\" must be YUV, GRAY or YCOCG color family!')

    if thr is None:
        thr = 0.35
    elif isinstance(thr, int) or isinstance(thr, float):
        if thr < 0.1 or thr > 10.0:
            raise ValueError(funcName + ': \"thr\" must be in [0.1, 10.0]!')
    else:
        raise TypeError(funcName + ': \"thr\" must be an int or a float!')

    if radius is None:
        radius = 16 if src.width > 1024 or src.height > 576 else 12
    elif isinstance(radius, int):
        if radius <= 0:
            raise ValueError(funcName + '\"radius\" must be strictly positive.')
    else:
        raise TypeError(funcName + '\"radius\" must be an int!')

    if elast is None:
        elast = 3.0
    elif isinstance(elast, int) or isinstance(elast, float):
        if elast < 1:
            raise ValueError(funcName + ':valid range of \"elast\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elast\" must be an int or a float!')

    if mask is None:
        mask = 2
    elif not isinstance(mask, int):
        raise TypeError(funcName + ': \"mask\" must be an int!')

    if lsb is None:
        lsb = False

    if smode is None:
        smode = 1
    elif smode not in [0, 1, 2, 3]:
        raise ValueError(funcName + ': \"thr\" must be in [0, 1, 2, 3]!')
    if smode == 0:
        if radius not in list(range(1, 68+1)):
            raise ValueError(funcName + ': \"radius\" must be in 1-68 for smode=0 !')
    elif smode == 1:
        if radius not in list(range(1, 128+1)):
            raise ValueError(funcName + ': \"radius\" must be in 1-128 for smode=1 !')

    if thr_det is None:
        thr_det = 2 + round(max(thr - 0.35, 0) / 0.3)
    elif isinstance(thr_det, float):
        if thr_det <= 0.0:
            raise ValueError(funcName + '" \"thr_det\" must be strictly positive!')
    else:
        raise TypeError(funcName + ': \"mask\" must be a float!')

    if debug is None:
        debug = False
    elif not isinstance(debug, bool) and debug not in [0, 1]:
        raise TypeError(funcName + ': \"debug\" must be a bool!')

    if thrc is None:
        thrc = thr
    elif thrc < 0.1 or thrc > 10.0:
        raise ValueError(funcName + ': \"thrc\" must be in [0.1, 10.0]!')

    if radiusc is None:
        radiusc = radius
    elif isinstance(radiusc, int):
        if radiusc <= 0:
            raise ValueError(funcName + '\"radiusc\" must be strictly positive.')
    else:
        raise TypeError(funcName + '\"radiusc\" must be an int!')
    if smode == 0:
        if radiusc not in list(range(1, 68+1)):
            raise ValueError(funcName + ': \"radiusc\" must be in 1-68 for smode=0 !')
    elif smode == 1:
        if radiusc not in list(range(1, 128+1)):
            raise ValueError(funcName + ': \"radiusc\" must be in 1-128 for smode=1 !')

    if elastc is None:
        elastc = elast
    elif isinstance(elastc, int) or isinstance(elastc, float):
        if elastc < 1:
            raise ValueError(funcName + ':valid range of \"elastc\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elastc\" must be an int or a float!')

    if planes is None:
        planes = list(range(src.format.num_planes))

    if ref is None:
        ref = src
    elif not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"ref\" must be a clip!')
    if ref.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcName + ': \"ref\" must be YUV, GRAY or YCOCG color family!')

    bits = src.format.bits_per_sample
    src_16 = core.fmtc.bitdepth(src, bits=16, planes=planes) if bits < 16 else src
    src_8 = core.fmtc.bitdepth(src, bits=8, planes=[0]) if bits != 8 else src
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
        flt = core.std.ShufflePlanes([flt_y, flt_c], [0, 1, 2], src.format.color_family)
    else:
        flt = flt_y

    # Edge/detail mask

    td_lo = max(thr_det * 0.75, 1.0)
    td_hi = max(thr_det, 1.0)
    mexpr = 'x {tl} - {th} {tl} - / 255 *'.format(tl=td_lo - 0.0001, th=td_hi+ 0.0001)

    if mask > 0:
        dmask = mvf.GetPlane(src_8, 0)
        dmask = Build_gf3_range_mask(dmask)
        dmask = core.std.Expr([dmask], [mexpr])
        dmask = core.rgvs.RemoveGrain([dmask], [22])
        if mask > 1:
            dmask = core.rgvs.RemoveGrain([dmask], [11])
            if mask > 2:
                dmask = core.rgvs.RemoveGrain([dmask], [20])
        dmask = core.fmtc.bitdepth(dmask, bits=16)
        res_16 = core.std.MaskedMerge(flt, src_16, dmask, planes=planes, first_plane=True)
    else:
        res_16 = flt

    # Dithering

    result = res_16 if lsb else core.fmtc.bitdepth(res_16, bits=bits, planes=planes, dmode=mode, ampo=ampo,
                                                   ampn=ampn, dyn=dyn, staticnoise=staticnoise, patsize=pat)

    if debug:
        last = mvf.GetPlane(dmask, 0)
        if lsb:
            last = core.fmtc.bitdepth(last, bits=16)
    else:
        last = result

    return last

def GF3_smooth(src_16, ref_16, smode, radius, thr, elast, planes):
    core = vs.get_core()

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
    core = vs.get_core()
    raise RuntimeError(funcName + ': SmoothGrad has not been ported to VapourSynth!')
    '''
    ela_2 = max(elast * 0.83, 1.0)
    ela_3 = max(elast * 0.67, 1.0)
    r2 = radius * 2 // 3
    r3 = radius * 3 // 3
    r4 = radius * 4 // 4
    last = src
    last = SmoothGrad(radius=r2, thr=thr, elast=elast, ref=ref, planes=planes) if r2 >= 1 else last
    last = SmoothGrad(radius=r3, thr=thr * 0.7, elast=ela_2, ref=ref, planes=planes) if r3 >= 1 else last
    last = SmoothGrad(radius=r4, thr=thr * 0.46, elast=ela_3, ref=ref, planes=planes) if r4 >= 1 else last
    return last
    '''

def GF3_smoothgrad_multistage_3(src, radius, thr, elast, planes):
    core = vs.get_core()
    raise RuntimeError(funcName + ': SmoothGrad has not been ported to VapourSynth!')
    '''
    ref = SmoothGrad(src, radius=radius // 3, thr=thr * 0.8, elast=elast)
    last = Boxfilter(src, radius=radius, planes=planes)
    last = Boxfilter(last, radius=radius, planes=planes)
    last = mvf.LimitFilter(last, src, thr=thr * 0.6, elast=elast, ref=ref, planes=planes)
    return last
    '''

def GF3_dfttest(src, ref, radius, thr, elast, planes):
    core = vs.get_core()
    hrad = max(radius * 3 // 4, 1)
    last = core.dfttest.DFTTest(src, sigma=hrad * thr * thr * 32, sbsize=hrad * 4,
                                sosize=hrad * 3, tbsize=1, planes=planes)
    last = mvf.LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)

    return last

def GF3_bilateral_multistage(src, ref, radius, thr, elast, planes):
    core = vs.get_core()
    thr_1 = max(thr * 4.5, 1.25)
    thr_2 = max(thr * 9, 5.0)
    r4 = max(radius * 4 / 3, 4.0)
    r2 = max(radius * 2 / 3, 3.0)
    r1 = max(radius * 1 / 3, 2.0)

    last = src
    last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r4, sigmaR=thr_1 / 255, planes=planes, algorithm=0) if r4 >= 2 else last
    last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r2, sigmaR=thr_2 / 255, planes=planes, algorithm=0) if r2 >= 2 else last
    last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r1, sigmaR=thr_2 / 255, planes=planes, algorithm=0) if r1 >= 2 else last

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

def AnimeEdgeMask(clip, shift1=0, shift2=None, thY1=0, thY2=255):
# Only the first plane of "clip" would be processd
# For Anime's ringing mask, it's recomended to set "shift1" to about 0.75.
    core = vs.get_core()
    funcName = 'AnimeEdgeMask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if clip.format.color_family != vs.GRAY:
        clip = mvf.GetPlane(clip, 0)

    bits = clip.format.bits_per_sample

    if shift2 is None:
        shift2 = shift1
    
    peak = (1 << bits) - 1

    thY1 = haf.scale(thY1)
    thY2 = haf.scale(thY2)
    
    fmtc_args = dict(fulls=True, fulld=True)
    mask1 = core.std.Convolution(clip, [0, 2, -1, 0, -1, 0, 0, 0, 0], saturate=True).fmtc.resample(sx=shift1, sy=shift2, **fmtc_args)
    mask2 = core.std.Convolution(clip, [0, 0, 0, 0, -1, 0, -1, 2, 0], saturate=True).fmtc.resample(sx=-shift1, sy=-shift2, **fmtc_args)
    mask3 = core.std.Convolution(clip, [-1, 0, 0, 2, -1, 0, 0, 0, 0], saturate=True).fmtc.resample(sx=shift1, sy=-shift2, **fmtc_args)
    mask4 = core.std.Convolution(clip, [0, 0, 0, 0, -1, 2, 0, 0, -1], saturate=True).fmtc.resample(sx=-shift1, sy=shift2, **fmtc_args)

    expr = 'x x * y y * + z z * + a a * + sqrt'
    mask = core.std.Expr([mask1, mask2, mask3, mask4], [expr])

    limitexpr = 'x {thY1} < 0 x {thY2} >= {peak} x ? ?'.format(thY1=thY1, thY2=thY2, peak=peak)
    mask = core.std.Expr([mask], [limitexpr])

    if bits != 16:
        mask = core.fmtc.bitdepth(mask, bits=bits, dmode=1, **fmtc_args)

    return mask
