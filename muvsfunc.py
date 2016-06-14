import vapoursynth as vs
import havsfunc as haf

def AAMerge(Bsrc, aa_h, aa_v, mrad=0, power=1.0, show=0):
    core = vs.get_core()
    funcName = 'AAMerge'
    
    if not isinstance(Bsrc, vs.VideoNode):
        raise TypeError(funcName + ': \"Bsrc\" must be a clip!')
         
    if not isinstance(aa_h, vs.VideoNode):
        raise TypeError(funcName + ': \"aa_h\" must be a clip!')
    if Bsrc.format.id != aa_h.format.id:
        raise ValueError(funcName + ': clip \"aa_h\" must be of the same format as the Bsrc clip!')
    if Bsrc.width != aa_h.width or Bsrc.height != aa_h.height:
        raise ValueError(funcName + ': clip \"aa_h\" must be of the same size as the Bsrc clip!')
    
    if not isinstance(aa_v, vs.VideoNode):
        raise TypeError(funcName + ': \"aa_v\" must be a clip!')
    if Bsrc.format.id != aa_v.format.id:
        raise ValueError(funcName + ': clip \"aa_v\" must be of the same format as the Bsrc clip!')
    if Bsrc.width != aa_v.width or Bsrc.height != aa_v.height:
        raise ValueError(funcName + ': clip \"aa_v\" must be of the same size as the Bsrc clip!')   
        
    if not isinstance(mrad, int):
        raise TypeError(funcName + '\"mrad\" must be an int!')
    
    if not isinstance(power, float):
        raise TypeError(funcName + '\"power\" must be a float!')

    if not isinstance(show, int):
        raise TypeError(funcName + '\"show\" must be an int!')
    
    hmap = core.std.Convolution(Bsrc, matrix=[-1, 2, -1, -1, 2, -1, -1, 2, -1], saturate=False)
    vmap = core.std.Convolution(Bsrc, matrix=[-1, -1, -1, 2, 2, 2, -1, -1, -1], saturate=False)
    if mrad > 0:
        hmap = haf.mt_expand_multi(hmap, sw=mrad, sh=0)
        vmap = haf.mt_expand_multi(vmap, sw=0, sh=mrad)
    elif mrad < 0:
        hmap = haf.mt_inpand_multi(hmap, sw=-mrad, sh=0)
        vmap = haf.mt_inpand_multi(vmap, sw=0, sh=-mrad)  
    
    bits = Bsrc.format.bits_per_sample
    ldexpr = '{peak} 1 y x / {power} pow + /'.format(peak=(1 << bits) - 1, power=power)
    ldmap = core.std.Expr([hmap, vmap], [ldexpr])
        
    if show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap
    else:
        return core.std.MaskedMerge(aa_h, aa_v, ldmap)

def Compare(src, flt, power=1.5, chroma=False):
    core = vs.get_core()
    funcName = 'Compare'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if not isinstance(flt, vs.VideoNode):
        raise TypeError(funcName + ': \"flt\" must be a clip!')
    if src.format.id != flt.format.id:
        raise TypeError(funcName + ': \"src\" and \"flt\" must have the same format')

    isGray = src.format.color_family == vs.GRAY
    bits = src.format.bits_per_sample

    expr = 'x y - abs 1 + {power} pow 1 -'.format(power=power)

    chroma = chroma or isGray

    if bits < 16:
        src = core.fmtc.bitdepth(src, bits=16)
        flt = core.fmtc.bitdepth(flt, bits=16)
        diff = core.std.Expr([src, flt], [expr] if chroma else [expr, '{neutral}'.format(neutral=1 << (16 - 1))])
        diff = core.fmtc.bitdepth(diff, bits=bits, dmode=1)
    else:
        diff = core.std.Expr([src, flt], [expr] if chroma else [expr, '{neutral}'.format(neutral=1 << (bits - 1))])

    return diff

def MaskProcess(clip, mrad=0, msmooth=0, mblur=0, mode='rectangle', planes=[0, 1, 2]):
    core = vs.get_core()
    funcName = 'MaskProcess'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')
    
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
            if mode != 'rectangle' and mode != 'losange' and mode != 'ellipse':
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
