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

    if not (isinstance(src, vs.VideoNode) and isinstance(flt, vs.VideoNode)):
        raise TypeError(funcName + ': This is not a clip')
    if src.format.id != flt.format.id:
        raise TypeError(funcName + ': Clips must have the same format')

    isGray = src.format.color_family == vs.GRAY
    bits = src.format.bits_per_sample 

    expr = 'x y - abs {power} pow'.format(power=power)

    chroma = chroma or isGray

    return core.std.Expr([src, flt], [expr] if chroma else [expr, '{neutral}'.format(neutral=1 << (bits - 1))])
