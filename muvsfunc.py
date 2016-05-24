import vapoursynth as vs
import havsfunc as haf

def AAMerge(Bsrc, aa_h, aa_v, rad=0, power=1.0, show=0):
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
        
    if not isinstance(rad, int):
        raise TypeError(funcName + '\"rad\" must be an int!')
    
    if not isinstance(power, float):
        raise TypeError(funcName + '\"power\" must be a float!')

    if not isinstance(show, int):
        raise TypeError(funcName + '\"show\" must be an int!')
     
    h_matrix = [-1, 2, -1, -1, 2, -1, -1, 2, -1]  
    v_matrix = [-1, -1, -1, 2, 2, 2, -1, -1, -1]  
    
    hmap = core.std.Convolution(Bsrc, h_matrix, saturate=False)
    vmap = core.std.Convolution(Bsrc, v_matrix, saturate=False)
    if rad > 0:
        hmap = haf.mt_expand_multi(hmap, sw=rad, sh=0)
        vmap = haf.mt_expand_multi(vmap, sw=0, sh=rad)
    elif rad < 0:
        hmap = haf.mt_inpand_multi(hmap, sw=-rad, sh=0)
        vmap = haf.mt_inpand_multi(vmap, sw=0, sh=-rad)  
    
    bits = Bsrc.format.bits_per_sample
    ldexpr = '{peak} 1 y x / {power} pow + /'.format(peak=(1 << bits) - 1, power=power)
    ldmap = core.std.Expr([hmap, vmap], [ldexpr])
    
    if show == 0:
        return core.std.MaskedMerge(aa_h, aa_v, ldmap)
    elif show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap
