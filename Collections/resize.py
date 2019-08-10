def resize(clip, w=None, h=None, sx=0, sy=0, sw=None, sh=None, kernel="spline36", a1=None, a2=None, mpeg2_cplace=True):
    """Experimental wrapper function for vszimg resizer in a fmtconv-like API"""

    assert core.version_number() >= 44

    def _expand(shift, num_planes):
        if isinstance(shift, (int, float)):
            return [shift for _ in range(num_planes)]
        elif len(shift) > 0:
            _shift = list(shift)
            while len(_shift) < num_planes:
                _shift.append(_shift[-1])
            return _shift

    if w is None:
        w = clip.width
    if h is None:
        h = clip.height
    if sw is None:
        sw = clip.width
    if sh is None:
        sh = clip.height

    kernel = kernel.capitalize()
    if kernel == "Bicubic":
        a1, a2 = 0, 0.5
    elif kernel == "Lanczos":
        a1 = 3

    num_planes = clip.format.num_planes
    sx = _expand(sx, num_planes)
    sy = _expand(sy, num_planes)
    sw = _expand(sw, num_planes)
    sh = _expand(sh, num_planes)

    if num_planes == 1:
        res = eval(f"core.resize.{kernel}")(clip, w, h, src_left=sx[0], src_top=sy[0], src_width=sw[0], src_height=sh[0], filter_param_a=a1, filter_param_b=a2)
    else:
        # copied from nnedi3_resample.py
        hSubS = 1 << clip.format.subsampling_w
        hCPlace = 0.5 - hSubS / 2 if mpeg2_cplace else 0
        hScale = w / clip.width

        vSubS = 1 << clip.format.subsampling_h
        vCPlace = 0
        vScale = h / clip.height

        planes = [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in range(num_planes)]
        for i in range(num_planes):
            if i == 0:
                planes[i] = eval(f"core.resize.{kernel}")(planes[0], w, h, src_left=sx[0], src_top=sy[0], src_width=sw[0], src_height=sh[0], filter_param_a=a1, filter_param_b=a2)
            else:
                planes[i] = eval(f"core.resize.{kernel}")(planes[i], w // (1 << clip.format.subsampling_w), h // (1 << clip.format.subsampling_h), src_left=((sx[i]-hCPlace) * hScale + hCPlace) / hScale / hSubS, src_top=((sy[i]-vCPlace) * vScale + vCPlace) / vScale / vSubS, src_width=sw[i] // (1 << clip.format.subsampling_w), src_height=sh[i] // (1 << clip.format.subsampling_h), filter_param_a=a1, filter_param_b=a2)
        res = core.std.ShufflePlanes(planes, [0] * num_planes, clip.format.color_family)

    return res
