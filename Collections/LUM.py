"""LUM filters for VapourSynth

    Ref:
        [1] Hardie, R. C., & Boncelet, C. (1993). LUM filters: a class of rank-order-based filters for smoothing and sharpening. IEEE transactions on signal processing, 41(3), 1061-1076.
"""

def lum_smoother(input, k):
    low = muf.Sort(input, k, mode='min')
    max = muf.Sort(input, k, mode='max')

    return haf.Clamp(input, max, min)  # or just core.rgvs.RemoveGrain(input, k - 1)


def lum_sharper(input, l=2):
    if l not in range(1, 6):
        raise ValueError("\'l\' must be in [1, 5] !")

    low1 = muf.Sort(input, l, mode='min')
    high1 = muf.Sort(input, l, mode='max')

    return core.std.Expr([input, low1, high1], ['x y z + 2 / <= x y min x z max ?'])


def lum_filter(input, k=3, l=4):
    if (not isinstance(l, int)) or (not isinstance(l, int)) or (l < k):
        raise ValueError("\'k\' and \'l\' must be in [1, 5] and \'k\' must be not greater than \'l\'!")

    low_k = muf.Sort(input, k, mode='min')
    low_l = muf.Sort(input, l, mode='min')
    high_l = muf.Sort(input, l, mode='max')
    high_k = muf.Sort(input, k, mode='max')

    return core.std.Expr([input, low_k, low_l, high_l, high_k], ['x z a + 2 / <= x y < y x z min ? x b > b x a max ? ?', ''])


def asymmetric_lum_filter(input, k=3, l=4, q=6, r=7):
    if (not isinstance(k, int)) or (not isinstance(l, int)) or (not isinstance(q, int)) or (not isinstance(r, int)) or (not 1 <= k <= l <= q <= r <= 9):
        raise ValueError("\'k\', \'l\', \'q\' and \'r\' must be in [1, 9] in ascending order!")

    order_k = muf.Sort(input, k, mode='min')
    order_l = muf.Sort(input, l, mode='min')
    order_q = muf.Sort(input, q, mode='min')
    order_r = muf.Sort(input, r, mode='min')

    return core.std.Expr([input, order_k, order_l, order_q, order_r], 
        ['x z a + 2 / <= x y < y x z min ? x b > b x a max ? ?'])