""" Python port of kewenyu's VapourSynth-BezierCurve (https://github.com/kewenyu/VapourSynth-BezierCurve)

"""

import vapoursynth as vs
import muvsfunc as muf
import math

def normalize(x, bits, range):
    """ Internal used function

    """

    scale = ((1 << bits) - 1) // 255

    if range == 0:
        return x / (255 * scale)
    else:
        if x < 16 * scale:
            return 0
        elif x > 255 * scale:
            result = 1
        else:
            return (x - 16 * scale) / (219 * scale)


def quadraticBezierCurve(clip, accur=0.01, input_range=0, begin=None, end=None, x1=None, y1=None, planes=None):
    """ Python port of core.bezier.Quadratic()
        
    Args:
        You can refer to https://github.com/kewenyu/VapourSynth-BezierCurve

    Diff:
        1. "begin", "end", "x1", "y1" are in 8 bit scale, and their default values are changed to 0, 255, 128, 128.
        2. Add parameter "planes". By default, every plane will be processed.

    """

    core = vs.get_core()
    funcName = 'quadraticBezierCurve'

    def bezierT(x, accur, x1):
        t = 0
        absDiffLast = 2 # Anything larger than 1

        while t <= 1 + accur:
            xCalculated = bezierX(t, x1)
            absDiff = abs(xCalculated - x)

            if absDiff >= absDiffLast:
                return t

            absDiffLast = absDiff
            t += accur

        return 0


    def bezierX(t, x1):
        return 2 * (1 - t) * t * x1 + t ** 2


    def bezierY(t, begin, y1, end):
        return  begin * (1 - t)**2 + 2 * (1 - t) * t * y1 + end * t**2


    bits = clip.format.bits_per_sample
    scale = ((1 << bits) - 1) // 255

    # Set default values
    begin = 0 if begin is None else muf.scale(begin, bits)
    x1 = muf.scale(85, bits) if x1 is None else muf.scale(x1, bits)
    y1 = muf.scale(85, bits) if y1 is None else muf.scale(y1, bits)
    end = muf.scale(255, bits) if end is None else muf.scale(end, bits)
    
    if planes is None:
        planes = list(range(clip.format.num_planes))
    
    # Check parameters
    if clip.format.sample_type == vs.FLOAT or clip.format.bits_per_sample not in [8, 16]:
        raise TypeError(funcName + ': only constant format of 8bit or 16bit integer input is supported')
    
    if input_range not in [0, 1]:
        raise ValueError(funcName + ': range must be 0 - PC range or 1 - TV range')
    
    if accur < 0 or accur > 1:
        raise ValueError(funcName + ': accur must be between 0 and 1')
    
    if begin < 0 or begin > 255 * scale:
        raise ValueError(funcName + ': begin must be between 0 and {}'.format(255 * scale))
    
    if end < 0 or end > 255 * scale:
        raise ValueError(funcName + ': end must be between 0 and {}'.format(255 * scale))
    
    if x1 < 0 or x1 > 255 * scale:
        raise ValueError(funcName + ': x1 must be between 0 and {}'.format(255 * scale))
    
    # Allocate memory for lut table
    x1 = normalize(x1, bits, input_range)

    lutTable = list(range(255 * scale + 1))

    for i in range(len(lutTable)):
        t = bezierT(normalize(i, bits, input_range), accur, x1)
        y = math.floor(bezierY(t, begin, y1, end))
        if y < 0:
            lutTable[i] = 0
        elif y > 255 * scale:
            lutTable[i] = 255 * scale
        lutTable[i] = min(max(y, 0), 255 * scale)

    return core.std.Lut(clip, lut=lutTable, planes=planes)


def cubicBezierCurve(clip, accur=0.01, input_range=0, begin=None, end=None, x1=None, y1=None, x2=None, y2=None, planes=None):
    """ Python port of kewenyu's core.bezier.Cubic()

    Args:
        You can refer to https://github.com/kewenyu/VapourSynth-BezierCurve
        
    Diff:
        1. "begin", "end", "x1", "y1", "x2", "y2" are in 8 bit scale and their default values are changed to 0, 255, 85, 85, 170, 170.
        2. Add parameter "planes". By default, every plane will be processed.

    """

    core = vs.get_core()
    funcName = 'cubicBezierCurve'

    def bezierT(x, accur, x1, x2):
        t = 0
        absDiffLast = 2 # Anything larger than 1

        while t <= 1 + accur:
            xCalculated = bezierX(t, x1, x2)
            absDiff = abs(xCalculated - x)

            if absDiff >= absDiffLast:
                return t

            absDiffLast = absDiff
            t += accur

        return 0


    def bezierX(t, x1, x2):
        return 3 * t * x1 * (1 - t)**2 + 3 * (1 - t) * x2 * t**2 + t**3


    def bezierY(t, begin, y1, y2, end):
        return begin * (1 - t)**3 + 3 * t * y1 * (1 - t)**2 + 3 * (1 - t) * y2 * t**2 + end * t**3


    bits = clip.format.bits_per_sample
    scale = ((1 << bits) - 1) // 255

    # Set default values
    begin = 0 if begin is None else muf.scale(begin, bits)
    x1 = muf.scale(85, bits) if x1 is None else muf.scale(x1, bits)
    y1 = muf.scale(85, bits) if y1 is None else muf.scale(y1, bits)
    x2 = muf.scale(170, bits) if x2 is None else muf.scale(x2, bits)
    y2 = muf.scale(170, bits) if y2 is None else muf.scale(y2, bits)
    end = muf.scale(255, bits) if end is None else muf.scale(end, bits)
    
    if planes is None:
        planes = list(range(clip.format.num_planes))
    
    # Check parameters
    if clip.format.sample_type == vs.FLOAT or clip.format.bits_per_sample not in [8, 16]:
        raise TypeError(funcName + ': only constant format of 8bit or 16bit integer input is supported')
    
    if input_range not in [0, 1]:
        raise ValueError(funcName + ': range must be 0 - PC range or 1 - TV range')
    
    if accur < 0 or accur > 1:
        raise ValueError(funcName + ': accur must be between 0 and 1')
    
    if begin < 0 or begin > 255 * scale:
        raise ValueError(funcName + ': begin must be between 0 and {}'.format(255 * scale))
    
    if end < 0 or end > 255 * scale:
        raise ValueError(funcName + ': end must be between 0 and {}'.format(255 * scale))
    
    if x1 < 0 or x1 > 255 * scale:
        raise ValueError(funcName + ': x1 must be between 0 and {}'.format(255 * scale))
    
    if x2 < 0 or x2 > 255 * scale:
        raise ValueError(funcName + ': x2 must be between 0 and {}'.format(255 * scale))
    
    if x1 >= x2:
        raise ValueError(funcName + ': x1 must be smaller than x2')
    
    # Allocate memory for lut table
    x1 = normalize(x1, bits, input_range)
    x2 = normalize(x2, bits, input_range)

    lutTable = list(range(255 * scale + 1))

    for i in range(len(lutTable)):
        t = bezierT(normalize(i, bits, input_range), accur, x1, x2)
        y = math.floor(bezierY(t, begin, y1, y2, end))
        if y < 0:
            lutTable[i] = 0
        elif y > 255 * scale:
            lutTable[i] = 255 * scale
        lutTable[i] = min(max(y, 0), 255 * scale)

    return core.std.Lut(clip, lut=lutTable, planes=planes)
