# cython: boundscheck=False, initializedcheck=False, language_level=3, nonecheck=False, overflowcheck=False, wraparound=False

cimport cython
from cython cimport view


cdef Py_ssize_t clamp(const Py_ssize_t val, const Py_ssize_t low, const Py_ssize_t high):
    return min(max(val, low), high)


cpdef void sigma_filter(
    const float [:, ::view.contiguous] src, float [:, ::view.contiguous] dst, 
    const int radius, const float threshold):
    """Sigma filter"""

    cdef Py_ssize_t height = src.shape[0]
    cdef Py_ssize_t width = src.shape[1]

    cdef float center, val, acc
    cdef int count, x, y, i, j

    for y in range(height):
        for x in range(width):
            center = src[y, x]

            acc = 0.
            count = 0

            for j in range(-radius, radius + 1):
                for i in range(-radius, radius + 1):
                    val = src[clamp(y + j, 0, height - 1), clamp(x + i, 0, width - 1)]

                    if abs(center - val) < threshold:
                        acc += val
                        count += 1
            
            dst[y, x] = acc / count
