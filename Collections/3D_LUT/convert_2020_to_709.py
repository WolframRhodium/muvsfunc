import numpy as np
import math
import matplotlib.pyplot as plt

# Implementation of Annex 5 of ITU-R BT.2407-0
# main function: convert_2020_to_709
# TODO: Vectorization
# this function will be used to generate 3D LUT file in the future

# Constants
RGB2XYZ_2020 = np.array([[0.637, 0.145, 0.169], [0.263, 0.678, 0.059], [0., 0.028, 1.061]])
RGB2XYZ_709 = np.array([[0.412, 0.358, 0.181], [0.213, 0.715, 0.072], [0.019, 0.119, 0.951]])
D65_XYZ = np.array([95.047, 100., 108.883])


def convert_2020_to_709(pixel_rgb, matrix_2020, matrix_709, D65_XYZ, beta=0.2):
    """pixel-wise colour gamut conversion from BT.2020 to BT.709

    Implementation of Annex 5 of ITU-R BT.2407-0.

    Args:
        rgb: an 1-D array of linear RGB data, rgb.shape == (3, )
        matrix_*: 3x3 matrix used to convert from RGB to XYZ
        ...

    Ref:
        [1] Series, B. T. "Colour gamut conversion from Recommendation ITU-R BT. 2020 to Recommendation ITU-R BT. 709.", (2017).

    TODO: Vectorization
    """

    pixel_xyz = matrix_2020 @ pixel_rgb
    pixel_y = pixel_xyz[1]
    pixel_uv = np.array([4 * pixel_xyz[0], 9 * pixel_xyz[1]]) / (pixel_xyz[0] + 15 * pixel_xyz[1] + 3 * pixel_xyz[2])

    D65_uv = np.array([4 * D65_XYZ[0], 9 * D65_XYZ[1]]) / (D65_XYZ[0] + 15 * D65_XYZ[1] + 3 * D65_XYZ[2])

    vertex_2020 = _find_vertex(matrix_2020, pixel_y, D65_uv)
    intersection_2020 = _find_intersection(pixel_uv, D65_uv, vertex_2020)
    w_p2020 = np.linalg.norm(intersection_2020 - D65_uv)

    vertex_709 = _find_vertex(matrix_709, pixel_y, D65_uv)
    intersection_709 = _find_intersection(pixel_uv, D65_uv, vertex_709)
    w_p709 = np.linalg.norm(intersection_709 - D65_uv)

    alpha = w_p2020 / w_p709 - 1

    def f(r, alpha, beta):
        if r <= 1 - beta:
            return r
        elif r <= 1 + alpha:
            return r - alpha / (beta - alpha)**2 * (beta - math.sqrt(beta**2 + (alpha - beta) * (r + beta - 1)))**2 # the formula in the report is wrong
        else:
            return 1

    c_709 = D65_uv + f(np.linalg.norm(pixel_uv - D65_uv) / w_p709, alpha, beta) * (intersection_709 - D65_uv) # the formula in the report is wrong

    x, z = _yuv2xz(pixel_y, c_709[0], c_709[1])

    res_xyz = np.array([x, pixel_y, z])

    res_rgb = np.linalg.solve(RGB2XYZ_709, res_xyz)

    return res_rgb


def _find_vertex(matrix, Y, white_point):
    # find intersection on RGB cube

    # matrix: 3x3 matrix
    vertex_list = []

    eps = 1e-7

    for unknown in range(3):
        for i in range(2):
            for j in range(2):
                index1 = 0 if unknown != 0 else 1
                index2 = 3 - index1 - unknown
                tmp = (Y - i * matrix[1, index1] - j * matrix[1, index2]) / matrix[1, unknown]

                if 0 <= tmp <= 1:
                    point = np.empty(3)
                    point[index1] = i
                    point[index2] = j
                    point[unknown] = tmp
                    vertex = matrix[[0, 2], :] @ point
                    vertex[:] = np.array([4 * vertex[0], 9 * Y]) / (vertex[0] + 15 * Y + 3 * vertex[1] + eps)
                    vertex_list.append(vertex)

    # vertex sort
    comp = lambda x: math.atan2(x[1] - white_point[1], x[0] - white_point[0])
    vertex_list.sort(key=comp)

    new_vertex_list = []
    last_in_vertex = -1

    for vertex in vertex_list:
        if not np.allclose(vertex, last_in_vertex):
            new_vertex_list.append(vertex)
            last_in_vertex = vertex

    return np.asarray(new_vertex_list)


def _seg_intersect(a1, a2, b1, b2):
    # return intersection of line a1-a2 and b1-b2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = np.array([-da[1], da[0]])
    denom = dap @ db
    num = dap @ dp
    return num / denom * db + b1


def _find_intersection(pixel_uv, white_point_uv, vertex_list):
    # find projections on effective source and target gamuts
    length = len(vertex_list)
    for i in range(length):
        vertex_a, vertex_b = vertex_list[i], vertex_list[(i+1) % length]
        intersection = _seg_intersect(pixel_uv, white_point_uv, vertex_a, vertex_b)

        distance1 = np.linalg.norm(intersection - pixel_uv)
        distance2 = np.linalg.norm(intersection - white_point_uv)

        if distance1 < distance2:
            return intersection


def _yuv2xz(y, u, v):
    # Convert CIE 1976 𝑢’𝑣’ to CIE 1931 XZ
    return 9 * u * y / (4 * v), (12 - 3 * u - 20 * v) * y / (4 * v)

"""
rgb = np.array([0.3, 0.4, 0.5])
print(convert_2020_to_709(rgb, RGB2XYZ_2020, RGB2XYZ_709, D65_XYZ))
"""

"""
rgb = np.array([0.263/0.263, 0, 0])

vertex_2020, vertex_709 = convert_2020_to_709(rgb, RGB2XYZ_2020, RGB2XYZ_709)

plt.scatter(vertex_709[:, 0], vertex_709[:, 1])
plt.scatter(vertex_2020[:, 0], vertex_2020[:, 1])
plt.show()
"""