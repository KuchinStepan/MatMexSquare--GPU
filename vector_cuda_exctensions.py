from numba import cuda


@cuda.jit
def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        -(a[0] * b[2] - a[2] * b[0]),
        a[0] * b[1] - b[0] * a[1]
    )


@cuda.jit
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit
def norm(a):
    return dot(a, a) ** 0.5


@cuda.jit
def mul(vect, scal):
    return (
        vect[0] * scal,
        vect[1] * scal,
        vect[2] * scal
    )


@cuda.jit
def add(a, b):
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2]
    )


@cuda.jit
def get_mixed_product(x, y, z):
    return dot(cross(x, y), z)
