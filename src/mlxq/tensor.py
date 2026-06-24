import mlx.core as mx


def kron(a: mx.array, b: mx.array) -> mx.array:
    """Kronecker product for 2D square matrices using reshape broadcasting.

    Produces A ⊗ B with shape (a0*b0, a1*b1) by building a 4D outer product
    with dimensions (a0, b0, a1, b1), avoiding incorrect 3D broadcasting.
    """
    a0, a1 = a.shape
    b0, b1 = b.shape
    A4 = mx.reshape(a, (a0, 1, a1, 1))
    B4 = mx.reshape(b, (1, b0, 1, b1))
    prod = A4 * B4
    return mx.reshape(prod, (a0 * b0, a1 * b1))
