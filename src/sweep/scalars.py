import math
from fractions import Fraction
import numpy as np

def regular_grid_coes(M):
    """Finite difference coefficients for the normal grid.

    Args:
        M (length of operator): 2*M is the difference order.

    Returns:
       Array : Coefficients for the finite difference operator with shape (M,).
    """
    a_m = np.zeros(M, dtype=np.float32)
    
    for m in range(1, M + 1):
        product = 1.0
        for n in range(1, M + 1):
            if n != m:
                product *= np.abs(n**2 / (n**2 - m**2))
        a_m[m - 1] = (-1)**(m + 1) / (m**2) * product

    return a_m

def staggered_grid_coes(M):
    """Finite difference coefficients for the staggered grid.

    Args:
        M (length of operator): 2*M is the difference order.

    Returns:
       Array : Coefficients for the finite difference operator with shape (M,).
    """
    a = np.zeros(M, dtype=np.float32)
    
    for m in range(1, M + 1):
        a_m = (-1) ** (m + 1) / (2 * m - 1)
        
        prod = 1.0
        for n in range(1, M + 1):
            if n != m:
                numerator = (2 * n - 1) ** 2
                denominator = numerator - (2 * m - 1) ** 2
                prod *= np.abs(numerator / denominator)
        
        a_m *= prod
        a[m - 1] = a_m
    
    return a

def fd_coefficients(derivative_order=2, accuracy_order=2):
    
    n = accuracy_order // 2
    num_points = 2 * n + 1
    points = np.arange(-n, n + 1, dtype=np.float64)

    # Taylor expansion matrix
    matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        matrix[i, :] = points ** i

    rhs = np.zeros(num_points)
    rhs[derivative_order] = math.factorial(derivative_order)

    coefficients = np.linalg.solve(matrix, rhs)

    return coefficients[-n:]

def second_derivative_coefficients_fp32(accuracy_order):
    """Centred second-derivative coefficients in float32, ``(c0, c)`` with
    ``c0 + 2 * sum(c) == 0`` exactly in their float32 values, so the stencil
    ``c0 u_0 + sum_k c_k (u_k + u_-k)`` is zero on a constant field.

    Rounding each fp64 coefficient to float32 on its own leaves the sum off by
    up to ~1e-7 (7.45e-8 at 4th order).  In the wave equation that is a term
    ``v^2 (sum / h^2) u``, whose relative effect, ~ sum / (k h)^2, grows as the
    grid is refined: past ~100 points per wavelength the record stops
    converging.  The residual goes into ``c[-1]``, the smallest coefficient and
    the only one where it is exactly representable; it moves by < 6e-8.  The
    CUDA kernels of orders 4, 6 and 8 hold the same values
    (``operators/laplace.cuh``).
    """
    c = fd_coefficients(2, accuracy_order)
    c0 = np.float32(-2 * c.sum())
    c = c.astype(np.float32)
    residual = Fraction(float(c0)) + 2 * sum(Fraction(float(x)) for x in c)
    if residual:
        target = Fraction(float(c[-1])) - residual / 2
        c[-1] = np.float32(float(target))
        if Fraction(float(c[-1])) != target:
            raise RuntimeError(f"no float32 second-derivative stencil of order {accuracy_order} "
                               "sums to zero through its last coefficient")
    return c0, c

def generate_convolution_kernel(spatial_order, derivative_order=2, mode='full', no_center=False, grid='normal', sign=1):
    """Generate convolution kernel

    Args:
        spatial_order (int): The order of the taylor expansion(Must be even)

    Returns:
        Array: The convolution kernel with shape (spatial_order+1, spatial_order+1).
    """
    assert mode in ['full', 'x', 'z'], "mode must be one of ['full', 'x', 'z']"
    assert spatial_order % 2 == 0, "spatial_order must be even"
    assert grid in ['normal', 'staggered'], "grid must be one of ['normal', 'staggered']"

    c0 = None   # the centre coefficient, when the stencil fixes it
    if grid == 'normal':
        # constant = normal_grid_coes(spatial_order//2)
        if derivative_order == 2:
            c0, constant = second_derivative_coefficients_fp32(spatial_order)
        else:
            constant = fd_coefficients(derivative_order, spatial_order)
    elif grid == 'staggered':
        constant = staggered_grid_coes(spatial_order//2)

    kernel_size = spatial_order + 1
    kernel = np.zeros((kernel_size, kernel_size),dtype=np.float32)
    center = spatial_order // 2
    if mode=='full':
        # X axis
        kernel[center, center+1:] = constant
        kernel[center, 0:center] = constant[::-1]
        # Z axis
        kernel[center+1:, center] = constant
        kernel[0:center, center] = constant[::-1]
        # Center
        if not no_center:
            kernel[center, center] = 2 * c0 if c0 is not None else -2*2*np.sum(constant)
    
    if mode=='z':
        # X axis
        kernel[center+1:, center] = constant # down
        kernel[0:center, center] = sign*constant[::-1] # up

    if mode=='x':
        # Z axis
        kernel[center, center+1:] = constant # right
        kernel[center, 0:center] = sign*constant[::-1] # left

    if mode in ['x', 'z'] and not no_center:
        kernel[center, center] = c0 if c0 is not None else -2*np.sum(constant)

    return kernel.reshape(1, 1, *kernel.shape)


def fd_weights(derivative_order, offsets):
    """Finite-difference weights of the ``derivative_order``-th derivative at 0
    from samples at ``offsets`` (in grid spacings, any spacing, e.g. half-integers
    for a staggered grid), by Fornberg's recursion (Math. Comp. 51, 1988).

    The recursion runs in exact rational arithmetic (``fractions.Fraction``) and
    rounds once, to float64, so the weights of a high-order stencil carry no
    cancellation error.  ``offsets`` must be distinct and number more than
    ``derivative_order``.  Returns a float64 array, one weight per offset.
    """
    from fractions import Fraction

    m = int(derivative_order)
    xs = [Fraction(o).limit_denominator(1 << 20) for o in offsets]
    n = len(xs)
    if m < 0:
        raise ValueError(f"derivative_order must be >= 0, got {derivative_order!r}")
    if n <= m:
        raise ValueError(f"a derivative of order {m} needs more than {m} points, got {n}")
    if len(set(xs)) != n:
        raise ValueError(f"offsets must be distinct, got {list(offsets)!r}")
    # c[k][j]: weight of sample j for the k-th derivative, over the first i+1 samples.
    c = [[Fraction(0)] * n for _ in range(m + 1)]
    c[0][0] = Fraction(1)
    c1 = Fraction(1)
    for i in range(1, n):
        c2 = Fraction(1)
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:                       # the new sample's weights, from the old c[.][i-1]
                for k in range(min(i, m), -1, -1):
                    prev = c[k - 1][i - 1] if k else Fraction(0)
                    c[k][i] = c1 / c2 * (k * prev - xs[i - 1] * c[k][i - 1])
            for k in range(min(i, m), -1, -1):
                prev = c[k - 1][j] if k else Fraction(0)
                c[k][j] = (xs[i] * c[k][j] - k * prev) / c3
        c1 = c2
    return np.array([float(w) for w in c[m]], dtype=np.float64)


def fd_stencil(derivative_order, accuracy_order, stagger=None):
    """The symmetric stencil of the ``derivative_order``-th derivative with error
    ``O(h**accuracy_order)``: ``(shifts, weights)``.

    ``stagger=None``: centred on a grid point; the result is at that point.
    ``stagger="forward"`` / ``"backward"`` (odd orders only): the result is half a
    cell after / before the point, from the samples on either side of it -- the
    staggered-grid convention of :class:`sweep.operators.StaggeredDerivative`.
    ``shifts`` are integer sample offsets from the output's grid point ``i``:
    the derivative at ``i`` (or ``i +- 1/2``) is ``sum(w * u[i + s])``.
    ``accuracy_order`` must be even (a symmetric stencil gains it in steps of two).
    """
    n, p = int(derivative_order), int(accuracy_order)
    if n < 1:
        raise ValueError(f"derivative_order must be >= 1, got {derivative_order!r}")
    if p < 2 or p % 2:
        raise ValueError(f"accuracy_order must be even and >= 2, got {accuracy_order!r}")
    if stagger is None:
        npts = p + n - (n % 2 == 0)              # odd: p + n - 1 for even n, p + n for odd n
        half = npts // 2
        shifts = np.arange(-half, half + 1)
        return shifts, fd_weights(n, shifts)
    if stagger not in ("forward", "backward"):
        raise ValueError(f"stagger must be None, 'forward' or 'backward', got {stagger!r}")
    if n % 2 == 0:
        raise ValueError(f"a staggered stencil takes an odd derivative order (it moves the "
                         f"result half a cell); got {n}: use stagger=None")
    half = (p + n - 1) // 2                      # p + n - 1 samples, half-integer offsets
    offsets = np.arange(-half, half) + 0.5       # -(half - 1/2) ... half - 1/2
    w = fd_weights(n, offsets)
    if stagger == "forward":                     # output at i + 1/2: sample i + s has offset s - 1/2
        shifts = np.arange(-half + 1, half + 1)
    else:                                        # output at i - 1/2: sample i + s has offset s + 1/2
        shifts = np.arange(-half, half)
    return shifts, w
