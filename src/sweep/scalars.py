import torch, math
import numpy as np

def normal_grid_coes(M):
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

def staggered_grid_coes_torch(M: int) -> torch.Tensor:
    """Finite difference coefficients for the staggered grid.

    Args:
        M (length of operator): 2*M is the difference order.

    Returns:
       Array : Coefficients for the finite difference operator with shape (M,).
    """
    a = torch.zeros(M, dtype=torch.float32)
    
    for m in range(1, M + 1):
        a_m = (-1) ** (m + 1) / (2 * m - 1)
        
        prod = 1.0
        for n in range(1, M + 1):
            if n != m:
                numerator = (2 * n - 1) ** 2
                denominator = numerator - (2 * m - 1) ** 2
                prod *= abs(numerator / denominator)
        
        a_m *= prod
        a[m - 1] = a_m
    
    return a

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

    if grid == 'normal':
        # constant = normal_grid_coes(spatial_order//2)
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
            kernel[center, center] = -2*2*np.sum(constant)
    
    if mode=='z':
        # X axis
        kernel[center+1:, center] = constant # down
        kernel[0:center, center] = sign*constant[::-1] # up

    if mode=='x':
        # Z axis
        kernel[center, center+1:] = constant # right
        kernel[center, 0:center] = sign*constant[::-1] # left

    if mode in ['x', 'z'] and not no_center:
        kernel[center, center] = -2*np.sum(constant)

    return kernel.reshape(1, 1, *kernel.shape)

def generate_convolution_kernel3d(spatial_order, **kwargs):
    """Generate convolution kernel

    Args:
        n (int): The order of the taylor expansion

    Returns:
        _type_: Tensor, the convolution kernel
    """
    constant = normal_grid_coes(spatial_order//2)
    kernel_size = spatial_order + 1
    kernel = np.zeros((kernel_size, kernel_size, kernel_size), dtype=np.float32)
    center = spatial_order // 2
    # Apply along z-axis
    for i, c in enumerate(constant):
        offset = i + 1
        kernel[center - offset, center, center] = c
        kernel[center + offset, center, center] = c

    # Apply along y-axis
    for i, c in enumerate(constant):
        offset = i + 1
        kernel[center, center - offset, center] = c
        kernel[center, center + offset, center] = c

    # Apply along x-axis
    for i, c in enumerate(constant):
        offset = i + 1
        kernel[center, center, center - offset] = c
        kernel[center, center, center + offset] = c

    # Center value: minus sum of all non-center coefficients
    kernel[center, center, center] = -2 * 3 * np.sum(constant)

    return kernel