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

def generate_convolution_kernel(spatial_order):
    """Generate convolution kernel

    Args:
        spatial_order (int): The order of the taylor expansion(Must be even)

    Returns:
        Array: The convolution kernel with shape (spatial_order+1, spatial_order+1).
    """

    constant = normal_grid_coes(spatial_order//2)
    kernel_size = spatial_order + 1
    kernel = np.zeros((kernel_size, kernel_size),dtype=np.float32)
    center = spatial_order // 2
    # Z axis
    kernel[center, center+1:] = constant
    kernel[center, 0:center] = constant[::-1]
    # X axis
    kernel[center+1:, center] = constant
    kernel[0:center, center] = constant[::-1]
    # Center
    kernel[center, center] = -2*2*np.sum(constant)

    return kernel.reshape(1, 1, *kernel.shape)