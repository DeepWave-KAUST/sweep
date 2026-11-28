# Derivative products

Jacobian-, Gauss-Newton- and Hessian-vector products of a propagator's misfit
in one call: exact by forward mode where the propagator carries a tangent, by
central differences of first derivatives everywhere else. Usage guide:
[Second derivatives](../user-guide/propagators.md#second-derivatives-hessian-vector-products);
the routes compared in the
[Hessian-vector products notebook](../notebooks/46_hessian_vector_product.ipynb).

::: sweep.autodiff
    options:
      members:
        - jacobian_vector_product
        - gauss_newton_vector_product
        - hessian_vector_product
        - hessian_vector_products
