"""The ``impl='c'`` autograd drivers, one module per equation family.

Each module names its struct field families once, fills the forward and
backward pybind structs through :mod:`sweep.sem.cdrive`, and holds both the
fused single-domain Function (the C entry runs the time loop) and the strip
Function of :mod:`sweep.sem.dd` (Python steps the stepped entries with one
exchange per assembly) where the family has both.
"""
