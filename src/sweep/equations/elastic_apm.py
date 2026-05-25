"""``ElasticAPM`` — author-named alias for :class:`Elastic` running the
Cao & Chen (2018) Adaptive Parameter-Modified free-surface path.

``Elastic`` dispatches to APM whenever the user passes a topography via
the propagator's ``topography=`` argument with
``topo_method='apm'`` (the default for elastic equations).
Importing ``ElasticAPM`` documents the intent — "this run uses APM" —
but it is literally :class:`Elastic`.

Example
-------

.. code-block:: python

    from sweep.equations import ElasticAPM   # or Elastic; same class
    from sweep.propagator.torch import PropTorch

    eq = ElasticAPM(spatial_order=4, device='cuda')
    prop = PropTorch(eq, shape=(nz, nx), topography=air_mask_2d,
                     abcn=30, dh=10.0, dt=1e-3, impl='eager')
    record = prop(wavelet, sources, receivers, models=[vp, vs, rho])
"""

from __future__ import annotations

from .elastic import Elastic

# Public alias — keep the original name importable so existing user code
# `from sweep.equations import ElasticAPM` continues to work unchanged.
ElasticAPM = Elastic

__all__ = ["ElasticAPM"]
