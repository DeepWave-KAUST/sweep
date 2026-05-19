"""Unified factory for SWEEP's acoustic anisotropic wave equations.

This module deliberately does **not** wrap a solver (``PropTorch``).  The
equation classes and the propagator are separate concepts in SWEEP; the
factory's only job is to dispatch ``(method, symmetry, ndim)`` to the
correct equation class and return a ready-to-use *equation instance*.
The user then constructs ``PropTorch`` themselves and feeds the instance
in, exactly as they would with any raw equation class.

Dispatch table:

    method        | symmetry | ndim | class
    --------------+----------+------+-------------------------------------
    "duveneck"    | vti      | 2    | AcousticVTI1st        (Duveneck 2008)
    "duveneck"    | vti      | 3    | AcousticVTI1st3D
    "liang"       | vti      | 2    | AcousticVTI           (Liang K. 2022)
    "liang"       | tti      | 2    | AcousticTTI
    "alkhalifah"  | vti / tti| 2    | AcousticTariq         (Alkhalifah 2000)

Typical use::

    from sweep.equations import AcousticAniso
    from sweep.propagator.torch import PropTorch

    eq = AcousticAniso(method="duveneck", symmetry="vti", ndim=2,
                       spatial_order=4, device="cuda", backend="torch")
    prop = PropTorch(eq, shape=(nz, nx),
                     source_type=["sH", "sV"], receiver_type=["vz"],
                     dh=10.0, dt=1e-3, abcn=30)
    record = prop(wavelet, sources, receivers,
                  models=[vp, eps, delta, rho])

Mirrors the legacy ``DAS`` facade naming but is intentionally simpler:
no embedded propagator, no DAS-style post-processing.  For the raw
classes use ``AcousticVTI1st`` / ``AcousticVTI`` / ``AcousticTariq`` /
``AcousticTTI`` / ``AcousticVTI1st3D`` directly.
"""

from __future__ import annotations


_VALID_METHODS = {"duveneck", "liang", "alkhalifah"}
_VALID_SYMMETRIES = {"vti", "tti"}


def _normalize_method(value: str) -> str:
    v = str(value).lower()
    if v not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{value}'. Supported: {sorted(_VALID_METHODS)}."
        )
    return v


def _normalize_symmetry(value: str) -> str:
    v = str(value).lower()
    if v not in _VALID_SYMMETRIES:
        raise ValueError(
            f"Unknown symmetry '{value}'. Supported: {sorted(_VALID_SYMMETRIES)}."
        )
    return v


def _select_equation_class(method: str, symmetry: str, ndim: int):
    """Resolve ``(method, symmetry, ndim)`` → equation class.

    Imports are lazy so this module stays cheap to import even when only
    the dispatch table is needed.
    """
    method = _normalize_method(method)
    symmetry = _normalize_symmetry(symmetry)
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3; got {ndim}.")

    from . import acoustic_vti_1st as _avti1
    from . import qP_vti as _qpv
    from . import qP_tariq as _qpt

    if method == "duveneck":
        if symmetry != "vti":
            raise NotImplementedError(
                "Duveneck (2008) is a VTI formulation; TTI extension via "
                "Bond rotation is a future spec."
            )
        return _avti1.AcousticVTI1st if ndim == 2 else _avti1.AcousticVTI1st3D

    if method == "liang":
        if ndim != 2:
            raise NotImplementedError(
                "Liang K. et al. (2022) classes are 2-D only in this repo; "
                "use method='duveneck' for 3-D VTI."
            )
        if symmetry == "vti":
            return _qpv.AcousticVTI
        try:
            from . import qP_tti as _qpt2
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "AcousticTTI (Liang TTI variant) requires the qP_tti module."
            ) from exc
        return _qpt2.AcousticTTI

    if method == "alkhalifah":
        if ndim != 2:
            raise NotImplementedError(
                "Alkhalifah η-formulation (AcousticTariq) is 2-D only."
            )
        # The η-acoustic class covers both VTI and TTI (TTI emerges via
        # rotated η; see qP_tariq.py for details).
        return _qpt.AcousticTariq

    raise AssertionError(f"unhandled method {method!r}")


class AcousticAniso:
    """Factory class — instantiating returns an equation instance.

    Construction is intentionally lightweight: ``__new__`` resolves the
    target equation class and constructs it with the user-supplied
    keyword arguments, returning that instance directly.  ``isinstance(x,
    AcousticAniso)`` will therefore be ``False`` — ``x`` is the actual
    equation class (e.g. ``AcousticVTI1st``).

    Parameters
    ----------
    method : {"duveneck", "liang", "alkhalifah"}, default "duveneck"
    symmetry : {"vti", "tti"}, default "vti"
    ndim : {2, 3}, default 2
    spatial_order : int, default 4
    device : str, default "cpu"
    backend : str, default "torch"
    **equation_kwargs : forwarded to the resolved equation constructor.

    Examples
    --------
    >>> from sweep.equations import AcousticAniso
    >>> from sweep.propagator.torch import PropTorch
    >>> eq = AcousticAniso(method="duveneck", symmetry="vti", ndim=2)
    >>> isinstance(eq, AcousticAniso)
    False
    >>> type(eq).__name__
    'AcousticVTI1st'
    """

    def __new__(
        cls,
        *,
        method: str = "duveneck",
        symmetry: str = "vti",
        ndim: int = 2,
        spatial_order: int = 4,
        device: str = "cpu",
        backend: str = "torch",
        **equation_kwargs,
    ):
        equation_cls = _select_equation_class(method, symmetry, ndim)
        return equation_cls(
            spatial_order=spatial_order,
            device=device,
            backend=backend,
            **equation_kwargs,
        )

    # ------------------------------------------------------------------
    # Static introspection helpers — useful when callers want to look up
    # the resolved class without actually instantiating it.
    # ------------------------------------------------------------------

    @staticmethod
    def equation_class(*, method: str = "duveneck",
                        symmetry: str = "vti",
                        ndim: int = 2):
        """Return the equation *class* (not instance) for the given keys."""
        return _select_equation_class(method, symmetry, ndim)

    @staticmethod
    def supported():
        """Return a list of supported (method, symmetry, ndim) triples."""
        return [
            ("duveneck",   "vti", 2),
            ("duveneck",   "vti", 3),
            ("liang",      "vti", 2),
            ("liang",      "tti", 2),
            ("alkhalifah", "vti", 2),
            ("alkhalifah", "tti", 2),
        ]
