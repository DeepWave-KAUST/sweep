"""Reverse-driver dispatch shared by the boundary-saving backends.

Single source of truth for how a reverse-reconstruction driver is picked for
an equation — used by both the eager (PyTorch) path
(``PropTorch._eager_bs_reverse_mode``) and the JAX path
(``_jax_boundary_saving.resolve_reverse_mode``)::

    2nd-order             -> 'swap2nd'  (reuses func with time levels swapped)
    1st-order + substeps  -> 'substep'  (reuses interior_substeps reversed)

1st-order equations without ``interior_substeps`` are unsupported (no
approximate fallback), and a 'substep' reconstruction under a free surface is
refused unless the equation declares ``supports_bs_free_surface`` (its
substeps carry the free-surface BC).  Each backend passes its own driver
``registry`` and a backend-appropriate ``alternatives`` hint for the error
messages.
"""


def resolve_reverse_mode(equation, registry, *, forced=None, free_surface=False,
                         label="boundary saving",
                         alternatives="chunk checkpointing (use_ckpt=True)"):
    """Pick (and validate) the reverse driver for ``equation``.

    Args:
        equation: The wave-equation instance.
        registry: Mapping of driver name -> driver for this backend.
        forced: Force a driver name (None auto-dispatches on the time scheme).
        free_surface: Whether the propagator runs with a free surface.
        label: Feature name used in error messages.
        alternatives: Suggested fallbacks appended to error messages.

    Returns:
        The driver name (a key of ``registry``).
    """
    from sweep.equations.base import SecondOrderEquation
    is_2nd = isinstance(equation, SecondOrderEquation)
    has_substeps = callable(getattr(equation, "interior_substeps", None))
    if forced is not None:
        if forced not in registry:
            raise ValueError(
                f"Unknown {label} reverse mode {forced!r}; "
                f"registered drivers: {sorted(registry)}."
            )
        if forced == "substep" and not has_substeps:
            raise ValueError(
                f"reverse mode 'substep' needs {type(equation).__name__}"
                ".interior_substeps()."
            )
        reverse_mode = forced
    elif is_2nd:
        reverse_mode = "swap2nd"
    elif has_substeps:
        reverse_mode = "substep"
    else:
        raise NotImplementedError(
            f"{label.capitalize()} for the 1st-order equation "
            f"'{type(equation).__name__}' requires an interior_substeps() "
            "hook (the exact reverse driver). Add one (reusing the "
            f"equation's step core), or use {alternatives}."
        )
    if (
        reverse_mode == "substep"
        and free_surface
        and not getattr(equation, "supports_bs_free_surface", False)
    ):
        raise NotImplementedError(
            f"{label.capitalize()} with a free surface is not supported for "
            f"{type(equation).__name__} (its interior_substeps would drop "
            f"the free-surface BC). Options: free_surface=False, or "
            f"{alternatives}. Equations whose interior_substeps carry the "
            "free-surface BC set supports_bs_free_surface=True."
        )
    return reverse_mode
