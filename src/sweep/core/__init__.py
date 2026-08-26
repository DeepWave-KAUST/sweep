"""Cross-cutting building blocks shared by the propagators and the equations.

Pure, state-free helpers pulled out of the propagator base class so each concern
lives in one cohesive place instead of as another 60 lines of a 1100-line class.
The propagator still OWNS the resulting state; these modules only compute it,
which is what makes them testable without building a propagator at all.
"""
