# Optimization Algorithms Library 

## Summary

This library implements classic numerical optimization techniques for unconstrained problems (1D and multi-D). It aims to provide educational and reusable implementations of methods such as:

- Unidimensional (unconstrained):
  - Bisection (on derivative) (`soon`)
  - Golden section (`soon`)
  - Fibonacci (`soon`)
  - Unidimensional Newton

- Multidimensional (unconstrained):
  - Gradient / steepest descent 
  - Multivariate Newton (Hessian usage)
  - Numerical gradient and Hessian approximations

First we focus on unconstrained problems. 

### Main structure (relevant files in `src/`)

- `Optimizer_Uns_Base.py` — `UnconstrainedOptimizerBase` with numerical gradient/Hessian approximations and stopping utilities.
- `Unidimensional_Uns_optimizer.py` — 1D methods (Newton).
- `Multidimensional_Uns_Optimizer.py` — Multi-D methods (steepest descent, Newton), plotting helpers.
- `utils.py` (`soon`)



