# stokes-flow
Software for simulations of Stokes flow (low Reynolds number flow) with slip boundary conditions.

Finite elements are used to discretize the governing equations. The following velocity-pressure element pairs are considered:
  - Taylor-Hood: P2 + P1 continuous Lagrange polynomials 
  - Divergence conforming: 1st order Brezzi-Douglas-Marini + 0th order discontinuous Galerkin
