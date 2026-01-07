import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import unittest
from inverse_scattering_jax.src.helmholtz import (
  HelmholtzSolver, extend_model, Array
)
from inverse_scattering_jax.src.inverse_scattering import (
  IncomingDirections, 
  create_forward_with_adjoint, 
  get_projection_op, 
  misfit
)
from typing import Any
import jaxopt

class TestInverseScattering(unittest.TestCase):
  def test_full_inverse_pipeline(self) -> None:
    # Parameters.
    nxint: int = 20
    nyint: int = 20
    npml: int = 5
    nx: int = nxint + 2 * npml
    ny: int = nyint + 2 * npml
    h: float = 1.0 / (nxint - 1)
    omega: float = 2.0
    sigma_max: float = 10.0
    order: int = 2
    
    # Solver.
    solver = HelmholtzSolver(
      nx, ny, npml, h, omega, sigma_max, order, mode='stencil'
    )
    
    # Incoming directions.
    n_theta: int = 4
    inc = IncomingDirections(nx, ny, npml, h, omega, n_theta)
    
    # Projection (sample at some points).
    theta_r: Array = jnp.linspace(0, 2 * jnp.pi, 4)
    points_query: Array = 0.4 * jnp.stack(
      [jnp.cos(theta_r), jnp.sin(theta_r)], axis=1
    )
    
    # Grid x, y including PML, shifted to be centered at 0.
    x: Array = (jnp.arange(nx) - npml - nxint//2) * h
    y: Array = (jnp.arange(ny) - npml - nyint//2) * h
    
    projection_op = get_projection_op(x, y, points_query)
    
    # Forward model with custom adjoint.
    forward_fun = create_forward_with_adjoint(solver, inc, projection_op)
    
    # True perturbation (a small square).
    eta_true: Array = jnp.zeros((nyint, nxint))
    eta_true = eta_true.at[nyint//4:nyint//2, nxint//4:nxint//2].set(0.1)
    eta_true_flat: Array = eta_true.flatten()
    
    # Generate "data".
    print("Generating synthetic data...")
    data: Array = forward_fun(eta_true_flat)
    
    # Objective function.
    def objective(eta_flat: Array) -> Array:
      return misfit(eta_flat, forward_fun, data)
    
    # Value and grad.
    val, grad = jax.value_and_grad(objective)(jnp.zeros_like(eta_true_flat))
    print(f"Initial misfit: {val}")
    print(f"Gradient norm: {jnp.linalg.norm(grad)}")
    
    # Optimization.
    print("Starting optimization...")
    lbfgs = jaxopt.LBFGS(fun=objective, maxiter=5)
    res = lbfgs.run(jnp.zeros_like(eta_true_flat))
    
    print(f"Final misfit: {res.state.value}")
    print(f"Optimization finished with status {res.state}")
    
    # Assert that misfit decreased.
    self.assertLess(res.state.value, val)

  def test_misfit_gradient(self) -> None:
    """Verify the gradient of the misfit function using finite differences."""
    nxint, nyint = 20, 20
    npml = 5
    nx, ny = nxint + 2 * npml, nyint + 2 * npml
    h = 1.0 / (nxint - 1)
    omega, sigma_max = 2.0, 10.0
    
    solver = HelmholtzSolver(nx, ny, npml, h, omega, sigma_max, mode='stencil')
    inc = IncomingDirections(nx, ny, npml, h, omega, n_theta=4)
    
    # Random sampling points.
    theta_r = jnp.linspace(0, 2 * jnp.pi, 5)
    points_query = 0.4 * jnp.stack([jnp.cos(theta_r), jnp.sin(theta_r)], axis=1)
    x = (jnp.arange(nx) - npml - nxint//2) * h
    y = (jnp.arange(ny) - npml - nyint//2) * h
    projection_op = get_projection_op(x, y, points_query)
    forward_fun = create_forward_with_adjoint(solver, inc, projection_op)
    
    # Fake data.
    eta_true = jnp.zeros(nxint * nyint)
    data = forward_fun(eta_true)
    
    # Random model and perturbation.
    key = jax.random.PRNGKey(42)
    eta = jax.random.normal(key, (nxint * nyint,)) * 0.01
    v = jax.random.normal(key, (nxint * nyint,)) * 0.01
    
    def objective(e):
      return misfit(e, forward_fun, data)
    
    # Custom JAX gradient.
    grad_custom = jax.grad(objective)(eta)
    
    # Finite difference approximation.
    eps = 1e-6
    obj_0 = objective(eta)
    obj_eps = objective(eta + eps * v)
    
    expected_diff = jnp.dot(grad_custom, v)
    actual_diff = (obj_eps - obj_0) / eps
    
    print(f"Misfit Gradient Check:")
    print(f"  Expected diff (grad * v): {expected_diff}")
    print(f"  Actual diff (FD): {actual_diff}")
    
    self.assertAlmostEqual(float(expected_diff), float(actual_diff), places=4)

if __name__ == "__main__":
  unittest.main()
