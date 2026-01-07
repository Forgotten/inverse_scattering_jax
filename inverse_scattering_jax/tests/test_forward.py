import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from inverse_scattering_jax.src.helmholtz import HelmholtzSolver, GMRESOptions
from inverse_scattering_jax.src.inverse_scattering import (
  IncomingDirections, 
  create_forward_with_adjoint, 
  get_projection_op
)
from absl.testing import parameterized
import unittest

class TestForwardProblem(parameterized.TestCase):
  def setUp(self) -> None:
    self.nxint, self.nyint = 20, 20
    self.npml = 5
    self.nx = self.nxint + 2 * self.npml
    self.ny = self.nyint + 2 * self.npml
    self.h = 1.0 / (self.nxint - 1)
    self.omega = 5.0
    self.sigma_max = 10.0
    self.order = 2
    
    self.n_theta = 4
    self.inc = IncomingDirections(
      self.nx, self.ny, self.npml, self.h, self.omega, self.n_theta
    )
    
    # Sampling points.
    theta_r = jnp.linspace(0, 2 * jnp.pi, 5)
    self.points_query = 0.4 * jnp.stack(
      [jnp.cos(theta_r), jnp.sin(theta_r)], axis=1
    )
    
    x = (jnp.arange(self.nx) - self.npml - self.nxint//2) * self.h
    y = (jnp.arange(self.ny) - self.npml - self.nyint//2) * self.h
    self.projection_op = get_projection_op(x, y, self.points_query)

  @parameterized.product(
    dtype=[jnp.complex128, jnp.complex64],
    mode=['matrix', 'stencil', 'conv']
  )
  def test_operator_adjoint(self, dtype, mode) -> None:
    """Check that <Au, v> = <u, A^H v> for different precisions."""
    key = jax.random.PRNGKey(42)
    tol = 1e-12 if dtype == jnp.complex128 else 1e-5
    
    u_vec = jax.random.normal(key, (self.nx * self.ny,), dtype=dtype)
    v_vec = jax.random.normal(key, (self.nx * self.ny,), dtype=dtype)
    m_ext = 1.0 + jax.random.normal(key, (self.ny, self.nx), 
                                   dtype=jnp.float64 if dtype==jnp.complex128 else jnp.float32) * 0.1
    
    solver = HelmholtzSolver(
      self.nx, self.ny, self.npml, self.h, self.omega, 
      self.sigma_max, self.order, mode=mode, dtype=dtype
    )
    Au = solver.operator(u_vec, m_ext)
    Adv = solver.hermitian_operator(v_vec, m_ext)
    
    inner1 = jnp.vdot(v_vec, Au)
    inner2 = jnp.vdot(Adv, u_vec)
    
    err = jnp.abs(inner1 - inner2) / jnp.abs(inner1)
    self.assertLess(float(err), tol)

  @parameterized.parameters('stencil', 'conv')
  def test_mode_consistency(self, mode) -> None:
    """Verify other modes result in the same operator output as 'matrix'."""
    key = jax.random.PRNGKey(123)
    u_vec = jax.random.normal(key, (self.nx * self.ny,))
    m_ext = 1.0 + jax.random.normal(key, (self.ny, self.nx)) * 0.1
    
    solver_ref = HelmholtzSolver(
      self.nx, self.ny, self.npml, self.h, self.omega, 
      self.sigma_max, self.order, mode='matrix'
    )
    res_ref = solver_ref.operator(u_vec, m_ext)
    
    solver = HelmholtzSolver(
      self.nx, self.ny, self.npml, self.h, self.omega, 
      self.sigma_max, self.order, mode=mode
    )
    res = solver.operator(u_vec, m_ext)
      
    diff = jnp.linalg.norm(res_ref - res) / jnp.linalg.norm(res_ref)
    self.assertLess(float(diff), 1e-6)

  def test_vjp(self) -> None:
    """Verify the custom VJP via finite differences."""
    dtype = jnp.complex128
    solver = HelmholtzSolver(
      self.nx, self.ny, self.npml, self.h, self.omega, 
      self.sigma_max, self.order, mode='stencil', dtype=dtype
    )
    forward_fun = create_forward_with_adjoint(solver, self.inc, self.projection_op)
    
    eta = jnp.zeros(self.nxint * self.nyint, dtype=jnp.float64)
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (self.nxint * self.nyint,), dtype=jnp.float64) * 0.01
    
    def objective(e):
      scattered = forward_fun(e)
      return 0.5 * jnp.sum(jnp.abs(scattered)**2)
    
    grad_custom = jax.grad(objective)(eta)
    eps = 1e-6
    obj_0 = objective(eta)
    obj_eps = objective(eta + eps * v)
    expected_diff = jnp.dot(grad_custom, v)
    actual_diff = (obj_eps - obj_0) / eps
    
    self.assertAlmostEqual(float(expected_diff), float(actual_diff), places=4)

  def test_forward_output_shape(self) -> None:
    solver = HelmholtzSolver(
      self.nx, self.ny, self.npml, self.h, self.omega, 
      self.sigma_max, self.order, mode='stencil'
    )
    forward_fun = create_forward_with_adjoint(solver, self.inc, self.projection_op)
    eta = jnp.zeros(self.nxint * self.nyint)
    scattered = forward_fun(eta)
    self.assertEqual(scattered.shape, (self.points_query.shape[0], self.n_theta))

if __name__ == "__main__":
  unittest.main()
