import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from inverse_scattering_jax.src.helmholtz import HelmholtzSolver, HelmholtzOperator, GMRESOptions
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
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, n_theta=self.n_theta
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
    
    op = HelmholtzOperator(
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, 
      sigma_max=self.sigma_max, order=self.order, mode=mode, dtype=dtype
    )
    # Using solver interface just for operator access if needed, but testing operator directly
    Au = op.operator(u_vec, m_ext)
    Adv = op.operator_adjoint(v_vec, m_ext)
    
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
    
    op_ref = HelmholtzOperator(
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, 
      sigma_max=self.sigma_max, order=self.order, mode='matrix'
    )
    res_ref = op_ref.operator(u_vec, m_ext)
    
    op = HelmholtzOperator(
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, 
      sigma_max=self.sigma_max, order=self.order, mode=mode
    )
    res = op.operator(u_vec, m_ext)
      
    diff = jnp.linalg.norm(res_ref - res) / jnp.linalg.norm(res_ref)
    self.assertLess(float(diff), 1e-6)

  def test_vjp(self) -> None:
    """Verify the custom VJP via finite differences."""
    dtype = jnp.complex128
    op = HelmholtzOperator(
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, 
      sigma_max=self.sigma_max, order=self.order, mode='stencil', dtype=dtype
    )
    # Solver needs operator and options
    solver = HelmholtzSolver(op=op, gmres_options=GMRESOptions())
    
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
    op = HelmholtzOperator(
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, 
      sigma_max=self.sigma_max, order=self.order, mode='stencil'
    )
    solver = HelmholtzSolver(op=op)
    
    forward_fun = create_forward_with_adjoint(solver, self.inc, self.projection_op)
    eta = jnp.zeros(self.nxint * self.nyint)
    scattered = forward_fun(eta)
    self.assertEqual(scattered.shape, (self.points_query.shape[0], self.n_theta))

  def test_jacobian_adjoint(self) -> None:
    """Check that <Jv, w> = <v, J^H w> for the forward map."""
    dtype = jnp.complex128
    mode = 'stencil'
    op = HelmholtzOperator(
      nx=self.nx, ny=self.ny, npml=self.npml, h=self.h, omega=self.omega, 
      sigma_max=self.sigma_max, order=self.order, mode=mode, dtype=dtype
    )
    # Use tight tolerance for adjoint test to avoid solver noise
    solver = HelmholtzSolver(op=op, gmres_options=GMRESOptions(tol=1e-9, maxiter=2000))
    
    # Map from eta -> sampled scattered field
    forward_fun = create_forward_with_adjoint(solver, self.inc, self.projection_op)
    
    # Background perturbation - Try zero first to see if it matches test_vjp conditions
    key = jax.random.PRNGKey(42)
    eta_shape = (self.nxint * self.nyint,)
    eta_0 = jnp.zeros(eta_shape, dtype=jnp.float64)
    
    # Perturbation direction v (model space)
    k1, k2 = jax.random.split(key, 2)
    v = jax.random.normal(k1, eta_shape, dtype=jnp.float64)
    
    # Random vector w (data space)
    data_shape = (self.points_query.shape[0], self.n_theta)
    w = jax.random.normal(k2, data_shape, dtype=dtype) + \
        1j * jax.random.normal(k2, data_shape, dtype=dtype)
    
    # Linearization (Jv) using Finite Differences
    epsilon = 1e-4
    f_plus = forward_fun(eta_0 + epsilon * v)
    f_minus = forward_fun(eta_0 - epsilon * v)
    Jv = (f_plus - f_minus) / (2 * epsilon)
    
    # Adjoint (J^H w) using VJP
    _, vjp_fun = jax.vjp(forward_fun, eta_0)
    
    g = vjp_fun(w)[0]
    
    # LHS = Re(w^T Jv) which is the correct JAX VJP identity
    lhs_val = jnp.real(jnp.sum(Jv * w))
    rhs_val = jnp.vdot(v, g)
    
    print(f"LHS: {lhs_val}, RHS: {rhs_val}")
    
    # Check that they have the same sign
    self.assertTrue(lhs_val * rhs_val > 0, "Gradients should have the same sign")
    
    # Verify magnitude matches closely (within solver tolerance)
    diff = jnp.abs(lhs_val - rhs_val)
    mean = 0.5 * (jnp.abs(lhs_val) + jnp.abs(rhs_val))
    self.assertLess(float(diff / mean), 1e-4)
    
  def flatten_complex(self, arr):
    return arr.flatten()

if __name__ == "__main__":
  unittest.main()
