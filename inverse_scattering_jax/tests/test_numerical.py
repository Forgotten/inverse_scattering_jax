import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from inverse_scattering_jax.src.helmholtz import HelmholtzSolver, HelmholtzOperator
from scipy.special import hankel1
import unittest

class TestNumericalValidation(unittest.TestCase):
  def test_convergence(self) -> None:
    """Verify 2nd order convergence of the finite difference scheme."""
    def get_solution(nx_int):
      ny_int = nx_int
      npml = 15 # Large PML to minimize boundary errors.
      nx, ny = nx_int + 2 * npml, ny_int + 2 * npml
      h = 1.0 / (nx_int - 1)
      omega = 5.0
      sigma_max = 40.0
      
      op = HelmholtzOperator(nx, ny, npml, h, omega, sigma_max, mode='matrix')
      solver = HelmholtzSolver(op)
      
      # Source at center.
      f = jnp.zeros((ny, nx))
      f = f.at[ny//2, nx//2].set(1.0 / h**2)
      m_ext = jnp.ones((ny, nx))
      
      u_vec, _ = solver.solve(f.flatten(), m_ext)
      u = u_vec.reshape((ny, nx))
      return u, h

    # Run for three resolutions.
    u1, h1 = get_solution(41)
    u2, h2 = get_solution(81)
    u3, h3 = get_solution(161)
    
    # Sample point at fixed physical distance from center (0.2, 0.0).
    dist = 0.2
    val1 = u1[41//2, 41//2 + int(dist/h1)]
    val2 = u2[81//2, 81//2 + int(dist/h2)]
    val3 = u3[161//2, 161//2 + int(dist/h3)]
    
    # Richardson extrapolation estimate of convergence order.
    # Error E(h) ~ C * h^p.
    # (u1 - u2) / (u2 - u3) ~ (h1^p - h2^p) / (h2^p - h3^p).
    # Since h2 = h1/2 and h3 = h2/2:
    # (u1 - u2) / (u2 - u3) ~ (h1^p - (h1/2)^p) / ((h1/2)^p - (h1/4)^p)
    # = (1 - 2^-p) / (2^-p - 4^-p) = (1 - 2^-p) / (2^-p * (1 - 2^-p)) = 2^p.
    
    ratio = jnp.abs(val1 - val2) / jnp.abs(val2 - val3)
    p = jnp.log2(ratio)
    print(f"Observed convergence order p: {p}")
    
    # We expect p ~ 2.
    self.assertGreater(float(p), 1.5)

  def test_analytical_greens(self) -> None:
    """Compare numerical solution against the analytical Green's function."""
    nxint, nyint = 100, 100
    npml = 20
    nx, ny = nxint + 2 * npml, nyint + 2 * npml
    h = 0.02
    omega = 10.0
    sigma_max = 30.0
    
    op = HelmholtzOperator(nx, ny, npml, h, omega, sigma_max, mode='stencil')
    solver = HelmholtzSolver(op)
    
    # Source at center.
    src_idx_x, src_idx_y = nx // 2, ny // 2
    f = jnp.zeros((ny, nx))
    f = f.at[src_idx_y, src_idx_x].set(1.0 / h**2)
    m_ext = jnp.ones((ny, nx))
    
    u_vec, _ = solver.solve(f.flatten(), m_ext)
    u = u_vec.reshape((ny, nx))
    
    # Analytical solution: G(r) = i/4 * H0(omega * r).
    # Meshgrid of distances from source.
    y_coords = (jnp.arange(ny) - src_idx_y) * h
    x_coords = (jnp.arange(nx) - src_idx_x) * h
    X, Y = jnp.meshgrid(x_coords, y_coords)
    R = jnp.sqrt(X**2 + Y**2)
    
    # Avoid singularity at R=0.
    mask = (R > 2*h) & (R < 0.4) # Stay away from source and PML.
    u_numerical = u[mask]
    u_analytical = (1j / 4.0) * hankel1(0, omega * R[mask])
    
    # Debug prints.
    print(f"Numerical (sample): {u_numerical[0:5]}")
    print(f"Analytical (sample): {u_analytical[0:5]}")
    
    rel_error = jnp.linalg.norm(u_numerical - u_analytical) / jnp.linalg.norm(u_analytical)
    print(f"Relative error against Green's function: {rel_error}")
    self.assertLess(float(rel_error), 0.05) # 5% error is reasonable for 2nd order on coarse grid.

  def test_pml_effectiveness(self) -> None:
    """Verify that waves are attenuated in the PML region."""
    nxint, nyint = 60, 60
    npml = 20
    nx, ny = nxint + 2 * npml, nyint + 2 * npml
    h = 0.05
    omega = 8.0
    sigma_max = 40.0
    
    op = HelmholtzOperator(nx, ny, npml, h, omega, sigma_max, mode='stencil')
    solver = HelmholtzSolver(op)
    
    # Source near one corner (but inside interior).
    f = jnp.zeros((ny, nx))
    f = f.at[npml + 5, npml + 5].set(1.0 / h**2)
    m_ext = jnp.ones((ny, nx))
    
    u_vec, _ = solver.solve(f.flatten(), m_ext)
    u = u_vec.reshape((ny, nx))
    
    # Magnitude at source-ish area.
    mag_near_source = jnp.abs(u[npml + 5, npml + 5])
    # Magnitude at the far edge (deep in PML).
    mag_at_edge = jnp.abs(u[-1, -1])
    
    reduction = mag_at_edge / mag_near_source
    print(f"PML reduction factor: {reduction}")
    self.assertLess(float(reduction), 1e-3)

if __name__ == "__main__":
  unittest.main()
