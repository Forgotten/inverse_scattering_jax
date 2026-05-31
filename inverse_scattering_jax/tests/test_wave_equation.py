import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import unittest

from inverse_scattering_jax.src.helmholtz import (
  HelmholtzOperator, HelmholtzSolver, GMRESOptions, Domain
)
from inverse_scattering_jax.src.wave_equation import WaveOperator, WaveStepOptions


def _make_grid(nx_int=80, npml=5, h=0.025, omega=6.0, sigma_max=0.0, mode='stencil'):
  nx = ny = nx_int + 2 * npml
  domain = Domain(nx=nx, ny=ny, npml=npml, h=h)
  op = HelmholtzOperator(
    domain=domain, omega=omega, sigma_max=sigma_max, mode=mode
  )
  return op


def _gaussian_pulse(op, width=0.08):
  x = (jnp.arange(op.nx) - op.nx / 2) * op.h
  y = (jnp.arange(op.ny) - op.ny / 2) * op.h
  X, Y = jnp.meshgrid(x, y, indexing='xy')
  return jnp.exp(-(X ** 2 + Y ** 2) / (2 * width ** 2))


class TestWaveCore(unittest.TestCase):
  def test_time_convergence_second_order(self) -> None:
    """Leapfrog should be 2nd order in dt (self-convergence, fixed grid)."""
    op = _make_grid()
    m_ext = jnp.ones((op.ny, op.nx))
    u0 = _gaussian_pulse(op)
    v0 = jnp.zeros_like(u0)

    # dt is linear in cfl, so halving cfl exactly halves dt. Doubling nsteps
    # in lockstep keeps the final time T = N * dt0 identical across runs, so the
    # only varying error is the temporal truncation (isolates 2nd-order in dt).
    wave0 = WaveOperator(op, WaveStepOptions(cfl=0.4, use_pml=False))
    N = int(round(0.25 / wave0.dt))  # short enough to avoid the boundary

    def final_state(cfl, nsteps):
      wave = WaveOperator(op, WaveStepOptions(cfl=cfl, use_pml=False))
      uf, _, _ = wave.propagate(u0, v0, m_ext, None, None, nsteps)
      return uf

    u_a = final_state(0.4, N)
    u_b = final_state(0.2, 2 * N)
    u_c = final_state(0.1, 4 * N)

    e1 = jnp.linalg.norm(u_a - u_b)
    e2 = jnp.linalg.norm(u_b - u_c)
    ratio = float(e1 / e2)
    order = jnp.log2(ratio)
    print(f"Self-convergence ratio {ratio:.3f} -> temporal order {float(order):.3f}")
    self.assertGreater(float(order), 1.8)

  def test_energy_conservation(self) -> None:
    """Undamped (no-PML) wave equation should conserve energy to high accuracy."""
    op = _make_grid()
    m_ext = jnp.ones((op.ny, op.nx))
    u0 = _gaussian_pulse(op)
    v0 = jnp.zeros_like(u0)

    wave = WaveOperator(op, WaveStepOptions(cfl=0.4, use_pml=False))
    nsteps = int(round(0.3 / wave.dt))
    _, _, traj = wave.propagate(u0, v0, m_ext, None, None, nsteps)

    dt = wave.dt
    # Discrete leapfrog energy: 0.5||(u^{n+1}-u^n)/dt||^2 - 0.5 <u^{n+1}, L u^n>.
    def energy(n):
      un = traj[n]
      unp = traj[n + 1]
      v = (unp - un) / dt
      Lun = wave._laplacian_real(un)
      return 0.5 * jnp.sum(v ** 2) - 0.5 * jnp.sum(unp * Lun)

    energies = jnp.array([energy(n) for n in range(0, nsteps, max(1, nsteps // 40))])
    drift = float(jnp.max(jnp.abs(energies - energies[0])) / jnp.abs(energies[0]))
    print(f"Max relative energy drift: {drift:.2e}")
    self.assertLess(drift, 1e-3)


def _extract_phasor(traj, dt, omega, n_per):
  """1-bin DFT over the last period: u_hat = (2/T) int_{last T} u e^{i w t} dt."""
  nsteps = traj.shape[0] - 1
  idx = jnp.arange(nsteps - n_per, nsteps + 1)  # n_per intervals, inclusive
  tt = dt * idx
  w = jnp.exp(1j * omega * tt)
  u_seg = traj[idx]
  weights = jnp.ones(n_per + 1).at[0].set(0.5).at[-1].set(0.5)  # trapezoid
  integ = jnp.sum(
    (u_seg * w[:, None, None]) * (dt * weights)[:, None, None], axis=0
  )
  return (2.0 / (dt * n_per)) * integ


class TestWavePML(unittest.TestCase):
  def test_pml_reflection_below_tolerance(self) -> None:
    """PML reflection coefficient must be small.

    We compare the small (PML) domain's interior against a *large* reference
    domain whose boundary is too far for reflections to have returned by the
    comparison time. Subtracting the two cancels the physical 2D wake and
    isolates the reflected field; we normalise by the incident peak amplitude.
    """
    h, omega, sigma_max = 0.05, 6.0, 40.0
    width = 0.18  # band-limited (~3.6 cells) so the pulse is well resolved
    # Small domain: interior half-width 1.0. Large: half-width 3.0.
    op_s = _make_grid(nx_int=40, npml=15, h=h, omega=omega, sigma_max=sigma_max)
    op_b = _make_grid(nx_int=120, npml=15, h=h, omega=omega, sigma_max=sigma_max)
    ms = jnp.ones((op_s.ny, op_s.nx))
    mb = jnp.ones((op_b.ny, op_b.nx))
    u0s = _gaussian_pulse(op_s, width=width)
    u0b = _gaussian_pulse(op_b, width=width)
    incident_peak = float(jnp.max(jnp.abs(u0s)))

    ws = WaveOperator(op_s, WaveStepOptions(cfl=0.4, use_pml=True))
    wb = WaveOperator(op_b, WaveStepOptions(cfl=0.4, use_pml=True))
    Tc = 2.5  # main pulse has entered the PML; big-domain interior still free
    ufs, _, _ = ws.propagate(u0s, jnp.zeros_like(u0s), ms, None, None,
                             int(round(Tc / ws.dt)))
    ufb, _, _ = wb.propagate(u0b, jnp.zeros_like(u0b), mb, None, None,
                             int(round(Tc / wb.dt)))

    # Physical interior [-1,1]^2: small indices [15:55], big indices [55:95].
    a = ufs[15:55, 15:55]
    b = ufb[55:95, 55:95]
    refl = float(jnp.max(jnp.abs(a - b)) / incident_peak)
    print(f"PML reflection coefficient (max|reflected|/incident): {refl:.2e}")
    self.assertLess(refl, 2e-3)

  def test_harmonic_steady_state_matches_helmholtz(self) -> None:
    """The harmonic steady state must reproduce HelmholtzSolver.solve (PML aligned)."""
    op = _make_grid(nx_int=40, npml=15, h=0.05, omega=6.0, sigma_max=30.0)
    solver = HelmholtzSolver(op=op, gmres_options=GMRESOptions(tol=1e-7, maxiter=4000))
    m_ext = jnp.ones((op.ny, op.nx))

    f = _gaussian_pulse(op, width=0.10)
    u_ref, _ = solver.solve(f.flatten(), m_ext)
    u_ref = u_ref.reshape((op.ny, op.nx))

    wave = WaveOperator(op, WaveStepOptions(cfl=0.3, use_pml=True))
    dt, omega = wave.dt, op.omega
    T = 2 * jnp.pi / omega
    n_per = int(round(T / dt))
    nsteps = n_per * 30  # 30 periods to settle the steady state

    t = dt * jnp.arange(nsteps)
    # Drive S = -f cos(omega t) so the steady state satisfies A u_hat = f.
    signal = -jnp.cos(omega * t)
    zero = jnp.zeros((op.ny, op.nx))
    _, _, traj = wave.propagate(zero, zero, m_ext, f, signal, nsteps)

    u_hat = _extract_phasor(traj, dt, omega, n_per)

    npml = op.npml
    sl = (slice(npml, -npml), slice(npml, -npml))
    a, b = u_hat[sl], u_ref[sl]
    # Robust to the e^{+/-i w t} convention (u_hat vs its conjugate).
    rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    rel_conj = float(jnp.linalg.norm(jnp.conj(a) - b) / jnp.linalg.norm(b))
    amp = float(jnp.linalg.norm(a) / jnp.linalg.norm(b))
    print(f"steady-state rel={rel:.3f}, rel_conj={rel_conj:.3f}, |a|/|b|={amp:.3f}")
    self.assertLess(min(rel, rel_conj), 0.12)


if __name__ == "__main__":
  unittest.main()
