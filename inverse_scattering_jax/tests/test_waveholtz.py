"""Unit tests for the WaveHoltz fixed-point Helmholtz solver.

The two filter kernels target complementary regimes (this is the key physics):

* ``'exponential'`` extracts the full complex phasor ``e^{i w t}`` and is
  *contractive on damped modes*, so it is the method for the absorbing **PML**
  Helmholtz problem (validated against :class:`HelmholtzSolver`).
* ``'cosine'`` (the original AGR2020 filter) extracts only the in-phase part and
  is *contractive on real modes*, so it solves the real, energy-conserving
  (no-PML) problem (validated by the self-consistency residual ``||A v - f||``).
"""

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import unittest

from inverse_scattering_jax.src.helmholtz import (
  HelmholtzOperator, HelmholtzSolver, GMRESOptions, Domain
)
from inverse_scattering_jax.src.waveholtz import (
  WaveHoltzSolver, WaveHoltzOptions, filter_beta
)


def _make_op(nx_int=40, npml=15, h=0.05, omega=6.0, sigma_max=30.0, mode='stencil'):
  nx = ny = nx_int + 2 * npml
  domain = Domain(nx=nx, ny=ny, npml=npml, h=h)
  return HelmholtzOperator(
    domain=domain, omega=omega, sigma_max=sigma_max, mode=mode
  )


def _gaussian_pulse(op, width=0.10, cx=0.0, cy=0.0):
  x = (jnp.arange(op.nx) - op.nx / 2) * op.h
  y = (jnp.arange(op.ny) - op.ny / 2) * op.h
  X, Y = jnp.meshgrid(x, y, indexing='xy')
  return jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))


class TestFilterBeta(unittest.TestCase):
  """The analytic WaveHoltz transfer function ``beta(mu)``."""

  def test_resonance_is_unity(self) -> None:
    """beta(omega) == 1 for both kernels and several window lengths."""
    omega = 6.0
    for kind in ('cosine', 'exponential'):
      for n_per in (1, 2, 3):
        b = filter_beta(jnp.array(omega), omega, kind=kind, n_periods=n_per)
        print(f"beta(omega) kind={kind} n_per={n_per}: {complex(b):.6f}")
        self.assertAlmostEqual(float(jnp.abs(b - 1.0)), 0.0, places=8)

  def test_cosine_contractive_on_real_modes(self) -> None:
    """Cosine filter: |beta(mu)| < 1 for real off-resonance modes.

    This is the AGR2020 contraction property that makes the (Richardson)
    fixed-point iteration converge for the energy-conserving wave equation.
    """
    omega = 6.0
    mu = jnp.concatenate([
      jnp.linspace(0.1 * omega, 0.85 * omega, 40),
      jnp.linspace(1.15 * omega, 3.0 * omega, 40),
    ])
    b = filter_beta(mu, omega, kind='cosine', n_periods=1)
    mx = float(jnp.max(jnp.abs(b)))
    print(f"cosine max|beta| on real modes: {mx:.4f}")
    self.assertLess(mx, 1.0)

  def test_exponential_contractive_on_damped_modes(self) -> None:
    """Exponential filter: |beta(mu)| < 1 for damped (PML) modes.

    PML modes have ``mu = mu_r - i*gamma`` with ``gamma > 0``; the phasor filter
    contracts them (this is why exponential WaveHoltz converges with PML, while
    the cosine filter — which amplifies damped modes — does not).
    """
    omega = 6.0
    mu_r = jnp.linspace(0.1 * omega, 3.0 * omega, 60)
    for gamma in (0.2, 0.5, 1.0):
      mu = mu_r - 1j * gamma
      b = filter_beta(mu, omega, kind='exponential', n_periods=1)
      mx = float(jnp.max(jnp.abs(b)))
      print(f"exponential max|beta| (gamma={gamma}): {mx:.4f}")
      self.assertLess(mx, 1.0)


class TestWaveHoltzPML(unittest.TestCase):
  """Exponential-filter WaveHoltz reproduces the PML Helmholtz solution.

  The comparison is on the physical interior (the PML-layer auxiliary fields are
  formulation-dependent and excluded). The residual floor (~5-6%) is the
  time-domain ADE-PML vs frequency-domain PML realisation mismatch, the same
  floor measured by the wave-equation steady-state test.
  """

  @classmethod
  def setUpClass(cls):
    cls.op = _make_op()
    cls.npml = cls.op.npml
    cls.m_ext = jnp.ones((cls.op.ny, cls.op.nx))
    cls.f = _gaussian_pulse(cls.op, width=0.10).flatten().astype(jnp.complex128)
    solver = HelmholtzSolver(
      op=cls.op, gmres_options=GMRESOptions(tol=1e-9, maxiter=5000)
    )
    u_ref, _ = solver.solve(cls.f, cls.m_ext)
    cls.u_ref = u_ref.reshape((cls.op.ny, cls.op.nx))

  def _interior_relerr(self, v):
    sl = (slice(self.npml, -self.npml), slice(self.npml, -self.npml))
    a = v.reshape((self.op.ny, self.op.nx))[sl]
    b = self.u_ref[sl]
    rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    rel_conj = float(jnp.linalg.norm(jnp.conj(a) - b) / jnp.linalg.norm(b))
    return min(rel, rel_conj)

  def test_exponential_gmres(self) -> None:
    opts = WaveHoltzOptions(filter='exponential', solver='gmres',
                            n_periods=6, tol=1e-8, maxiter=400)
    solver = WaveHoltzSolver(op=self.op, options=opts)
    v, info = solver.solve(self.f, self.m_ext)
    err = self._interior_relerr(v)
    print(f"[exp/gmres] interior relerr={err:.4f} "
          f"res={float(info['residual']):.2e}")
    self.assertLess(err, 0.09)

  def test_exponential_fixed_point(self) -> None:
    """Richardson converges too: PML damping keeps the spectrum contractive."""
    opts = WaveHoltzOptions(filter='exponential', solver='fixed_point',
                            n_periods=6, tol=1e-7, maxiter=500)
    solver = WaveHoltzSolver(op=self.op, options=opts)
    v, info = solver.solve(self.f, self.m_ext)
    err = self._interior_relerr(v)
    print(f"[exp/fixed_point] interior relerr={err:.4f} "
          f"iters={int(info['iters'])} res={float(info['residual']):.2e}")
    self.assertLess(err, 0.09)
    self.assertLess(float(info['residual']), 1e-6)

  def test_fixed_point_matches_gmres(self) -> None:
    """Both outer solvers converge to the same fixed point of (I - S)."""
    base = dict(filter='exponential', n_periods=6, tol=1e-9)
    s_g = WaveHoltzSolver(op=self.op,
                          options=WaveHoltzOptions(solver='gmres',
                                                   maxiter=400, **base))
    s_f = WaveHoltzSolver(op=self.op,
                          options=WaveHoltzOptions(solver='fixed_point',
                                                   maxiter=2000, **base))
    vg, _ = s_g.solve(self.f, self.m_ext)
    vf, _ = s_f.solve(self.f, self.m_ext)
    rel = float(jnp.linalg.norm(vg - vf) / jnp.linalg.norm(vg))
    print(f"[gmres vs fixed_point] rel diff={rel:.2e}")
    self.assertLess(rel, 1e-4)


class TestWaveHoltzCosineReal(unittest.TestCase):
  """Cosine-filter WaveHoltz solves the real, no-PML Helmholtz problem.

  Validated by the self-consistency residual ``||A v - f|| / ||f||`` (no PML, so
  ``A`` is real and the solution is real). This is the original AGR2020 filter
  in its proper, energy-conserving regime.
  """

  def test_cosine_real_self_consistent(self) -> None:
    op = _make_op(sigma_max=0.0)  # no PML -> real operator, real solution
    m_ext = jnp.ones((op.ny, op.nx))
    f = _gaussian_pulse(op, width=0.10).flatten().astype(jnp.complex128)

    for solver, maxiter in (('gmres', 400), ('fixed_point', 3000)):
      opts = WaveHoltzOptions(filter='cosine', solver=solver, n_periods=4,
                              use_pml=False, tol=1e-9, maxiter=maxiter)
      s = WaveHoltzSolver(op=op, options=opts)
      v, info = s.solve(f, m_ext)
      resid = float(jnp.linalg.norm(op.operator(v, m_ext) - f)
                    / jnp.linalg.norm(f))
      imfrac = float(jnp.linalg.norm(v.imag) / jnp.linalg.norm(v))
      print(f"[cosine/{solver}] ||Av-f||/||f||={resid:.4f} imag_frac={imfrac:.1e}")
      self.assertLess(resid, 0.02)
      self.assertLess(imfrac, 1e-6)  # cosine recovers the real solution


class TestWaveHoltzBatch(unittest.TestCase):
  """vmap batching over multiple right-hand sides."""

  def test_batched_equals_looped(self) -> None:
    op = _make_op()
    m_ext = jnp.ones((op.ny, op.nx))
    opts = WaveHoltzOptions(filter='exponential', solver='gmres',
                            n_periods=4, tol=1e-8, maxiter=400)
    solver = WaveHoltzSolver(op=op, options=opts)

    cols = [
      _gaussian_pulse(op, width=0.10, cx=0.0, cy=0.0).flatten(),
      _gaussian_pulse(op, width=0.08, cx=0.3, cy=-0.2).flatten(),
      _gaussian_pulse(op, width=0.12, cx=-0.25, cy=0.15).flatten(),
    ]
    F = jnp.stack(cols, axis=1).astype(jnp.complex128)

    V = solver.solve_batched(F, m_ext)
    self.assertEqual(V.shape, F.shape)

    for j in range(F.shape[1]):
      vj, _ = solver.solve(F[:, j], m_ext)
      rel = float(jnp.linalg.norm(V[:, j] - vj) / jnp.linalg.norm(vj))
      print(f"[batch col {j}] rel diff vs single solve={rel:.2e}")
      self.assertLess(rel, 1e-6)


class TestWaveHoltzPMLAlignment(unittest.TestCase):
  """Stage 3: the WaveHoltz (I - S) system is the PML-aligned preconditioner.

  ``WaveHoltz`` does not act as a black-box inverse of the indefinite Helmholtz
  operator ``A``; instead the map ``I - S`` *is* the preconditioned system, with
  a spectrum clustered in ``{1 - beta}``.  Its PML is aligned with the
  frequency-domain operator *by construction*: the time-domain ADE-PML damping
  ``exp(-sigma*dt)`` and the frequency-domain stretch ``1 + i*sigma/omega`` read
  the *same* ``op.sx`` / ``op.sy`` arrays.  We check that sharing structurally,
  and that the alignment yields the right interior field across PML strengths.
  """

  def test_sigma_profiles_are_shared(self) -> None:
    """The wave solver derives its PML damping from the operator's own sigma."""
    op = _make_op(sigma_max=30.0)
    opts = WaveHoltzOptions(filter='exponential', solver='gmres', n_periods=4)
    solver = WaveHoltzSolver(op=op, options=opts)
    # Same operator instance -> identical sigma arrays (no second realisation).
    self.assertIs(solver.wave.op, op)
    dt = solver.wave.dt
    self.assertTrue(jnp.allclose(solver.wave._decay_x, jnp.exp(-op.sx * dt)))
    self.assertTrue(jnp.allclose(solver.wave._decay_y, jnp.exp(-op.sy * dt)))
    # The frequency-domain stretch consumes the identical sigma profile.
    self.assertTrue(jnp.allclose(op._denom_x, 1.0 + 1j / op.omega * op.sx))

  def test_alignment_across_pml_strength(self) -> None:
    """Interior solution tracks the PML Helmholtz solver for several sigma_max.

    If the PML were misaligned (wrong sigma, sign, or location) the interior
    would not match; that it matches for every absorption strength is the
    alignment check.
    """
    for sigma_max in (15.0, 30.0, 45.0):
      op = _make_op(sigma_max=sigma_max)
      m_ext = jnp.ones((op.ny, op.nx))
      f = _gaussian_pulse(op, width=0.10).flatten().astype(jnp.complex128)
      ref_solver = HelmholtzSolver(
        op=op, gmres_options=GMRESOptions(tol=1e-9, maxiter=5000)
      )
      u_ref, _ = ref_solver.solve(f, m_ext)
      u_ref = u_ref.reshape((op.ny, op.nx))

      opts = WaveHoltzOptions(filter='exponential', solver='gmres',
                              n_periods=6, tol=1e-8, maxiter=400)
      v, _ = WaveHoltzSolver(op=op, options=opts).solve(f, m_ext)

      sl = (slice(op.npml, -op.npml), slice(op.npml, -op.npml))
      a = v.reshape((op.ny, op.nx))[sl]
      b = u_ref[sl]
      rel = min(float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b)),
                float(jnp.linalg.norm(jnp.conj(a) - b) / jnp.linalg.norm(b)))
      print(f"[align sigma_max={sigma_max:g}] interior relerr={rel:.4f}")
      self.assertLess(rel, 0.09)


class TestWaveHoltzAdvertisedSteps(unittest.TestCase):
  """Stage 4: WaveHoltz solves Helmholtz in the advertised step count.

  The headline AGR2020 property is a *bounded, mesh/frequency-robust* iteration
  count, because ``I - S`` has a clustered spectrum.  We check (a) WaveHoltz
  needs far fewer Krylov iterations than unpreconditioned GMRES on ``A``, and
  (b) that count is essentially mesh-independent while accuracy improves under
  refinement.  These reuse the benchmark's ``measure`` to test the real harness.
  """

  def test_iteration_count_advantage(self) -> None:
    """WaveHoltz-GMRES converges in O(10) iters; direct GMRES needs many more."""
    from inverse_scattering_jax.benchmarks.benchmark_waveholtz import measure
    r = measure(8.0, ppw=10.0, box=2.0, do_fixed_point=True)
    print(f"[adv steps] direct={r.its_direct} wh_gmres={r.its_wh} "
          f"wh_fp={r.its_fp} res={r.res_wh:.1e}")
    self.assertEqual(r.info_wh, 0)               # GMRES on (I - S) converged
    self.assertLess(r.res_wh, 1e-5)
    self.assertLess(r.its_wh, 20)                # advertised: ~O(10) steps
    self.assertLess(r.its_fp, 25)                # Richardson is bounded too
    self.assertGreater(r.its_direct, 4 * r.its_wh)  # huge iteration-count gap

  def test_mesh_independence(self) -> None:
    """WaveHoltz iters ~ constant under refinement; direct grows; accuracy up."""
    from inverse_scattering_jax.benchmarks.benchmark_waveholtz import measure
    coarse = measure(8.0, ppw=9.0, box=2.0, do_fixed_point=False)
    fine = measure(8.0, ppw=15.0, box=2.0, do_fixed_point=False)
    print(f"[mesh indep] wh: {coarse.its_wh}->{fine.its_wh}  "
          f"direct: {coarse.its_direct}->{fine.its_direct}  "
          f"relerr: {coarse.relerr:.3f}->{fine.relerr:.3f}")
    self.assertLessEqual(fine.its_wh, coarse.its_wh + 3)   # mesh-independent
    self.assertGreater(fine.its_direct, coarse.its_direct)  # direct grows
    self.assertLess(fine.relerr, coarse.relerr)             # converges to A^{-1}f


if __name__ == "__main__":
  unittest.main()
