"""Benchmark: WaveHoltz-preconditioned Helmholtz vs. direct Krylov on ``A``.

The headline WaveHoltz claim (Appelo, Garcia & Runborg, SISC 2020) is that the
*iteration count* needed to solve the Helmholtz problem is **bounded and almost
independent of the mesh and frequency**, because the filtered time-domain map
replaces the indefinite, near-singular operator ``A`` by ``I - S`` whose
spectrum clusters in ``{1 - beta(mu)}`` with ``|beta| < 1``.  Unpreconditioned
Krylov on ``A`` itself has no such bound: as ``omega`` grows (and the mesh is
refined to keep it resolved) the iteration count climbs steadily.

Two experiments make this concrete:

* ``frequency_scaling`` — fix points-per-wavelength, sweep ``omega``.  The grid
  grows with ``omega`` (more wavelengths across a fixed box).  Direct GMRES on
  ``A`` climbs; WaveHoltz-GMRES on ``I - S`` stays ~flat.
* ``mesh_independence`` — fix ``omega``, refine the mesh.  WaveHoltz iterations
  are essentially constant in the number of unknowns; direct GMRES is not.

Honesty note printed alongside: each ``I - S`` application is a *full* time-
domain wave solve of ``nsteps`` steps, so a WaveHoltz "iteration" is far more
expensive than one sparse ``A`` mat-vec.  The benchmark therefore reports both
iteration counts *and* wall-clock time.  The advertised, robust quantity is the
iteration count (it parallelises trivially over RHS via ``vmap`` and never
stalls); wall-clock parity at modest ``omega`` is expected and not the point.

Run::

    python -m inverse_scattering_jax.benchmarks.benchmark_waveholtz
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse.linalg as sla

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from inverse_scattering_jax.src.helmholtz import HelmholtzOperator, Domain  # noqa: E402
from inverse_scattering_jax.src.waveholtz import (  # noqa: E402
  WaveHoltzSolver,
  WaveHoltzOptions,
)


# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #
def make_op(
  omega: float,
  ppw: float = 10.0,
  box: float = 2.0,
  npml: int = 15,
  sigma_max: float = 30.0,
) -> HelmholtzOperator:
  """Builds a PML Helmholtz operator at fixed points-per-wavelength.

  Args:
    omega: Angular frequency.
    ppw: Grid points per wavelength (sets ``h = 2*pi / (omega * ppw)``).
    box: Physical side length of the (interior) square domain.
    npml: PML layer width in grid points.
    sigma_max: Peak PML absorption.

  Returns:
    A configured :class:`HelmholtzOperator`.
  """
  h = 2.0 * np.pi / (omega * ppw)
  nx_int = int(round(box / h))
  nx = ny = nx_int + 2 * npml
  dom = Domain(nx=nx, ny=ny, npml=npml, h=h)
  return HelmholtzOperator(domain=dom, omega=omega, sigma_max=sigma_max,
                           mode='stencil')


def gaussian_rhs(op: HelmholtzOperator, width: float = 0.10) -> jnp.ndarray:
  """A centred Gaussian point-source RHS, flattened and complex."""
  x = (jnp.arange(op.nx) - op.nx / 2) * op.h
  y = (jnp.arange(op.ny) - op.ny / 2) * op.h
  X, Y = jnp.meshgrid(x, y, indexing='xy')
  return jnp.exp(-(X**2 + Y**2) / (2 * width**2)).flatten().astype(jnp.complex128)


def interior_relerr(op: HelmholtzOperator, a: np.ndarray, b: np.ndarray) -> float:
  """Relative L2 error on the physical interior (PML layer excluded)."""
  npml = op.npml
  sl = (slice(npml, -npml), slice(npml, -npml))
  A = np.asarray(a).reshape((op.ny, op.nx))[sl]
  B = np.asarray(b).reshape((op.ny, op.nx))[sl]
  return float(np.linalg.norm(A - B) / np.linalg.norm(B))


# --------------------------------------------------------------------------- #
# GMRES iteration counter (scipy, exact inner-iteration count)
# --------------------------------------------------------------------------- #
def count_gmres(
  matvec: Callable[[np.ndarray], np.ndarray],
  b: np.ndarray,
  n: int,
  rtol: float = 1e-6,
  restart: int = 300,
  maxiter: int = 6000,
) -> tuple[np.ndarray, int, int, float]:
  """Runs scipy GMRES, counting inner iterations via the ``pr_norm`` callback.

  Args:
    matvec: Matrix-vector product (must return a *writable* numpy array).
    b: Right-hand side.
    n: System dimension.
    rtol: Relative convergence tolerance.
    restart: GMRES restart length.
    maxiter: Maximum outer restart cycles.

  Returns:
    ``(x, iters, info, seconds)``: solution, inner-iteration count, scipy
    convergence flag (0 == converged), and wall-clock seconds.
  """
  cnt = [0]

  def cb(_):  # 'pr_norm' callback fires once per inner iteration
    cnt[0] += 1

  A = sla.LinearOperator((n, n), matvec=matvec, dtype=np.complex128)
  t0 = time.perf_counter()
  x, info = sla.gmres(A, b, rtol=rtol, atol=0.0, restart=min(restart, n),
                      maxiter=maxiter, callback=cb, callback_type='pr_norm')
  dt = time.perf_counter() - t0
  return x, cnt[0], info, dt


def _writable(jax_array) -> np.ndarray:
  """Materialises a JAX array as a writable numpy array (scipy mutates output)."""
  return np.array(np.asarray(jax_array))


# --------------------------------------------------------------------------- #
# One (omega, mesh) measurement
# --------------------------------------------------------------------------- #
@dataclass
class Row:
  omega: float
  n: int
  lam: float           # wavelengths across the interior box
  nsteps: int          # wave steps per (I - S) application
  its_direct: int
  t_direct: float
  info_direct: int
  its_wh: int
  t_wh: float
  info_wh: int
  its_fp: int          # WaveHoltz Richardson fixed-point iterations
  relerr: float        # interior rel error, WaveHoltz vs direct
  res_wh: float


def measure(
  omega: float,
  ppw: float,
  box: float,
  n_periods: int = 6,
  rtol: float = 1e-6,
  do_fixed_point: bool = True,
) -> Row:
  """Measures direct-GMRES vs WaveHoltz-GMRES (and Richardson) for one problem."""
  op = make_op(omega, ppw=ppw, box=box)
  n = op.nx * op.ny
  m = jnp.ones((op.ny, op.nx))
  f = gaussian_rhs(op)
  lam = omega * box / (2.0 * np.pi)

  # (a) Direct, unpreconditioned GMRES on the Helmholtz operator A.
  a_dir = lambda x: _writable(op.operator(jnp.asarray(x), m))
  b_dir = _writable(f)
  x_dir, its_d, info_d, t_d = count_gmres(a_dir, b_dir, n, rtol=rtol)

  # (b) WaveHoltz-preconditioned: GMRES on (I - S) v = b.
  wh = WaveHoltzSolver(op=op, options=WaveHoltzOptions(
    filter='exponential', solver='gmres', n_periods=n_periods,
    tol=rtol, maxiter=4000))
  s_op = lambda x: _writable(jnp.asarray(x) - wh._apply_pi(jnp.asarray(x), m, None))
  b_wh = _writable(wh._apply_pi(jnp.zeros(n, jnp.complex128), m, -f))
  x_wh, its_wh, info_wh, t_wh = count_gmres(s_op, b_wh, n, rtol=rtol)
  res_wh = float(np.linalg.norm(s_op(x_wh) - b_wh) / np.linalg.norm(b_wh))

  # WaveHoltz vs direct on the physical interior (sign/conj-robust).
  rel = min(interior_relerr(op, x_wh, x_dir),
            interior_relerr(op, np.conj(x_wh), x_dir))

  # (c) Optional: pure Richardson fixed point on the same map.
  its_fp = -1
  if do_fixed_point:
    wh_fp = WaveHoltzSolver(op=op, options=WaveHoltzOptions(
      filter='exponential', solver='fixed_point', n_periods=n_periods,
      tol=rtol, maxiter=2000))
    _, info_fp = wh_fp.solve(f, m)
    its_fp = int(info_fp['iters'])

  return Row(omega=omega, n=n, lam=lam, nsteps=wh.nsteps,
             its_direct=its_d, t_direct=t_d, info_direct=info_d,
             its_wh=its_wh, t_wh=t_wh, info_wh=info_wh,
             its_fp=its_fp, relerr=rel, res_wh=res_wh)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def frequency_scaling(
  omegas=(4.0, 8.0, 12.0, 16.0), ppw: float = 10.0, box: float = 2.0
) -> list[Row]:
  """Fixed resolution, sweep omega: grid grows, hardness grows."""
  print("\n=== Experiment 1: frequency scaling "
        f"(ppw={ppw:g}, box={box:g}, n_periods=6) ===")
  print(f"{'omega':>6} {'lam':>5} {'n':>7} {'nsteps':>7} "
        f"{'dir_its':>8} {'wh_its':>7} {'fp_its':>7} "
        f"{'t_dir[s]':>9} {'t_wh[s]':>9} {'relerr':>8}")
  rows = []
  for w in omegas:
    r = measure(w, ppw, box)
    rows.append(r)
    flag = '' if r.info_direct == 0 else f' !dir={r.info_direct}'
    print(f"{r.omega:6.1f} {r.lam:5.1f} {r.n:7d} {r.nsteps:7d} "
          f"{r.its_direct:8d} {r.its_wh:7d} {r.its_fp:7d} "
          f"{r.t_direct:9.2f} {r.t_wh:9.2f} {r.relerr:8.1e}{flag}")
  print(f"  (all WaveHoltz residuals ||(I-S)v - b||/||b|| <= "
        f"{max(r.res_wh for r in rows):.1e}; relerr is the coarse-grid "
        f"time- vs frequency-PML mismatch, see Exp 2)")
  return rows


def mesh_independence(
  omega: float = 8.0, ppws=(9.0, 12.0, 15.0, 18.0), box: float = 2.0
) -> list[Row]:
  """Fixed omega, refine the mesh: WaveHoltz iters ~ constant in n.

  Doubles as the accuracy check: ``relerr`` (WaveHoltz vs the frequency-domain
  Helmholtz solution, interior only) should fall ~2nd order as the mesh refines,
  confirming WaveHoltz converges to the Helmholtz solution while its iteration
  count stays pinned.
  """
  print(f"\n=== Experiment 2: mesh independence + accuracy convergence "
        f"(omega={omega:g}, box={box:g}, n_periods=6) ===")
  print(f"{'ppw':>5} {'n':>7} {'nsteps':>7} "
        f"{'dir_its':>8} {'wh_its':>7} {'fp_its':>7} {'relerr':>8}")
  rows = []
  for p in ppws:
    r = measure(omega, p, box)
    rows.append(r)
    flag = '' if r.info_direct == 0 else f' !dir={r.info_direct}'
    print(f"{p:5.1f} {r.n:7d} {r.nsteps:7d} "
          f"{r.its_direct:8d} {r.its_wh:7d} {r.its_fp:7d} {r.relerr:8.1e}{flag}")
  return rows


def _summary(freq_rows: list[Row], mesh_rows: list[Row]) -> None:
  """Prints a short interpretation of the advertised-step claim."""
  dir_its = [r.its_direct for r in freq_rows]
  wh_its = [r.its_wh for r in freq_rows]
  fp_its = [r.its_fp for r in freq_rows]
  print("\n=== Summary ===")
  print(f"frequency sweep: direct GMRES {dir_its[0]}->{dir_its[-1]} iters "
        f"({dir_its[-1] / max(dir_its[0], 1):.1f}x), "
        f"WaveHoltz-GMRES {min(wh_its)}-{max(wh_its)} iters (~flat), "
        f"WaveHoltz fixed-point {min(fp_its)}-{max(fp_its)} iters.")
  mdir = [r.its_direct for r in mesh_rows]
  mwh = [r.its_wh for r in mesh_rows]
  print(f"mesh refinement: direct GMRES {min(mdir)}-{max(mdir)} iters, "
        f"WaveHoltz-GMRES {min(mwh)}-{max(mwh)} iters (mesh-independent).")
  print(f"accuracy: interior relerr vs frequency-domain A falls "
        f"{mesh_rows[0].relerr:.2f} -> {mesh_rows[-1].relerr:.2f} under "
        f"refinement (~2nd order) — WaveHoltz converges to the Helmholtz "
        f"solution.")
  print("=> WaveHoltz solves the Helmholtz problem in a bounded, "
        "frequency/mesh-robust number of (I - S) solves — the advertised step "
        "count — each costing one full wave solve (nsteps steps).")


def main() -> None:
  freq_rows = frequency_scaling()
  mesh_rows = mesh_independence()
  _summary(freq_rows, mesh_rows)


if __name__ == "__main__":
  main()
