"""WaveHoltz fixed-point solver for the finite-difference Helmholtz problem.

WaveHoltz (Appelo, Garcia & Runborg, SISC 2020, ``arXiv:1910.10148``; and the
PML/Krylov extensions, e.g. ``arXiv:2205.12349``) solves the time-harmonic
problem by filtering a time-domain wave solve over one period ``T = 2*pi/omega``:

  Pi[v](x) = (2/T) * integral_0^T  K(t) * u(x, t) dt,

where ``u`` solves ``u_tt = G u + S`` with ``u(0)=v``, ``u_t(0)=0`` and
``G = A - omega^2`` is the spatial operator of the (PML) Helmholtz operator
``A`` realised in time by :class:`WaveOperator`. Two filter kernels are offered:

* ``'cosine'``    — the original ``K = cos(omega t) - 1/4`` (real),
* ``'exponential'`` — the phasor kernel ``K = exp(i omega t)`` (complex),

both normalised so the resonant response is unity, hence the fixed point solves
``A v = f`` exactly (when forced with ``-f cos(omega t)``). The map is affine,
``Pi[v] = S v + b`` with ``b = Pi[0]``, so the fixed point is the solution of
``(I - S) v = b``. We expose both a Richardson fixed-point iteration and a
Krylov (GMRES) acceleration, and batch over right-hand sides with ``vmap``.

Because the PML makes ``A`` complex, the time field is evolved in complex
arithmetic (``G`` has real coefficients, so real/imag parts decouple) and the
filter returns the complex phasor — reproducing the complex Helmholtz solution.
"""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy.sparse.linalg as sp_linalg

from .helmholtz import HelmholtzOperator, Array
from .wave_equation import WaveOperator, WaveStepOptions


@dataclass(kw_only=True)
class WaveHoltzOptions:
  """Options for the WaveHoltz solver.

  Attributes:
    filter: Time filter kernel, ``'cosine'`` (real, original AGR2020) or
      ``'exponential'`` (complex phasor). The exponential kernel extracts the
      full complex phasor and is the method for absorbing/PML problems; the
      cosine kernel extracts only the in-phase (real) component and targets the
      real, energy-conserving formulation (typically ``use_pml=False``).
    solver: Outer solve, ``'fixed_point'`` (Richardson) or ``'gmres'`` (Krylov).
      The exponential filter is not strictly contractive (``|beta|`` can slightly
      exceed 1 in a sidelobe near resonance), so Krylov acceleration is
      preferred; the cosine filter is provably contractive.
    n_periods: Number of periods of time integration per ``Pi`` application.
      Longer windows sharpen the filter and reject PML/transient leakage.
    cfl: CFL fraction passed to the underlying :class:`WaveOperator`.
    use_pml: Whether the underlying wave solve evolves the ADE-PML (absorbing).
      Use ``True`` (default) with the exponential filter for the PML Helmholtz
      problem; ``False`` for the real, energy-conserving cosine WaveHoltz.
    tol: Convergence tolerance (relative residual).
    maxiter: Maximum outer iterations.
  """
  filter: Literal['cosine', 'exponential'] = 'exponential'
  solver: Literal['fixed_point', 'gmres'] = 'gmres'
  n_periods: int = 1
  cfl: float = 0.4
  use_pml: bool = True
  tol: float = 1e-6
  maxiter: int = 200


def filter_beta(
  mu: Array, omega: float, kind: str = 'cosine', n_periods: int = 1
) -> Array:
  """Analytic WaveHoltz transfer function ``beta(mu)`` for one filter.

  ``beta`` is the eigenvalue of the homogeneous operator ``S`` acting on a mode
  whose free oscillation frequency is ``mu`` (``u_tt = -mu^2 u``). By design
  ``beta(omega) = 1`` (resonant mode preserved) and ``|beta(mu)| < 1`` away from
  resonance, which drives convergence.

  Args:
    mu: Free oscillation frequency/frequencies (may be complex for damped/PML
      modes).
    omega: Target angular frequency.
    kind: ``'cosine'`` or ``'exponential'``.
    n_periods: Number of periods in the filter window.

  Returns:
    ``beta(mu)`` evaluated elementwise.
  """
  T = n_periods * 2.0 * jnp.pi / omega
  mu = jnp.asarray(mu) + 0j

  def _int_cos_cos():
    # (2/T) int_0^T cos(omega t) cos(mu t) dt.
    return (2.0 / T) * 0.5 * (
      _sinc_int(omega + mu, T) + _sinc_int(omega - mu, T)
    )

  def _int_cos():
    # (2/T) int_0^T cos(mu t) dt.
    return (2.0 / T) * _sinc_int(mu, T)

  def _int_exp_cos():
    # (2/T) int_0^T exp(i omega t) cos(mu t) dt.
    return (2.0 / T) * 0.5 * (
      _expint(1j * (omega + mu), T) + _expint(1j * (omega - mu), T)
    )

  if kind == 'cosine':
    return _int_cos_cos() - 0.25 * _int_cos()
  elif kind == 'exponential':
    return _int_exp_cos()
  raise ValueError(f"unknown filter kind {kind!r}")


def _sinc_int(a: Array, T: float) -> Array:
  """int_0^T cos(a t) dt = sin(a T)/a, with the ``a->0`` limit handled."""
  a = jnp.asarray(a) + 0j
  small = jnp.abs(a) < 1e-12
  safe = jnp.where(small, 1.0, a)
  return jnp.where(small, T + 0j, jnp.sin(safe * T) / safe)


def _expint(b: Array, T: float) -> Array:
  """int_0^T exp(b t) dt = (exp(b T)-1)/b, with the ``b->0`` limit handled."""
  b = jnp.asarray(b) + 0j
  small = jnp.abs(b) < 1e-12
  safe = jnp.where(small, 1.0, b)
  return jnp.where(small, T + 0j, (jnp.exp(safe * T) - 1.0) / safe)


@dataclass(kw_only=True, eq=False)
class WaveHoltzSolver:
  """WaveHoltz solver composing a :class:`WaveOperator` and a filter.

  Attributes:
    op: The Helmholtz operator providing geometry, omega and PML.
    options: WaveHoltz options.
  """
  op: HelmholtzOperator
  options: WaveHoltzOptions = dataclasses.field(default_factory=WaveHoltzOptions)

  def __post_init__(self):
    o = self.options
    self.wave = WaveOperator(
      self.op, WaveStepOptions(cfl=o.cfl, use_pml=o.use_pml)
    )
    self.dt = self.wave.dt
    self.omega = self.op.omega
    self.T = 2.0 * jnp.pi / self.omega
    self.nsteps = int(round(o.n_periods * self.T / self.dt))

    # Filter kernel weights K(t_n) * dt * w_trap[n], n = 0..nsteps.
    n = jnp.arange(self.nsteps + 1)
    t = n * self.dt
    norm = 2.0 / (o.n_periods * self.T)
    if o.filter == 'cosine':
      K = norm * (jnp.cos(self.omega * t) - 0.25)
    else:
      K = norm * jnp.exp(1j * self.omega * t)
    w = jnp.ones(self.nsteps + 1).at[0].set(0.5).at[-1].set(0.5)
    self.kernel = (K * w * self.dt).astype(jnp.complex128)

    # Real cosine drive signal cos(omega t_n), n = 0..nsteps-1.
    self.cos_signal = jnp.cos(self.omega * jnp.arange(self.nsteps) * self.dt)

  # ---- the WaveHoltz map Pi -------------------------------------------
  def _apply_pi(
    self, v_flat: Array, m_ext: Array, forcing_flat: Optional[Array]
  ) -> Array:
    """Applies ``Pi`` to a flattened field.

    Args:
      v_flat: Initial data ``v`` (flattened, complex).
      m_ext: Extended model parameters.
      forcing_flat: If given, the spatial forcing profile (flattened); the wave
        equation is driven by ``forcing_flat * cos(omega t)``. If ``None`` the
        homogeneous map ``S v`` is computed.

    Returns:
      ``Pi[v]`` (flattened, complex).
    """
    ny, nx = self.op.ny, self.op.nx
    v2 = v_flat.reshape((ny, nx))
    if forcing_flat is None:
      sf, sig = None, None
    else:
      sf, sig = forcing_flat.reshape((ny, nx)), self.cos_signal
    filtered, _, _ = self.wave.propagate_filtered(
      v2, jnp.zeros_like(v2), m_ext, sf, sig, self.kernel, self.nsteps
    )
    return filtered.flatten()

  # ---- the WaveHoltz-preconditioned Helmholtz system ------------------
  def waveholtz_system(
    self, f_flat: Array, m_ext: Array
  ) -> Tuple[Callable[[Array], Array], Array]:
    """Builds the WaveHoltz reformulation ``(I - S) v = b`` of ``A v = f``.

    WaveHoltz *is* the preconditioner: it replaces the indefinite, strongly
    non-normal Helmholtz operator ``A`` by the operator ``I - S`` whose spectrum
    is the cluster ``{1 - beta(mu)}`` with ``|beta| < 1`` on the (PML-damped)
    spectrum. That clustering — bounded away from the origin and from infinity —
    is what makes the problem amenable to Krylov iteration, whereas Krylov on
    ``A`` directly stalls on its near-zero / indefinite spectrum.

    The PML is aligned *by construction*: ``S`` and ``b`` are produced by the
    same :class:`WaveOperator`/:class:`HelmholtzOperator` pair (same ``sigma``
    profiles), so the time-domain ADE-PML matches the frequency-domain stretch.

    Args:
      f_flat: Helmholtz RHS (flattened).
      m_ext: Extended model parameters.

    Returns:
      Tuple ``(matvec, b)`` where ``matvec(v) = v - S v`` and ``b = Pi_{-f}[0]``;
      the WaveHoltz solution solves ``matvec(v) = b``.
    """
    f = f_flat.astype(jnp.complex128)
    b = self._apply_pi(jnp.zeros_like(f), m_ext, -f)
    matvec = lambda v: v - self._apply_pi(v, m_ext, None)
    return matvec, b

  # ---- single-RHS solve -----------------------------------------------
  def solve(
    self, f_flat: Array, m_ext: Array
  ) -> Tuple[Array, dict]:
    """Solves ``A v = f`` for a single right-hand side.

    Args:
      f_flat: Helmholtz RHS (flattened).
      m_ext: Extended model parameters.

    Returns:
      Tuple ``(v_flat, info)`` where ``info`` holds iteration diagnostics.
    """
    A_op, b = self.waveholtz_system(f_flat, m_ext)
    S = lambda v: self._apply_pi(v, m_ext, None)

    if self.options.solver == 'gmres':
      v, _ = sp_linalg.gmres(
        A_op, b, tol=self.options.tol, atol=0.0,
        maxiter=self.options.maxiter, restart=min(self.nsteps, 40),
      )
      res = jnp.linalg.norm(A_op(v) - b) / jnp.linalg.norm(b)
      return v, {'residual': res}

    # Richardson fixed point: v <- S v + b, until ||v_new - v|| small.
    bnorm = jnp.linalg.norm(b)

    def cond(state):
      _, k, res = state
      return jnp.logical_and(k < self.options.maxiter, res > self.options.tol)

    def step(state):
      v, k, _ = state
      v_new = S(v) + b
      res = jnp.linalg.norm(v_new - v) / bnorm
      return v_new, k + 1, res

    v0 = b
    v, k, res = lax.while_loop(cond, step, (v0, 0, jnp.inf))
    return v, {'iters': k, 'residual': res}

  # ---- batched solve (vmap over RHS columns) --------------------------
  @partial(jax.jit, static_argnums=(0,))
  def solve_batched(self, F: Array, m_ext: Array) -> Array:
    """Solves ``A V = F`` for many RHS simultaneously via ``vmap``.

    Args:
      F: RHS matrix with shape ``(n_dof, n_rhs)`` (columns are RHS).
      m_ext: Extended model parameters.

    Returns:
      Solution matrix ``V`` with shape ``(n_dof, n_rhs)``.
    """
    solve_one = lambda f: self.solve(f, m_ext)[0]
    return jax.vmap(solve_one, in_axes=1, out_axes=1)(F)
