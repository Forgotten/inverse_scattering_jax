"""Time-domain scalar wave propagation compatible with the frequency-domain
finite-difference Helmholtz operator.

The spatial discretisation is *reused* from :class:`HelmholtzOperator`: the same
``apply_Dxx``/``apply_Dyy`` stencils (and, for the PML, the same ``sx``/``sy``/
``sxp``/``syp`` profiles) drive a second-order-in-time leapfrog integrator. This
guarantees that the harmonic steady state of the time-domain solver reproduces
the discrete Helmholtz symbol exactly, which is the property WaveHoltz relies on.

Two regimes are supported:

* ``use_pml=False`` — undamped scalar wave equation ``u_tt = L u`` with
  ``L = Dxx + Dyy``. Used for the energy-conservation and time-convergence gates.
* ``use_pml=True`` — unsplit ADE-PML whose time-harmonic reduction equals the
  Helmholtz operator's ``s^{-1} d_x (s^{-1} d_x u)`` stretching (Stage 1b).
"""

import dataclasses
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple, Literal, Any

import jax
import jax.numpy as jnp
from jax import lax

from .helmholtz import HelmholtzOperator, Array


@dataclass(kw_only=True)
class WaveStepOptions:
  """Options controlling the time integrator.

  Attributes:
    cfl: Fraction of the stability limit used to pick the time step.
    scheme: Time-stepping scheme. Currently ``'leapfrog'`` (2nd order).
    use_pml: If True, evolve the unsplit ADE-PML auxiliary variables so the
      harmonic steady state matches the Helmholtz operator's PML stretching.
  """
  cfl: float = 0.5
  scheme: Literal['leapfrog'] = 'leapfrog'
  use_pml: bool = True


def _estimate_spectral_radius(
  apply: Callable[[Array], Array], shape: Tuple[int, int], iters: int = 50
) -> float:
  """Power-iteration estimate of the spectral radius of ``-apply``.

  ``apply`` is the (negative semi-definite) spatial operator ``L``; we want the
  largest magnitude eigenvalue of ``-L`` to size the leapfrog time step.

  Args:
    apply: Function mapping a 2D field to ``L`` applied to it.
    shape: Field shape ``(ny, nx)``.
    iters: Number of power iterations.

  Returns:
    Estimated spectral radius (a Python float).
  """
  key = jax.random.PRNGKey(0)
  v = jax.random.normal(key, shape)
  v = v / jnp.linalg.norm(v)
  lam = 0.0
  for _ in range(iters):
    w = -apply(v)
    lam = jnp.vdot(v, w).real
    nrm = jnp.linalg.norm(w)
    v = w / nrm
  return float(jnp.abs(lam))


class WaveOperator:
  """Leapfrog time-domain wave propagator sharing the Helmholtz stencils.

  Attributes:
    op: The underlying :class:`HelmholtzOperator` (provides stencils + PML).
    options: Time-stepping options.
    dt: Chosen time step (float).
  """

  def __init__(
    self,
    op: HelmholtzOperator,
    options: Optional[WaveStepOptions] = None,
  ):
    """Initialises the wave operator and selects a stable time step.

    Args:
      op: Helmholtz operator to borrow the discretisation from.
      options: Time-stepping options (defaults to :class:`WaveStepOptions`).
    """
    self.op = op
    self.options = options if options is not None else WaveStepOptions()

    # Size the time step from the *undamped* spatial operator spectral radius:
    #   leapfrog stability requires dt <= 2 / sqrt(rho(-L_max)) where L_max is
    #   the full spatial operator including the omega^2 * m mass shift. We size
    #   on the Laplacian (dominant at high resolution) plus the mass shift.
    rho = _estimate_spectral_radius(self._laplacian_real, (op.ny, op.nx))
    rho_total = rho + (op.omega ** 2)  # conservative mass-term contribution
    self.dt = float(self.options.cfl * 2.0 / jnp.sqrt(rho_total))

    # Per-step exponential relaxation factors for the ADE-PML memory variables,
    # exp(-sigma * dt), using the operator's own PML sigma profiles (op.sx/op.sy
    # are the same sigma arrays distrib_pml feeds the frequency-domain operator).
    self._decay_x = jnp.exp(-op.sx * self.dt)
    self._decay_y = jnp.exp(-op.sy * self.dt)

  # ---- spatial operators (real, no PML) --------------------------------
  def _laplacian_real(self, u: Array) -> Array:
    """Discrete Laplacian ``Dxx u + Dyy u`` on a real 2D field."""
    return self.op.apply_Dxx(u) + self.op.apply_Dyy(u)

  @partial(jax.jit, static_argnums=(0,))
  def spatial(self, u: Array, m_ext: Array) -> Array:
    """No-PML wave spatial operator ``L u + omega^2 (m-1) u``.

    The ``omega^2 (m-1)`` shift makes the harmonic steady state reproduce the
    Helmholtz operator's mass term ``omega^2 m`` (see module docstring).

    Args:
      u: 2D field ``(ny, nx)``.
      m_ext: Extended model parameters ``(ny, nx)``.

    Returns:
      Spatial operator applied to ``u``.
    """
    return self._laplacian_real(u) + (self.op.omega ** 2) * (m_ext - 1.0) * u

  # ---- ADE-PML acceleration -------------------------------------------
  def _accel_pml(
    self,
    u: Array,
    mem: Tuple[Array, Array, Array, Array],
    m_ext: Array,
    s_term: Array,
  ) -> Tuple[Array, Tuple[Array, Array, Array, Array]]:
    """Unsplit ADE-PML acceleration and advanced memory variables.

    Realises the Helmholtz PML symbol ``s^{-1} d_x (s^{-1} d_x u)`` per
    direction via two memory fields driven purely by ``apply_Dx``:

      p_x = Dx u - mx1     (-> s^{-1} Dx u at steady state)
      T_x = Dx p_x - mx2   (-> s^{-1} Dx (s^{-1} Dx u))

    Memory relaxes toward its driver with the *exact* exponential solution of
    ``d_t m = sigma (driver - m)`` over one step (unconditionally stable).

    Args:
      u: Current displacement field ``(ny, nx)``.
      mem: Memory tuple ``(mx1, mx2, my1, my2)`` at the current step.
      m_ext: Extended model parameters.
      s_term: Source contribution ``S^n`` for this step (array or scalar 0).

    Returns:
      Tuple ``(acc, mem_next)`` with the acceleration ``u_tt`` and advanced
      memory variables.
    """
    mx1, mx2, my1, my2 = mem
    Dxu = self.op.apply_Dx(u)
    px = Dxu - mx1
    Dxpx = self.op.apply_Dx(px)
    Tx = Dxpx - mx2

    Dyu = self.op.apply_Dy(u)
    py = Dyu - my1
    Dypy = self.op.apply_Dy(py)
    Ty = Dypy - my2

    pot = (self.op.omega ** 2) * (m_ext - 1.0)
    acc = Tx + Ty + pot * u + s_term

    # Exact relaxation of d_t m = sigma (driver - m) with frozen driver:
    #   m_next = driver + (m - driver) * exp(-sigma*dt).
    ex, ey = self._decay_x, self._decay_y
    mem_next = (
      Dxu + (mx1 - Dxu) * ex,
      Dxpx + (mx2 - Dxpx) * ex,
      Dyu + (my1 - Dyu) * ey,
      Dypy + (my2 - Dypy) * ey,
    )
    return acc, mem_next

  # ---- leapfrog propagation (dispatches on use_pml) --------------------
  @partial(jax.jit, static_argnums=(0, 6))
  def propagate(
    self,
    u0: Array,
    v0: Array,
    m_ext: Array,
    source_field: Optional[Array],
    source_signal: Optional[Array],
    nsteps: int,
  ) -> Tuple[Array, Array, Array]:
    """Propagates ``u_tt = L u + omega^2 (m-1) u + S(x,t)`` with leapfrog.

    Forcing is the memory-light factored form ``S^n = source_field *
    source_signal[n]`` (set either to ``None`` for the unforced equation). When
    ``options.use_pml`` is True the unsplit ADE-PML memory variables are
    co-evolved so the harmonic steady state reproduces the Helmholtz PML.

    Args:
      u0: Initial displacement field ``(ny, nx)``.
      v0: Initial velocity field ``(ny, nx)``.
      m_ext: Extended model parameters ``(ny, nx)``.
      source_field: Spatial source profile ``(ny, nx)`` or ``None``.
      source_signal: Time signal ``(nsteps,)`` or ``None``.
      nsteps: Number of time steps.

    Returns:
      Tuple ``(u_final, v_final, traj)`` where ``traj`` has shape
      ``(nsteps+1, ny, nx)`` containing ``u`` at every step (including ``u0``).
    """
    dt = self.dt
    use_pml = self.options.use_pml

    def src(n):
      if source_field is None or source_signal is None:
        return jnp.zeros_like(u0)
      return source_field * source_signal[n]

    zero_mem = tuple(jnp.zeros_like(u0) for _ in range(4))

    def accel(u, mem, n):
      """Returns (acc, mem_next) for either regime."""
      if use_pml:
        return self._accel_pml(u, mem, m_ext, src(n))
      return self.spatial(u, m_ext) + src(n), mem

    # Second-order accurate startup from (u0, v0).
    a0, mem1 = accel(u0, zero_mem, 0)
    u1 = u0 + dt * v0 + 0.5 * dt * dt * a0

    def body(carry, n):
      u_curr, u_prev, mem = carry
      a, mem_next = accel(u_curr, mem, n)
      u_next = 2.0 * u_curr - u_prev + dt * dt * a
      return (u_next, u_curr, mem_next), u_curr

    (u_last, u_prev, _), traj_mid = lax.scan(
      body, (u1, u0, mem1), jnp.arange(1, nsteps)
    )
    # traj_mid records u_curr at steps n=1..nsteps-1, i.e. [u^1, ..., u^{nsteps-1}].
    # Full trajectory is [u^0, u^1, ..., u^{nsteps-1}, u^nsteps].
    traj = jnp.concatenate([u0[None], traj_mid, u_last[None]], axis=0)
    v_final = (u_last - u_prev) / dt
    return u_last, v_final, traj

  @partial(jax.jit, static_argnums=(0, 7))
  def propagate_filtered(
    self,
    u0: Array,
    v0: Array,
    m_ext: Array,
    source_field: Optional[Array],
    source_signal: Optional[Array],
    kernel: Array,
    nsteps: int,
  ) -> Tuple[Array, Array, Array]:
    """Propagates and accumulates the filter integral ``sum_n kernel[n] u^n``.

    Identical dynamics to :meth:`propagate` but the trajectory is never
    materialised: instead the time-filtered field ``sum_n kernel[n] u^n`` is
    accumulated inside the scan. This keeps memory flat under ``vmap`` over many
    right-hand sides (no ``(nsteps, ny, nx)`` buffer per RHS).

    Args:
      u0: Initial displacement (may be complex for WaveHoltz iterates).
      v0: Initial velocity.
      m_ext: Extended model parameters.
      source_field: Spatial source profile ``(ny, nx)`` or ``None``.
      source_signal: Time signal ``(nsteps,)`` or ``None``.
      kernel: Filter weights ``(nsteps+1,)`` (already include ``dt`` and the
        trapezoidal endpoint factors); the result is ``sum_{n=0}^{nsteps}
        kernel[n] u^n``.
      nsteps: Number of time steps.

    Returns:
      Tuple ``(filtered, u_final, v_final)``.
    """
    dt = self.dt
    use_pml = self.options.use_pml
    cdtype = kernel.dtype  # filter output dtype (typically complex)

    def src(n):
      if source_field is None or source_signal is None:
        return jnp.zeros_like(u0)
      return source_field * source_signal[n]

    zero_mem = tuple(jnp.zeros_like(u0) for _ in range(4))

    def accel(u, mem, n):
      if use_pml:
        return self._accel_pml(u, mem, m_ext, src(n))
      return self.spatial(u, m_ext) + src(n), mem

    a0, mem1 = accel(u0, zero_mem, 0)
    u1 = u0 + dt * v0 + 0.5 * dt * dt * a0
    acc0 = kernel[0] * u0.astype(cdtype) + kernel[1] * u1.astype(cdtype)

    def body(carry, n):
      u_curr, u_prev, mem, acc = carry
      a, mem_next = accel(u_curr, mem, n)
      u_next = 2.0 * u_curr - u_prev + dt * dt * a
      acc = acc + kernel[n + 1] * u_next.astype(cdtype)
      return (u_next, u_curr, mem_next, acc), None

    (u_last, u_prev, _, acc), _ = lax.scan(
      body, (u1, u0, mem1, acc0), jnp.arange(1, nsteps)
    )
    v_final = (u_last - u_prev) / dt
    return acc, u_last, v_final
