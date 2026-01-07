import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import jax.scipy.sparse.linalg as sp_linalg
from functools import partial
import dataclasses
from typing import Tuple, Optional, Callable, Any, Union, Literal, Protocol

# Custom type aliases.
Array = jax.Array

class Operator(Protocol):
  """Protocol for a linear operator."""
  
  def operator(self, u_vec: Array, m_ext: Array) -> Array:
    """Applies the operator A(m) u.
    
    Args:
      u_vec: Input vector (flattened).
      m_ext: Extended model parameters.
      
    Returns:
      Result of the operator application.
    """
    ...

  def operator_adjoint(self, w_vec: Array, m_ext: Array) -> Array:
    """Applies the adjoint operator A(m)^H w.
    
    Args:
      w_vec: Input adjoint vector (flattened).
      m_ext: Extended model parameters.
      
    Returns:
      Result of the adjoint operator application.
    """
    ...

  @property
  def dtype(self) -> Any:
    """Datatype for the computation and results."""
    ...

class LinearSolver(Protocol):
  """Protocol for a linear solver."""
  
  def solve(
    self, f_vec: Array, m_ext: Array, x0: Optional[Array] = None
  ) -> Tuple[Array, Any]:
    """Solves the linear system A(m) u = f.
    
    Args:
      f_vec: Right-hand side vector (flattened).
      m_ext: Extended model parameters.
      x0: Optional initial guess.
      
    Returns:
      Tuple of (solution vector, solver info).
    """
    ...

  def solve_hermitian_adjoint(
    self, g_vec: Array, m_ext: Array, x0: Optional[Array] = None
  ) -> Tuple[Array, Any]:
    """Solves the adjoint linear system A(m)^H w = g.
    
    Args:
      g_vec: Right-hand side adjoint vector (flattened).
      m_ext: Extended model parameters.
      x0: Optional initial guess.
      
    Returns:
      Tuple of (solution vector, solver info).
    """
    ...

def fd_weights(z: float, x: Array, m: int) -> Array:
  """Finite-difference weights (Fornberg algorithm).

  Args:
    z: Expansion point.
    x: Vector of evaluation points.
    m: Order of derivative.

  Returns:
    Finite-difference weights for the specified derivative.
  """
  n = len(x) - 1
  c1 = 1.0
  c4 = x[0] - z
  c = jnp.zeros((n + 1, m + 1))
  c = c.at[0, 0].set(1.0)
  
  for i in range(1, n + 1):
    mn = min(i, m)
    c2 = 1.0
    c5 = c4
    c4 = x[i] - z
    for j in range(i):
      c3 = x[i] - x[j]
      c2 = c2 * c3
      if j == i - 1:
        for k in range(mn, 0, -1):
          c = c.at[i, k].set(
            c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
          )
        c = c.at[i, 0].set(-c1 * c5 * c[i - 1, 0] / c2)
      for k in range(mn, 0, -1):
        c = c.at[j, k].set(
                (c4 * jnp.array(c[j, k]) - k * jnp.array(c[j, k - 1])) / c3
            )
      c = c.at[j, 0].set(c4 * jnp.array(c[j, 0]) / c3)
    c1 = c2
  return c[:, m]

def get_fd_1d_matrix(n: int, h: float, order: int, deriv: int) -> Array:
  """Returns a sparse matrix (as a dense matrix) for the 1D FD operator.

  Args:
    n: Grid size.
    h: Grid spacing.
    order: Accuracy order.
    deriv: Derivative order.

  Returns:
    Dense matrix representing the 1D finite-difference operator.
  """
  x_nodes = jnp.arange(order + 1)
  bulk_weights = fd_weights(float(order // 2), x_nodes, deriv) / (h**deriv)
  
  matrix = jnp.zeros((n, n))
  diags = np.arange(-(order // 2), (order // 2) + 1)
  for i, d in enumerate(diags):
    if d == 0:
      matrix += jnp.eye(n) * bulk_weights[i]
    else:
      matrix += jnp.eye(n, k=int(d)) * bulk_weights[i]
      
  # Boundary adjustments.
  for i in range(order // 2 - 1):
    w_start = fd_weights(float(i), jnp.arange(order + 3), deriv) / (h**deriv)
    matrix = matrix.at[i, :order+2].set(w_start[1:])
    
    w_end = fd_weights(
      float(order + 2 - i), jnp.arange(order + 3), deriv
    ) / (h**deriv)
    matrix = matrix.at[-(i+1), -(order+2):].set(w_end[:-1])
    
  return matrix

def distrib_pml(
  nx: int, ny: int, npml: int, fac: float
) -> Tuple[Array, Array, Array, Array]:
  """Generates PML coefficients for the grid.

  Args:
    nx: Grid size in x.
    ny: Grid size in y.
    npml: Number of PML layers.
    fac: Maximum PML damping.

  Returns:
    Tule of (sigma_x, sigma_y, sigma_xp, sigma_yp) arrays.
  """
  t = jnp.linspace(0, 1, npml)
  sigma_x = jnp.zeros((ny, nx)).at[:, :npml].set(
    fac * t[::-1]**2
  ).at[:, -npml:].set(fac * t**2)
  sigma_y = jnp.zeros((ny, nx)).at[:npml, :].set(
    fac * t[:, None][::-1]**2
  ).at[-npml:, :].set(fac * t[:, None]**2)
  sigma_xp = jnp.zeros((ny, nx)).at[:, :npml].set(
    -2 * fac * t[::-1]
  ).at[:, -npml:].set(2 * fac * t)
  sigma_yp = jnp.zeros((ny, nx)).at[:npml, :].set(
    -2 * fac * t[:, None][::-1]
  ).at[-npml:, :].set(2 * fac * t[:, None])
  return sigma_x, sigma_y, sigma_xp, sigma_yp

def extend_model(m: Array, nxint: int, nyint: int, npml: int) -> Array:
  """Extends the model from the interior to the full domain (including PML).

  Args:
    m: Interior model parameters.
    nxint: Interior grid size in x.
    nyint: Interior grid size in y.
    npml: Number of PML layers.

  Returns:
    Extended model parameters with zero padding in PML.
  """
  m = m.reshape((nyint, nxint))
  ny = nyint + 2 * npml
  nx = nxint + 2 * npml
  m_ext = jnp.zeros((ny, nx))
  m_ext = m_ext.at[npml:-npml, npml:-npml].set(m)
  return m_ext

@dataclasses.dataclass
class GMRESOptions:
  """GMRES solver options.
  
  Attributes:
    tol: Tolerance for the solver.
    maxiter: Maximum number of iterations.
  """
  tol: float = 1e-3
  maxiter: int = 1000

class HelmholtzOperator:
  """Helmholtz operator using JAX.

  Attributes:
    nx: Total grid size in x.
    ny: Total grid size in y.
    npml: Number of PML layers.
    h: Grid spacing.
    omega: Angular frequency.
    sigma_max: Maximum PML damping.
    order: Accuracy order.
    mode: Operator implementation mode ('matrix', 'stencil', 'conv').
        'matrix': dense matrix representation per dimension.
        'stencil': stencil representation for matrix-free multiplication.
        'conv': convolution representation.
    dtype: Datatype for the computation and results.
  """
  def __init__(
    self, 
    nx: int, 
    ny: int, 
    npml: int, 
    h: float, 
    omega: float, 
    sigma_max: float, 
    order: int = 2,
    mode: Literal['matrix', 'stencil', 'conv'] = 'matrix',
    dtype: Any = jnp.complex128
  ):
    """Initializes the Helmholtz operator."""
    self.nx = nx
    self.ny = ny
    self.npml = npml
    self.h = h
    self.omega = omega
    self.sigma_max = sigma_max
    self.order = order
    self.mode = mode
    self.mode = mode
    self._dtype = dtype
    
    self.sx, self.sy, self.sxp, self.syp = distrib_pml(nx, ny, npml, sigma_max)
    
    denom_x = (1 + 1j / omega * self.sx)
    denom_y = (1 + 1j / omega * self.sy)
    self.cx = (1j / (omega * (npml - 1) * h) * self.sxp / denom_x**3).astype(dtype)
    self.cy = (1j / (omega * (npml - 1) * h) * self.syp / denom_y**3).astype(dtype)
    self.ax = (-1.0 / denom_x**2).astype(dtype)
    self.ay = (-1.0 / denom_y**2).astype(dtype)
    
    # Precompute weights.
    self.weights_1 = fd_weights(
      float(order // 2), jnp.arange(order + 1), 1
    ) / (h**1)
    self.weights_2 = fd_weights(
      float(order // 2), jnp.arange(order + 1), 2
    ) / (h**2)
    
    # Boundary weights.
    b_range = range(order//2 - 1)
    if list(b_range):
      self.b_weights_1_start = jnp.stack([
        fd_weights(float(i), jnp.arange(order + 3), 1)[1:] / (h**1) 
        for i in b_range
      ])
      self.b_weights_1_end = jnp.stack([
        fd_weights(float(order + 2 - i), jnp.arange(order + 3), 1)[:-1] / (h**1) 
        for i in b_range
      ])
      self.b_weights_2_start = jnp.stack([
        fd_weights(float(i), jnp.arange(order + 3), 2)[1:] / (h**2) 
        for i in b_range
      ])
      self.b_weights_2_end = jnp.stack([
        fd_weights(float(order + 2 - i), jnp.arange(order + 3), 2)[:-1] / (h**2) 
        for i in b_range
      ])
    else:
      self.b_weights_1_start = self.b_weights_1_end = \
        self.b_weights_2_start = self.b_weights_2_end = \
        jnp.zeros((0, order+2))

    if mode == 'matrix':
      self.Dx1d = get_fd_1d_matrix(nx, h, order, 1).astype(dtype)
      self.Dy1d = get_fd_1d_matrix(ny, h, order, 1).astype(dtype)
      self.Dxx1d = get_fd_1d_matrix(nx, h, order, 2).astype(dtype)
      self.Dyy1d = get_fd_1d_matrix(ny, h, order, 2).astype(dtype)

  @property
  def dtype(self) -> Any:
    """Datatype for the computation and results."""
    return self._dtype

  @partial(jax.jit, static_argnums=(0, 2, 3))
  def _apply_2d_core(self, u: Array, dim: int, deriv: int) -> Array:
    """Core 2D operator application for stencil or conv modes.

    Args:
      u: Input 2D field.
      dim: Dimension to apply (0 for x, 1 for y).
      deriv: Derivative order.

    Returns:
      Resulting field.
    """
    half_order = self.order // 2
    weights = (
      self.weights_1 if deriv == 1 else self.weights_2
    ).astype(u.dtype)
    
    if self.mode == 'stencil':
      res = jnp.zeros_like(u)
      diags = np.arange(-half_order, half_order + 1)
      for i, d in enumerate(diags):
        if d == 0:
          res += weights[i] * u
        elif d > 0:
          if dim == 0: # x-direction
            res = res.at[:, :-d].add(weights[i] * u[:, d:])
          else: # y-direction
            res = res.at[:-d, :].add(weights[i] * u[d:, :])
        else: # d < 0
          if dim == 0:
            res = res.at[:, -d:].add(weights[i] * u[:, :d])
          else:
            res = res.at[-d:, :].add(weights[i] * u[:d, :])

    elif self.mode == 'conv':
      if dim == 0: # x-direction
        kernel = weights.reshape((1, 1, 1, -1))
      else: # y-direction
        kernel = weights.reshape((1, 1, -1, 1))
      # Input shape: [N, C, H, W]
      res = lax.conv_general_dilated(
        u[None, None, :, :], kernel, (1, 1), 'SAME'
      )[0, 0]
    
    if self.order > 2:
      b_s = (
        self.b_weights_1_start if deriv == 1 else self.b_weights_2_start
      ).astype(u.dtype)
      b_e = (
        self.b_weights_1_end if deriv == 1 else self.b_weights_2_end
      ).astype(u.dtype)
      for i in range(half_order - 1):
        if dim == 0: # x-direction
          res = res.at[:, i].set(u[:, :self.order+2] @ b_s[i])
          res = res.at[:, -(i+1)].set(u[:, -(self.order+2):] @ b_e[i])
        else: # y-direction
          res = res.at[i, :].set(b_s[i] @ u[:self.order+2, :])
          res = res.at[-(i+1), :].set(b_e[i] @ u[-(self.order+2):, :])
    return res

  @partial(jax.jit, static_argnums=(0, 2, 3, 4))
  def _apply_derivative(
    self, u: Array, dim: int, deriv: int, adjoint: bool = False
  ) -> Array:
    """Applies the derivative operator along a dimension.

    Args:
      u: Input 2D field.
      dim: Dimension to apply (0 for x, 1 for y).
      deriv: Derivative order.
      adjoint: If True, applies the hermitian adjoint.

    Returns:
      Resulting field.
    """
    if self.mode == 'matrix':
      if dim == 0:
        mat = self.Dx1d if deriv == 1 else self.Dxx1d
        if adjoint: mat = mat.T.conj()
        return jax.vmap(lambda row: mat @ row)(u)
      else:
        mat = self.Dy1d if deriv == 1 else self.Dyy1d
        if adjoint: mat = mat.T.conj()
        return jax.vmap(lambda col: mat @ col, in_axes=1, out_axes=1)(u)
    
    def apply_f(field):
      return self._apply_2d_core(field, dim, deriv)
    
    if not adjoint:
      return apply_f(u)
    else:
      def adj_fun(field):
        _, vjp_fun = jax.vjp(apply_f, jnp.zeros_like(field))
        return jnp.conj(vjp_fun(jnp.conj(field))[0])
      return adj_fun(u)

  @partial(jax.jit, static_argnums=(0,))
  def apply_Dx(self, u: Array) -> Array: 
    """Applies Dx operator."""
    return self._apply_derivative(u, 0, 1)

  @partial(jax.jit, static_argnums=(0,))
  def apply_Dy(self, u: Array) -> Array: 
    """Applies Dy operator."""
    return self._apply_derivative(u, 1, 1)

  @partial(jax.jit, static_argnums=(0,))
  def apply_Dxx(self, u: Array) -> Array: 
    """Applies Dxx operator."""
    return self._apply_derivative(u, 0, 2)

  @partial(jax.jit, static_argnums=(0,))
  def apply_Dyy(self, u: Array) -> Array: 
    """Applies Dyy operator."""
    return self._apply_derivative(u, 1, 2)

  @partial(jax.jit, static_argnums=(0,))
  def operator(self, u_vec: Array, m_ext: Array) -> Array:
    """Applies the Helmholtz operator A(m) u.

    Args:
      u_vec: Input field (flattened).
      m_ext: Extended model parameters.

    Returns:
      Applied operator result (flattened).
    """
    u = u_vec.reshape((self.ny, self.nx))
    res = - (self.omega**2) * m_ext.astype(self.dtype) * u
    res += self.cx * self.apply_Dx(u)
    res += self.cy * self.apply_Dy(u)
    res += self.ax * self.apply_Dxx(u)
    res += self.ay * self.apply_Dyy(u)
    return res.flatten()

  @partial(jax.jit, static_argnums=(0,))
  def operator_adjoint(self, w_vec: Array, m_ext: Array) -> Array:
    """Applies the Hermitian adjoint Helmholtz operator A(m)^H w.

    Args:
      w_vec: Input adjoint field (flattened).
      m_ext: Extended model parameters.

    Returns:
      Applied hermitian operator result (flattened).
    """
    w = w_vec.reshape((self.ny, self.nx))
    res = - jnp.conj(self.omega**2 * m_ext.astype(self.dtype)) * w
    res += self._apply_derivative(jnp.conj(self.cx) * w, 0, 1, adjoint=True)
    res += self._apply_derivative(jnp.conj(self.cy) * w, 1, 1, adjoint=True)
    res += self._apply_derivative(jnp.conj(self.ax) * w, 0, 2, adjoint=True)
    res += self._apply_derivative(jnp.conj(self.ay) * w, 1, 2, adjoint=True)
    return res.flatten()


class HelmholtzSolver:
  """Helmholtz solver that composes an Operator and a linear solver strategy.
  
  Attributes:
    op: The Helmholtz operator instance.
    gmres_options: Options for the GMRES solver.
  """
  def __init__(
    self, 
    op: Operator, 
    gmres_options: GMRESOptions = GMRESOptions()
  ):
    """Initializes the solver with an operator and options."""
    self.op = op
    self.gmres_options = gmres_options

  @property
  def nx(self): 
    """Grid size in x."""
    return self.op.nx

  @property
  def ny(self): 
    """Grid size in y."""
    return self.op.ny

  @property
  def npml(self): 
    """Number of PML layers."""
    return self.op.npml

  @property
  def h(self): 
    """Grid spacing."""
    return self.op.h

  @property
  def omega(self): 
    """Angular frequency."""
    return self.op.omega
  
  @partial(jax.jit, static_argnums=(0,))
  def solve(
    self, f_vec: Array, m_ext: Array, x0: Optional[Array] = None
  ) -> Tuple[Array, Any]:
    """Solves the forward Helmholtz problem for a single RHS.

    Args:
      f_vec: RHS vector (flattened).
      m_ext: Extended model parameters.
      x0: Initial guess.

    Returns:
      Tuple of (solution, info).
    """
    # Ensure output-compatible dtypes.
    # Ensure output-compatible dtypes.
    f_vec = f_vec.astype(self.op.dtype)
    if x0 is not None:
      x0 = x0.astype(self.op.dtype)

    def op_fun(u): return self.op.operator(u, m_ext)
    gmres_kwargs = dataclasses.asdict(self.gmres_options)
    sol, info = sp_linalg.gmres(
      op_fun, f_vec, x0=x0, **gmres_kwargs
    )
    return sol, info

  @partial(jax.jit, static_argnums=(0,))
  def solve_hermitian_adjoint(
    self, g_vec: Array, m_ext: Array, x0: Optional[Array] = None
  ) -> Tuple[Array, Any]:
    """Solves the adjoint Helmholtz problem for a single RHS.

    Args:
      g_vec: RHS adjoint vector (flattened).
      m_ext: Extended model parameters.
      x0: Initial guess.

    Returns:
      Tuple of (solution, info).
    """
    # Ensure output-compatible dtypes.
    g_vec = g_vec.astype(self.op.dtype)
    if x0 is not None:
      x0 = x0.astype(self.op.dtype)

    def op_fun(w): return self.op.operator_adjoint(w, m_ext)
    gmres_kwargs = dataclasses.asdict(self.gmres_options)
    sol, info = sp_linalg.gmres(
      op_fun, g_vec, x0=x0, **gmres_kwargs
    )
    return sol, info