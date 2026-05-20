import dataclasses
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import jaxopt
from jax import custom_vjp
from .helmholtz import HelmholtzSolver, extend_model, Array
from typing import Callable, Tuple, Any, Optional

@dataclass(slots=True, kw_only=True)
class IncomingDirections:
  """Encapsulates incident waves from multiple directions.

  Attributes:
    nx: Grid size in x.
    ny: Grid size in y.
    npml: Number of PML layers.
    h: Grid spacing.
    omega: Angular frequency.
    n_theta: Number of incident directions.
    x: 1D grid coordinates in x.
    y: 1D grid coordinates in y.
    grid_x: 2D meshgrid in x.
    grid_y: 2D meshgrid in y.
    theta: Array of incident angles.
    directions: Array of incident direction vectors.
    u_in: Incident wavefields for all directions.
  """
  nx: int
  ny: int
  npml: int
  h: float
  omega: float
  n_theta: int

  # Computed fields (not initialized via generated __init__)
  x: Array = field(init=False)
  y: Array = field(init=False)
  grid_x: Array = field(init=False)
  grid_y: Array = field(init=False)
  theta: Array = field(init=False)
  directions: Array = field(init=False)
  u_in: Array = field(init=False)

  def __post_init__(self):
    """Initializes the incoming directions and precomputes incident waves."""
    # Grid.
    self.x = (jnp.arange(self.nx) - self.npml - (self.nx - 2 * self.npml)//2) * self.h
    self.y = (jnp.arange(self.ny) - self.npml - (self.ny - 2 * self.npml)//2) * self.h
    self.grid_x, self.grid_y = jnp.meshgrid(self.x, self.y, indexing='ij')
    
    # Directions.
    dtheta = 2 * jnp.pi / self.n_theta
    self.theta = jnp.linspace(jnp.pi, 3 * jnp.pi - dtheta, self.n_theta)
    self.directions = jnp.stack([jnp.cos(self.theta), jnp.sin(self.theta)], axis=1)
    
    # Incident waves (nx * ny, n_theta).
    self.u_in = jnp.exp(
      1j * self.omega * (
        self.grid_x.flatten()[:, None] * self.directions[:, 0] + 
        self.grid_y.flatten()[:, None] * self.directions[:, 1]
      )
    )

  def get_rhs(self, eta_ext: Array) -> Array:
    """Computes the RHS for the scattered field equation.

    Args:
      eta_ext: Extended model perturbation.

    Returns:
      Right-hand side vector for all incident directions.
    """
    # S = -omega^2 * eta_ext * u_in.
    return -(self.omega**2) * eta_ext.flatten()[:, None] * self.u_in

def get_projection_op(
  x: Array, y: Array, points_query: Array
) -> Callable[[Array], Array]:
  """Returns a function that interpolates the wavefield at points_query.

  Args:
    x: Grid coordinates in x.
    y: Grid coordinates in y.
    points_query: Observation points [n_obs, 2].

  Returns:
    Function that projects the full wavefield onto the observation points.
  """
  from jax.scipy.interpolate import RegularGridInterpolator
  
  def projection_op(U: Array) -> Array:
    nx = len(x)
    ny = len(y)
    
    def interp_one(u_vec):
      u = u_vec.reshape((ny, nx))
      interp = RegularGridInterpolator((y, x), u)
      return interp(points_query[:, ::-1])
      
    return jax.vmap(interp_one, in_axes=1, out_axes=1)(U)
    
  return projection_op

class ForwardModel:
  """High-level forward model for the scattering problem.

  Attributes:
    solver: Helmholtz solver object instance.
    inc: IncomingDirections object instance.
    projection_op: Operator to sample the scattered field.
  """
  def __init__(
    self, 
    solver: HelmholtzSolver, 
    inc: IncomingDirections, 
    projection_op: Optional[Callable[[Array], Array]] = None
  ):
    """Initializes the forward model."""
    self.solver = solver
    self.inc = inc
    self.projection_op = projection_op if projection_op is not None else \
      lambda u: u

  def forward(self, eta: Array) -> Array:
    """Computes the scattered field for a given perturbation.

    Args:
      eta: Model perturbation (interior).

    Returns:
      Scattered field sampled at observation points.
    """
    nxi = self.solver.nx - 2 * self.solver.npml
    nyi = self.solver.ny - 2 * self.solver.npml
    eta_ext = extend_model(eta, nxi, nyi, self.solver.npml)
    m_ext = 1.0 + eta_ext
    rhs = self.inc.get_rhs(eta_ext)
    
    def solve_one(b: Array) -> Array:
      sol, _ = self.solver.solve(b, m_ext)
      return sol
    
    U = jax.vmap(solve_one, in_axes=1, out_axes=1)(rhs)
    return self.projection_op(U)

def create_forward_with_adjoint(
  solver: HelmholtzSolver, 
  inc: IncomingDirections, 
  projection_op: Callable[[Array], Array]
) -> Callable[[Array], Array]:
  """
  Creates a forward function with a custom adjoint (VJP).

  Args:
    solver: Helmholtz solver instance.
    inc: Incoming directions instance.
    projection_op: Observation operator.

  Returns:
    Function mapping eta to scattered field with custom adjoint support.
  """
  
  @custom_vjp
  def forward_fun(eta: Array) -> Array:
    nxi = solver.nx - 2 * solver.npml
    nyi = solver.ny - 2 * solver.npml
    eta_ext = extend_model(eta, nxi, nyi, solver.npml)
    m_ext = 1.0 + eta_ext
    rhs = inc.get_rhs(eta_ext)
    
    def solve_one(b: Array) -> Array:
      sol, _ = solver.solve(b, m_ext)
      return sol
    
    u_0 = jax.vmap(solve_one, in_axes=1, out_axes=1)(rhs)
    return projection_op(u_0)

  def forward_fwd(eta: Array) -> Tuple[Array, Tuple[Array, Array]]:
    nxi = solver.nx - 2 * solver.npml
    nyi = solver.ny - 2 * solver.npml
    eta_ext = extend_model(eta, nxi, nyi, solver.npml)
    m_ext = 1.0 + eta_ext
    rhs = inc.get_rhs(eta_ext)
    
    def solve_one(b: Array) -> Array:
      sol, _ = solver.solve(b, m_ext)
      return sol
      
    u_0 = jax.vmap(solve_one, in_axes=1, out_axes=1)(rhs)
    u_0_scattered = projection_op(u_0)
    return u_0_scattered, (eta, u_0)

  def forward_bwd(res: Tuple[Array, Array], v: Array) -> Tuple[Array]:
    """Computes the Vector-Jacobian Product (VJP) for the scattered field.

    Mathematically, for a perturbation eta, we solve:
      w = A^H \\ pt_v
    where pt_v is the projected cotangent vector in the observation space,
    and A is the Helmholtz operator. Since the operator A is complex symmetric, 
    we compute the adjoint wavefield stably using:
      w = conj( A^{-1} \\ conj(pt_v) )
    to avoid PML boundary amplification instability.

    Args:
      res: Saved forward variables (eta, u_0) from the forward pass.
      v: Cotangent vector corresponding to the scattered field outputs.

    Returns:
      A tuple containing the gradient vector with respect to the model perturbation eta.
    """
    eta, u_0 = res
    v = jnp.conj(v)
    nxi = solver.nx - 2 * solver.npml
    nyi = solver.ny - 2 * solver.npml
    npml = solver.npml
    
    eta_ext = extend_model(eta, nxi, nyi, npml)
    m_ext = 1.0 + eta_ext
    
    # check if this is correct.
    _, vjp_proj = jax.vjp(projection_op, u_0)
    pt_v = vjp_proj(v)[0]
    
    def solve_adj_one(b):
      sol, _ = solver.solve(jnp.conj(b), m_ext)
      return jnp.conj(sol)
      
    W = jax.vmap(solve_adj_one, in_axes=1, out_axes=1)(pt_v)
    U_total = u_0 + inc.u_in
    grad_ext = - jnp.real(
      jnp.sum(jnp.conj(U_total) * W, axis=1)
    ) * (solver.omega**2)
    grad_ext = grad_ext.reshape((solver.ny, solver.nx))
    grad = grad_ext[npml:-npml, npml:-npml]
    
    return (grad.flatten(),)

  forward_fun.defvjp(forward_fwd, forward_bwd)
  return forward_fun

def misfit(
  eta: Array, forward_fun: Callable[[Array], Array], data: Array
) -> Array:
  """
  Computes the L2 misfit between model and data.

  Args:
    eta: Model perturbation.
    forward_fun: Forward model function.
    data: Target observation data.

  Returns:
    Scalar misfit value.
  """
  scattered = forward_fun(eta)
  diff = scattered - data
  return 0.5 * jnp.sum(jnp.abs(diff)**2)

def solve_inverse_problem(
  eta_init: Array, 
  forward_fun: Callable[[Array], Array], 
  data: Array, 
  maxiter: int = 50, 
  learning_rate: float = 1e-2
) -> Tuple[Array, Any]:
  """
  Solves the inverse scattering problem using L-BFGS.

  Args:
    eta_init: Initial guess for the perturbation.
    forward_fun: Forward model function (with adjoint support).
    data: Target observation data.
    maxiter: Maximum L-BFGS iterations.
    learning_rate: Learning rate (not directly used by JAXOpt L-BFGS).

  Returns:
    Tuple of (optimized parameters, final state).
  """
  def objective(eta):
    return misfit(eta, forward_fun, data)
  solver_opt = jaxopt.LBFGS(fun=objective, maxiter=maxiter)
  res = solver_opt.run(eta_init)
  return res.params, res.state
