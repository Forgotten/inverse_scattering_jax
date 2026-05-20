import unittest
import jax
import jax.numpy as jnp
import numpy as np
from inverse_scattering_jax.src.inverse_scattering import get_projection_op

class TestProjection(unittest.TestCase):
    
    def setUp(self):
        self.nx = 20
        self.ny = 20
        self.h = 0.1
        self.x = jnp.arange(self.nx) * self.h
        self.y = jnp.arange(self.ny) * self.h
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='xy') # Match (ny, nx) layout

    def test_adjointness(self):
        """Test the adjointness of the projection operator."""
        # Random query points within the domain
        n_points = 15
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        points_x = jax.random.uniform(k1, (n_points,), minval=0, maxval=(self.nx-1)*self.h)
        points_y = jax.random.uniform(k2, (n_points,), minval=0, maxval=(self.ny-1)*self.h)
        points_query = jnp.stack([points_x, points_y], axis=1)
        
        op = get_projection_op(self.x, self.y, points_query)
        
        # Multiple inputs (batch dim should be axis 1)
        n_batch = 5
        u = jax.random.normal(k3, (self.nx * self.ny, n_batch)) + \
            1j * jax.random.normal(k3, (self.nx * self.ny, n_batch))
        
        # Forward application
        d = op(u)
        
        # Adjoint application using VJP
        v = jax.random.normal(k4, d.shape) + 1j * jax.random.normal(k4, d.shape)
        
        def apply_op_sum(u_in):
            return jnp.sum(op(u_in) * jnp.conj(v))
            
        # VJP gives partial/partial u^* (technically just vjp of linear map is adjoint * vector)
        # JAX vjp of linear f(u) -> u^T A^T v. We want <Au, v> = u^H A^H v.
        # In JAX complex: vjp(conj(v))[0] gives conjugate of (A^T conj(v))? No.
        # Let's stick to standard def: <Au, v> = sum(Au * conj(v))
        # <u, A* v> = sum(u * conj(A* v))
        
        # We can use jax.linear_transpose or vjp.
        # linear_transpose expects a doublet like (primals_out, cotangents_in) structure ?? 
        # simpler: just vjp.
        
        primals_out, vjp_fun = jax.vjp(op, u)
        # vjp_fun(v) returns (w,) where w = A^T v.
        # For complex numbers, adjoint is conjugate transpose.
        # <Au, v> = u . A^H v
        # LHS = sum( (Au) * conj(v) )
        # RHS = sum( u * conj( A^H v ) )
        
        # JAX VJP computes v^T J. If J is linear matrix A, it computes v^T A = (A^T v)^T.
        # So vjp(v) gives A^T v.
        # We need A^H v = conj( A^T conj(v) ).
        
        w_transposed = vjp_fun(jnp.conj(v))[0] # A^T conj(v)
        w_adjoint = jnp.conj(w_transposed)     # conj(A^T conj(v)) = A^H v
        
        lhs = jnp.vdot(d, v) # <Au, v>
        rhs = jnp.vdot(u, w_adjoint) # <u, A^H v>
        
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

    def test_identity_on_grid(self):
        """Test that P^H P is identity when sampling all grid points."""
        # Query points are exactly the grid points
        X_flat = self.X.flatten()
        Y_flat = self.Y.flatten()
        points_query = jnp.stack([X_flat, Y_flat], axis=1)
        
        # Note: input to get_projection_op is x, y 1D arrays
        # The projection_op expects points as [N, 2]
        # and assumes u is flattened as [Batch, ny*nx]? Or [Batch, nx*ny]?
        # We need to verify ordering. The code says:
        # u = u_vec.reshape((ny, nx))
        # interp = RegularGridInterpolator((y, x), u)
        # points in (y, x) order for interpolator?
        # code: interp(points_query[:, ::-1]) -> flips (x,y) to (y,x) for interp.
        # Correct.
        
        op = get_projection_op(self.x, self.y, points_query)
        
        # Create a random field
        n_batch = 1
        key = jax.random.PRNGKey(42)
        u = jax.random.normal(key, (self.nx * self.ny, n_batch))
        
        # Forward P u
        d = op(u)
        
        # For P to be identity, d should be equal to u (up to machine precision)
        # since we sample exactly at grid nodes.
        np.testing.assert_allclose(d, u, rtol=1e-5, atol=1e-5)
        
        # Now check adjoint P^H d.
        # If P = I, then P^H = I, so P^H P = I.
        
        _, vjp_fun = jax.vjp(op, u)
        # With real numbers, A^H = A^T.
        # If P is identity matrix, transpose is identity.
        
        w = vjp_fun(d)[0] # P^T (P u)
        
        np.testing.assert_allclose(w, u, rtol=1e-5, atol=1e-5)

    def test_subset_identity(self):
        """Test that P P^H = I when sampling a subset of grid points."""
        # Randomly select a subset of grid indices
        n_points = 50
        key = jax.random.PRNGKey(10)
        idx_linear = jax.random.choice(key, self.nx * self.ny, (n_points,), replace=False)
        
        points_x = self.X.flatten()[idx_linear]
        points_y = self.Y.flatten()[idx_linear]
        points_query = jnp.stack([points_x, points_y], axis=1)
        
        op = get_projection_op(self.x, self.y, points_query)
        
        # Random data vector in Data Space
        k2 = jax.random.PRNGKey(11)
        n_batch = 5
        d = jax.random.normal(k2, (n_points, n_batch)) + \
            1j * jax.random.normal(k2, (n_points, n_batch))
            
        # P^H d
        # Use vjp for adjoint application
        # Since op is linear, we can get vjp at any point (e.g. zeros)
        # We must ensure the input is complex so JAX traces it as complex.
        dummy_u = jnp.zeros((self.nx * self.ny, n_batch), dtype=jnp.complex128)
        _, vjp_fun = jax.vjp(op, dummy_u)
        
        # vjp_fun(d) returns P^T d. For complex, we want P^H d.
        # But wait, JAX vjp behavior for complex:
        # If f(u) = A u. vjp(v) = A^T v.
        # We want P P^H d.
        # P^H d = conj( P^T conj(d) ).
        # Let v = conj(d). Then w = vjp(v)[0] = P^T conj(d).
        # P^H d = conj(w).
        
        w_transposed = vjp_fun(jnp.conj(d))[0] # P^T conj(d)
        w_adjoint = jnp.conj(w_transposed)     # P^H d
        
        # Now apply P to w_adjoint
        res = op(w_adjoint) # P (P^H d)
        
        # Should be equal to d
        np.testing.assert_allclose(res, d, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
