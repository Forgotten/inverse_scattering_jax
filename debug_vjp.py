import jax
import jax.numpy as jnp

def test_vjp():
    A = jnp.array([[1+1j, 2-1j], [0.5j, 3]])
    def f(x):
        return A @ x
    
    x = jnp.array([1.0, 2.0+1j])
    w = jnp.array([1j, 1.0])
    
    y, vjp_fun = jax.vjp(f, x)
    v = vjp_fun(jnp.conj(w))[0]
    v = jnp.conj(v)
    
    AHw = A.conj().T @ w
    
    print(f"JAX VJP output: {v}")
    print(f"A^H w: {AHw}")
    print(f"Match: {jnp.allclose(v, AHw)}")

if __name__ == "__main__":
    test_vjp()
