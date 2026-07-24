import jax
import jax.numpy as jnp

@jax.jit
def f(x, i):
    print(f"hello {i}")  # only run once to print a tracer
    print(f"filename {f.__name__}")
    return x + 1

x = jnp.ones((5,))
f(x, 0)
f(x, 1)
f(x, 2)
