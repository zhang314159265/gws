import jax
import jax.numpy as jnp

# outputs are stacked
def f(carry, x):
    out = carry + x
    return out, out

x = jnp.arange(5)[:, None]
print(jax.lax.scan(f, jnp.array([100]), x))
