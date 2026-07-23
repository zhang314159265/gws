import jax
import jax.numpy as jnp
from jax.extend import core

s = jnp.ones(5) * 10

def f(x):
    # return jnp.sin(x) + 1
    return x * s

x = jnp.arange(5)
jaxpr = jax.make_jaxpr(f)(x)

fun = core.jaxpr_as_fun(jaxpr)

print(jaxpr)
print(fun(x))
print(fun(jnp.arange(5) + 5))

# didn't broadcast
# print(fun(jnp.ones([2, 5])))
# print(fun(jnp.ones([5, 2])))

# dynamic shape not supported in general
# print(fun(jnp.arange(6)))
