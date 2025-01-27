"""
Compare the PTX with or without use_fast_math.
Example output: https://gist.github.com/shunting314/7c06d1b641b499ce048e133051b7f05b 
"""

import itertools
from cuda_to_ptx import cuda_to_ptx

oplist = [
    "sin",
    "exp",
    "sqrt",
]

for op, use_fast_math in itertools.product(oplist, [False, True]):
    print("=====================")
    print(f"{op=} {use_fast_math=}")
    cuda_str = f"""
    __global__ void kernel_{op}(float *p, float x) {{
        p[0] = {op}(x);
    }}
    """
    ptx_str = cuda_to_ptx(cuda_str, use_fast_math=use_fast_math)
    print(ptx_str)

print("bye")
