import cutlass
from cutlass import cute
import torch

@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("hello world!")

@cute.jit(preprocess=False)
def hello_world():
    cute.printf("hello world! (from host)")

    kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1)
    )

cutlass.cuda.initialize_cuda_context()
print("Running hello_world()")
hello_world()

print("Compiling")
hello_world_compiled = cute.compile(hello_world)

print("Running compiled version")
hello_world_compiled()

torch.cuda.synchronize()
