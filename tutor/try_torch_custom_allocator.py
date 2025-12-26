import torch
import ctypes
import gc
import torch.utils.cpp_extension
from torch.cuda.memory import MemPool

cuda_source = r"""
#include "cuda_runtime.h"

extern "C" {
void print_msg() {
    printf("Hello!\n");
}

void *my_malloc(size_t sz, int device, cudaStream_t stream) {
    void *ptr;
    cudaMallocAsync(&ptr, sz, stream);
    printf("Allocate %d bytes on stream %lld\n", sz, (int64_t) stream);
    return ptr;
}

void my_free(void* ptr, size_t sz, int device, cudaStream_t stream) {
    cudaFreeAsync(ptr, stream);
    printf("Free %d bytes on stream %lld\n", sz, (int64_t) stream);
}

}

"""

so_path = torch.utils.cpp_extension.load_inline(
    name = "myso",
    cpp_sources="",
    cuda_sources=cuda_source,
    is_python_module=False,
)
ctypes.CDLL(so_path).print_msg()

allocator = torch.cuda.memory.CUDAPluggableAllocator(
    so_path, "my_malloc", "my_free",    
)

mem_pool = MemPool(allocator.allocator())
with torch.cuda.use_mem_pool(mem_pool):
    torch.randn(15, device="cuda")

del mem_pool  # without this, the code will segfault in ~MemPool

print("bye")
