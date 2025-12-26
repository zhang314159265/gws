import torch
from torch.utils.cpp_extension import load_inline
import ctypes

cpp_source = r"""
#include <stdio.h>

extern "C" {
void print_msg(void) {
    printf("A message from C++\n");
}
}
"""

path = load_inline(
    name="myso",
    cpp_sources=cpp_source,
    is_python_module=False,
)
so_handle = ctypes.CDLL(path)

so_handle.print_msg()
