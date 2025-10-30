#include <pybind11/pybind11.h>

// long type since pass tensor.data_ptr() to float* directly
// cause wrong addresses used in the cuda kernel.
void plus1(long src, long dst, int N);

PYBIND11_MODULE(mylib, m) {
  m.def("plus1", plus1);
}
