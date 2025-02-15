#include "cute/tensor.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void kernel(float *ptr) {
  if (cute::thread0()) {
    auto layout = cute::make_layout(cute::make_shape(3, 5));
    cute::print(layout);
    cute::print("\n");
    cute::print_layout(layout);
    cute::print("\n");
    cute::print(ptr);
    cute::print("\n");

    auto tensor = make_tensor(ptr, layout);
    cute::print_tensor(tensor);
    cute::print("\n");

    cute::print(tensor(1, 2)); // 7.0f
    cute::print("\n");
    cute::print_latex(layout);
    cute::print("\n");
  }
}

int main() {
  using Element = float;

  thrust::host_vector<Element> h_S(15);
  for (int i = 0; i < h_S.size(); i++) {
    h_S[i] = static_cast<Element>(i);
  }
  thrust::device_vector<Element> d_S = h_S;

  kernel<<<32, 1024>>>(thrust::raw_pointer_cast(d_S.data()));

  cudaDeviceSynchronize();
  return 0;
}
