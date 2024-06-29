#include "cuda.h"
#include <iostream>
#include <fstream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <unistd.h>
#include "tritoncc/CUDAUtil.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Require cubin path" << std::endl;
    exit(1);
  }

  torch::Tensor a = torch::randn({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor b = torch::randn({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor expected = torch::matmul(a, b) + 5.0f;
  // std::cerr << "Expected:\n" << expected << std::endl;

  torch::Tensor actual = torch::zeros({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  const char* name = "dot_fn";
  if (argc >= 3) {
    name = argv[2];
  }
  int shared = 8192;
  CUfunction fun = tritoncc::loadCUDAFunctionFromFile(argv[1], name, shared);
  { // launch kernel
    int gridX = 1; // TODO don't hard code
    int gridY = 1;
    int gridZ = 1;
    int num_warps = 4;
    CUstream stream = 0;
    void* data_ptr_a = a.data_ptr();
    void* data_ptr_b = b.data_ptr();
    void* data_ptr_out = actual.data_ptr();
    void* params[] = {&data_ptr_a, &data_ptr_b, &data_ptr_out};
    CUDA_CHECK(cuLaunchKernel(fun, gridX, gridY, gridZ, 32 * num_warps, 1, 1, shared, stream, params, 0));
  }
  // print a cuda tensor will trigger synchronization?
  // std::cerr << "actual:\n" << actual << std::endl;
  double tol = 1e-2;
  if (at::allclose(expected, actual, tol, tol)) {
    std::cerr << "PASS!" << std::endl;
  } else {
    std::cerr << "FAIL!" << std::endl;
  }

  std::cerr << "bye" << std::endl;
  return 0;
}
