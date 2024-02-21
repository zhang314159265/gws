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

  torch::Tensor x = torch::randn({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor y = torch::randn({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor expected = torch::add(x, y);
  std::cerr << "expected:\n" << expected << std::endl;
  torch::Tensor actual = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  const char* name = "pointwise_fn";
  if (argc >= 3) {
    name = argv[2];
  }
  int shared = 0;
  CUfunction fun = tritoncc::loadCUDAFunctionFromFile(argv[1], name, shared);
  { // launch kernel
    int gridX = 32; // TODO don't hard code
    int gridY = 1;
    int gridZ = 1;
    int num_warps = 4;
    CUstream stream = 0;
    void* data_ptr_lhs = x.data_ptr();
    void* data_ptr_rhs = y.data_ptr();
    void* data_ptr_out = actual.data_ptr();
    int numel = 1024;
    void* params[] = {&data_ptr_lhs, &data_ptr_rhs, &data_ptr_out, &numel};
    CUDA_CHECK(cuLaunchKernel(fun, gridX, gridY, gridZ, 32 * num_warps, 1, 1, shared, stream, params, 0));
  }
  // print a cuda tensor will trigger synchronization?
  std::cerr << "actual:\n" << actual << std::endl;
  double tol = 1e-5;
  if (at::allclose(expected, actual, tol, tol)) {
    std::cerr << "PASS!" << std::endl;
  } else {
    std::cerr << "FAIL!" << std::endl;
  }

  std::cerr << "bye" << std::endl;
  return 0;
}
