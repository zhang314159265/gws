#include "cuda.h"
#include <iostream>
#include <fstream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <unistd.h>
#include "tritoncc/runtime.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Require cubin path" << std::endl;
    exit(1);
  }
  torch::Tensor x = torch::randn({1024, 1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor expected = x.sum(-1);
  torch::Tensor actual = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  const char* name = "sum_fn";
  if (argc >= 3) {
    name = argv[2];
  }
  int shared = 2064;
  CUfunction fun = tritoncc::loadCUDAFunctionFromFile(argv[1], name);
  { // launch kernel
    int gridX = 512; // TODO don't hard code
    int gridY = 1;
    int gridZ = 1;
    int num_warps = 4;
    CUstream stream = 0;
    void* data_ptr_in = x.data_ptr();
    void* data_ptr_out = actual.data_ptr();
    int xnumel = 1024, rnumel = 1024;
    void* params[] = {&data_ptr_in, &data_ptr_out, &xnumel, &rnumel};
    CUDA_CHECK(cuLaunchKernel(fun, gridX, gridY, gridZ, 32 * num_warps, 1, 1, shared, stream, params, 0));
  }
  cuStreamSynchronize(0);
  double tol = 1e-5;
  if (at::allclose(expected, actual, tol, tol)) {
    std::cerr << "PASS!" << std::endl;
  } else {
    if (getenv("PRINT_TENSOR")) {
      std::cerr << "expected:\n" << expected << std::endl;
      std::cerr << "actual:\n" << actual << std::endl;
    }
    std::cerr << "FAIL!" << std::endl;
  }

  std::cerr << "bye" << std::endl;
  return 0;
}
