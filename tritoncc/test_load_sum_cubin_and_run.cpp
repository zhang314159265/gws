#include "cuda.h"
#include <iostream>
#include <fstream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <unistd.h>

void cudaCheck(CUresult code, const char* file, int line) {
  if (code == CUDA_SUCCESS) {
    return;
  }
  const char *str;
  cuGetErrorString(code, &str);
  std::cerr << "Got error from cuda: " << str << std::endl;
  assert(false);
}

#define CUDA_CHECK(ans) \
  do { \
    cudaCheck(ans, __FILE__, __LINE__); \
  } while(0)

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Require cubin path" << std::endl;
    exit(1);
  }
  torch::Tensor x = torch::randn({1024, 1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor expected = x.sum(-1);
  std::cerr << "expected:\n" << expected << std::endl;
  torch::Tensor actual = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  int device = 0;
  CUcontext pctx = 0;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
  std::cerr << "pctx " << pctx << std::endl;

  std::string cubinBytes;
  {
    std::ifstream input(argv[1], std::ios::binary);
    cubinBytes = std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()
    );
  }
  CUmodule mod;
  CUfunction fun;
  const char* name = "sum_fn";
  if (argc >= 3) {
    name = argv[2];
  }
  CUDA_CHECK(cuModuleLoadData(&mod, cubinBytes.c_str()));
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));

  int n_regs = 0;
  int n_spills = 0;
  CUDA_CHECK(cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK(cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  std::cerr << "n_regs " << n_regs << ", n_spills " << n_spills << std::endl;

  int shared = 2064; // TODO: avoid hardcode
  assert(shared <= 49152);

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
