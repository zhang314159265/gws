#pragma once

#include "cuda.h"

namespace tritoncc {

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
    tritoncc::cudaCheck(ans, __FILE__, __LINE__); \
  } while(0)

static CUfunction loadCUDAFunctionFromFile(const char* path, const char* func_name, int shared) {
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
    std::ifstream input(path, std::ios::binary);
    cubinBytes = std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()
    );
  }
  CUmodule mod;
  CUfunction fun;

  CUDA_CHECK(cuModuleLoadData(&mod, cubinBytes.c_str()));
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, func_name));

  int n_regs = 0;
  int n_spills = 0;
  CUDA_CHECK(cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK(cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  std::cerr << "n_regs " << n_regs << ", n_spills " << n_spills << std::endl;

  assert(shared <= 49152);
  return fun;
}

}
