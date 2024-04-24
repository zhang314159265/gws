#pragma once

#define DEBUG_RUNTIME 0

namespace tritoncc {

void cudaCheck(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS) {
    return;
  }
  const char *str;
  cuGetErrorString(code, &str);
  std::cerr << file << ":" << line << ": Got error from cuda: " << str << std::endl;
  assert(false);
}

#define CUDA_CHECK(call) \
  do { \
    tritoncc::cudaCheck(call, __FILE__, __LINE__); \
  } while (0)

CUfunction loadCUDAFunctionFromFile(const char *path, const char *func_name) {
  int device = 0;
  CUcontext ctx = 0;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
  }

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

  #if DEBUG_RUNTIME
  int n_regs = 0;
  int n_spills = 0;
  CUDA_CHECK(cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK(cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  std::cerr << "n_regs " << n_regs << ", n_spills " << n_spills << std::endl;
  #endif
  return fun;
}

}
