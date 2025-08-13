#include <iostream>
#include <fstream>
#include <cassert>
#include "cuda.h"
#include "cuda_bf16.h"

using bfloat16 = __nv_bfloat16;

static void cuda_check(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return;
  const char *errstr;
  cuGetErrorString(code, &errstr);
  fprintf(stderr, "cuda call fail %s:%d: code %d, errstr %s\n", file, line, code, errstr);
  abort();
}

#define CUDA_CHECK(ans) cuda_check(ans, __FILE__, __LINE__)

std::string loadCubinFile(const char *path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(
    std::istreambuf_iterator<char>(input),
    std::istreambuf_iterator<char>());
}

#define N 1024

void allocate_memory(bfloat16* dptrs[3]) {
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cuMemAlloc((CUdeviceptr*) &dptrs[i], N * sizeof(bfloat16)));
  }
  // initialize inputs
  bfloat16* hptr = (bfloat16 *) malloc(N * sizeof(bfloat16));
  assert(hptr);
  for (int i = 0; i < N; ++i) {
    hptr[i] = 1.0;
  }
  CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr) dptrs[0], hptr, N * sizeof(bfloat16)));
  for (int i = 0; i < N; ++i) {
    hptr[i] = 2.0;
  }
  CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr) dptrs[1], hptr, N * sizeof(bfloat16)));
  free(hptr);
}

void validate_result(bfloat16 *outptr) {
  bfloat16* hptr = (bfloat16 *) malloc(N * sizeof(bfloat16));
  assert(hptr);
  CUDA_CHECK(cuMemcpyDtoH(hptr, (CUdeviceptr) outptr, N * sizeof(bfloat16)));
  int nfail = 0;
  for (int i = 0; i < N; ++i) {
    if (hptr[i] != 3.0) {
      ++nfail;
    }
  }
  free(hptr);

  if (nfail > 0) {
    std::cerr << "Incorrect result" << std::endl;
    abort();
  } else {
    std::cerr << "Pass the numeric check!" << std::endl;
  }
}

void release_memory(bfloat16* dptrs[3]) {
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cuMemFree((CUdeviceptr)dptrs[i]));
  }
}

int main(void) {
  const char *cubin_path = "add.cubin";
  const char *func_name = "add";
  int device = 0;
  CUstream stream = 0;

  CUfunction fun;
  CUmodule mod;
  std::string cubindata = loadCubinFile(cubin_path);

  CUDA_CHECK(cuInit(0));

  CUcontext ctx = 0;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
  }
  assert(ctx);

  std::cout << "cubin file size " << cubindata.size() << " bytes" << std::endl;

  CUDA_CHECK(cuModuleLoadData(&mod, cubindata.c_str()));
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, func_name));

  bfloat16* dptrs[3] = {nullptr};
  allocate_memory(dptrs);
  { // launch
    int gridX = 32, gridY = 1, gridZ = 1;
    int num_warps = 4;
    int numel = N;
    void *params[] = {&dptrs[0], &dptrs[1], &dptrs[2], &numel};
    int shared = 0;
    CUDA_CHECK(cuLaunchKernel(fun, gridX, gridY, gridZ, 32 * num_warps, 1, 1, shared, stream, params, 0));
  }
  validate_result(dptrs[2]);
  release_memory(dptrs);

  std::cout << "bye" << std::endl;
  return 0;
}
