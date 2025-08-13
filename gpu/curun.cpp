#include "cuda.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <iostream>

void cuda_check(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return;
  const char *errstr;
  cuGetErrorString(code, &errstr);
  fprintf(stderr, "cuda driver API fail at %s:%d: %s\n", file, line, errstr);
  abort();
}

#define CUDA_CHECK(ans) cuda_check(ans, __FILE__, __LINE__)

std::string loadCubinFile(const char *path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(
    std::istreambuf_iterator<char>(input),
    std::istreambuf_iterator<char>());
}

// return long rather than CUmodule so pybind don't need to understand
// CUmodule
long openCubin(const char *path) {
  int device = 0;

  CUDA_CHECK(cuInit(0));

  CUcontext ctx = 0;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
  }
  assert(ctx);

  std::string cubinData = loadCubinFile(path);

  CUmodule mod;
  CUDA_CHECK(cuModuleLoadData(&mod, cubinData.c_str()));
  return (long) mod;
}

long findSym(long _mod, const char *name) {
  CUmodule mod = (CUmodule) _mod;
  CUfunction fun;
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));
  return (long) fun;
}

void runKernel(long _func, int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int shared, long _stream, std::vector<long> args) {
  CUfunction func = (CUfunction) _func;
  CUstream stream = (CUstream) _stream;

  std::vector<void*> argsPtr(args.size());
  for (int i = 0; i < args.size(); ++i) {
    argsPtr[i] = &args[i];
  }

  CUDA_CHECK(cuLaunchKernel(
    func,
    gridX, gridY, gridZ,
    blockX, blockY, blockZ,
    shared,
    stream,
    argsPtr.data(),
    0
  ));
}

PYBIND11_MODULE(curun, m) {
  m.doc() = "Load and run cuda kernel from a cubin file";
  m.def("open", openCubin);
  m.def("sym", findSym);
  m.def("run", runKernel);
}
