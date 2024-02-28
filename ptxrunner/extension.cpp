#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cassert>
#include <iostream>
#include "cuda.h"
#include <vector>

void cuda_check(CUresult code, const char* file, int lineno) {
  if (code == CUDA_SUCCESS) {
    return;
  }
  const char* err_str;
  cuGetErrorString(code, &err_str);
  std::cerr << "Error from CUDA: " << err_str << std::endl;
  std::cerr << "  at " << file << ":" << lineno << std::endl;
  assert(false);
}

#define CUDA_CHECK(call) \
  do { \
    cuda_check(call, __FILE__, __LINE__); \
  } while (0)

PyObject *load_cufunc_from_bytes(PyObject *self, PyObject *args) {
  const char *data;
  Py_ssize_t data_size;
  const char* func_name;
  if (!PyArg_ParseTuple(args, "s#s", &data, &data_size, &func_name)) {
    return NULL;
  }
  CUfunction cu_func = nullptr;

  CUcontext ctx = nullptr;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  assert(ctx);

  CUmodule cu_mod;
  CUDA_CHECK(cuModuleLoadData(&cu_mod, data));
  CUDA_CHECK(cuModuleGetFunction(&cu_func, cu_mod, func_name));
  return PyLong_FromUnsignedLongLong((uint64_t) cu_func);
}

// caller is responsible to free the returned pointer.
std::vector<uint64_t> pylist_to_uint64_vec(PyObject *list_obj) {
  Py_ssize_t len = PyList_Size(list_obj);
  std::vector<uint64_t> vec;
  vec.reserve(len);
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyList_GetItem(list_obj, i);
    vec.push_back(PyLong_AsUnsignedLongLong(item));
  }
  return vec;
}

PyObject *launch(PyObject *self, PyObject *args) {
  CUfunction cu_func;
  PyObject *gridDimObj, *blockDimObj, *argsObj;
  int shared;

  if (!PyArg_ParseTuple(args, "KO!O!O!i", &cu_func, &PyList_Type, &gridDimObj,
      &PyList_Type, &blockDimObj, &PyList_Type, &argsObj, &shared)) {
    return nullptr;
  }

  std::vector<uint64_t> gridDim = pylist_to_uint64_vec(gridDimObj);
  assert(gridDim.size() == 3);
  std::vector<uint64_t> blockDim = pylist_to_uint64_vec(blockDimObj);
  assert(blockDim.size() == 3);
  std::vector<uint64_t> argsVal = pylist_to_uint64_vec(argsObj);
  std::vector<void*> argsPtr(argsVal.size());
  for (int i = 0; i < argsVal.size(); ++i) {
    argsPtr[i] = &argsVal[i];
  }

  CUstream stream = nullptr;
  CUDA_CHECK(cuLaunchKernel(cu_func, gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2], shared, stream, argsPtr.data(), 0));

  Py_RETURN_NONE;
}

PyMethodDef module_methods[] = {
  {"load_cufunc_from_bytes", load_cufunc_from_bytes, METH_VARARGS, "Load CUfunction from the cubin bytes"},
  {"launch", launch, METH_VARARGS, "Launch a kernel"},
  {NULL, NULL, 0, NULL}
};

struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT, "_C", NULL, -1, module_methods
};

PyMODINIT_FUNC PyInit__C(void) {
  PyObject *m = PyModule_Create(&module_def);
  if (!m) {
    return m;
  }
  PyModule_AddFunctions(m, module_methods);
  return m;
}
