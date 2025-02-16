#include <stdlib.h>
#include <stdio.h>
#include <cublasLt.h>
#include "../cuda/common.h"

const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void *cublaslt_workspace = NULL;
cublasLtHandle_t cublaslt_handle;

void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLASS ERROR]: %d %s %d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

template <typename T>
cudaDataType_t getCudaDataType() {
  assert(false);
}

template <>
cudaDataType_t getCudaDataType<float>() {
  return CUDA_R_32F;
}

template <typename T>
void matmul_cublaslt(T *d, const T *a, const T *b,
  int m, int n, int k, bool transA, bool transB) {

  cudaDataType_t datatype = getCudaDataType<T>();

  if (!cublaslt_workspace) {
    // lazy init the workspace & cublas handle
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
  }

  cublasLtMatmulDesc_t operationDesc;
  cublasCheck(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  int returnedResults = 0;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;

  cublasOperation_t opNoTranspose = CUBLAS_OP_N;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));

  cublasLtMatrixLayout_t ALayout;
  cublasLtMatrixLayout_t BLayout;
  cublasLtMatrixLayout_t CLayout;
  cublasLtMatrixLayout_t DLayout;
  if (transA) {
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, datatype, k, m, k));
  } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, datatype, m, k, m));
  }
  if (transB) {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, datatype, n, k, n));
  } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, datatype, k, n, k));
  }
  cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, datatype, m, n, m));
  cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, datatype, m, n, m));

  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // TODO support bias & gelu
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  cublasDataType_t scale_type = CUDA_R_32F;
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

  cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
      preference, 1, &heuristic, &returnedResults);
  if (returnedResults == 0) {
    printf("No cuBLASLt algorithm\n");
    exit(EXIT_FAILURE);
  }
  const float alpha = 1.0f, beta = 0.0f;

  cudaStream_t stream = 0; // TODO
  cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
      &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
      &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

  // cleanup
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
  cudaCheck(cudaGetLastError());
}
