#pragma once

void cudaCheck(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA runtime error: %s\n", file, line, cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  }
}

#define cudaCheck(op) cudaCheck((op), __FILE__, __LINE__)
