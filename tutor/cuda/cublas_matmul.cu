#include </usr/local/cuda-12.9/include/cublas_v2.h>
#include <cstdio>
#include <cassert>
#include <cmath>

// CPU reference implementation for matrix multiply (column-major)
// C = A * B where A is m×k, B is k×n, C is m×n
void cpu_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                // Column-major: A[row, i] = A[i * m + row], B[i, col] = B[col * k + i]
                sum += A[i * m + row] * B[col * k + i];
            }
            C[col * m + row] = sum;
        }
    }
}

// Compare two matrices with tolerance
bool verify_results(const float* gpu, const float* cpu, int size, float tol = 1e-5f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(gpu[i] - cpu[i]) > tol) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

#define CUBLAS_CHECK(EXPR) { \
  cublasStatus_t status = EXPR; \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    assert(false && "cublass call fail: " # EXPR); \
  } \
}

int main() {
    // Matrix sizes
    const int m = 2;
    const int n = 3;
    const int k = 4;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Host matrices (column-major!)
    float h_A[m * k] = {
        1, 2,   // column 0
        3, 4,   // column 1
        5, 6,   // column 2
        7, 8    // column 3
    }; // 2x4

    float h_B[k * n] = {
        1,  2,  3,  4,   // column 0
        5,  6,  7,  8,   // column 1
        9, 10, 11, 12    // column 2
    }; // 4x3

    float h_C[m * n] = {0}; // 2x3

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // C = alpha * A * B + beta * C
    // All matrices are column-major
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // A, B not transposed
        m, n, k,
        &alpha,
        d_A, m,   // lda = m
        d_B, k,   // ldb = k
        &beta,
        d_C, m    // ldc = m
    ));

    // Copy result back
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    float h_C_ref[m * n] = {0};
    cpu_matmul(h_A, h_B, h_C_ref, m, n, k);

    // Print GPU result (column-major)
    printf("GPU Result (C):\n");
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            printf("%6.1f ", h_C[col * m + row]);
        }
        printf("\n");
    }

    // Print CPU reference (column-major)
    printf("\nCPU Reference:\n");
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            printf("%6.1f ", h_C_ref[col * m + row]);
        }
        printf("\n");
    }

    // Verify results
    if (verify_results(h_C, h_C_ref, m * n)) {
        printf("\nVERIFICATION PASSED!\n");
    } else {
        printf("\nVERIFICATION FAILED!\n");
        return 1;
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

