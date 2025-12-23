#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>

#define NSTEP 1000
#define NKERNEL 20
#define N 500000

__global__ void short_kernel(float *out_ptr, float *in_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out_ptr[idx] = in_ptr[idx] * 2;
  }
}

void run_experiment_sync_after_kernel(float *d_out, float *d_in, cudaStream_t stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NSTEP; ++i) {
    for (int j = 0; j < NKERNEL; ++j) {
      short_kernel<<<blocks, threads, 0, stream>>>(d_out, d_in);
      cudaStreamSynchronize(stream);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "SyncAfterKenrel: " << elapse.count() << " ms" << std::endl;
}

void run_experiment_sync_after_step(float *d_out, float *d_in, cudaStream_t stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NSTEP; ++i) {
    for (int j = 0; j < NKERNEL; ++j) {
      short_kernel<<<blocks, threads, 0, stream>>>(d_out, d_in);
    }
    cudaStreamSynchronize(stream);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "SyncAfterStep: " << elapse.count() << " ms" << std::endl;
}

/*
 * Dot file generated for this cudagraph: https://gist.github.com/shunting314/1b7a70487687124b62cc7a1782d56689
 * Png file: https://github.com/zhang314159265/gws/blob/0ccab46252e1b95ab47491fb378744111504565d/cuda_graph.png
 */
void run_experiment_cuda_graphs(float *d_out, float *d_in, cudaStream_t stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  auto start = std::chrono::high_resolution_clock::now();

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  for (int i = 0; i < NSTEP; ++i) {
    if (i == 0) { // do graph capture
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
      for (int j = 0; j < NKERNEL; ++j) {
        short_kernel<<<blocks, threads, 0, stream>>>(d_out, d_in);
      }
      cudaStreamEndCapture(stream, &graph);
      const char *path = "/tmp/cuda_graph.dot";
      cudaGraphDebugDotPrint(graph, path, cudaGraphDebugDotFlagsVerbose);
      std::cout << "Cudagraph debug dot file is written to " << path << std::endl;
      cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "CudaGraphs: " << elapse.count() << " ms" << std::endl;
}

#ifdef DEF_MAIN
int main(void) {
#else
int run(void) {
#endif
  float *h_out, *h_in;
  float *d_out, *d_in;
  h_out = (float *) malloc(N * sizeof(*h_out));
  h_in = (float *) malloc(N * sizeof(*h_in));
  cudaMalloc(&d_out, N * sizeof(*d_out));
  cudaMalloc(&d_in, N * sizeof(*d_in));
  for (int i = 0; i < N; ++i) {
    h_in[i] = 1.0;
  }
  cudaMemcpy(d_in, h_in, N * sizeof(*h_in), cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  run_experiment_sync_after_kernel(d_out, d_in, stream);
  run_experiment_sync_after_step(d_out, d_in, stream);
  run_experiment_cuda_graphs(d_out, d_in, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaMemcpy(h_out, d_out, N * sizeof(*h_out), cudaMemcpyDeviceToHost);
  float sum = 0.0;
  for (int i = 0; i < N; ++i) {
    sum += h_out[i];
  }
  printf("sum is %f\n", sum);
  assert(sum == N * 2.0);
  printf("Pass the check!\n");

  free(h_out);
  free(h_in);
  cudaFree(d_out);
  cudaFree(d_in);
  return 0;
}
