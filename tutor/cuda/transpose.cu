// follow https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

#define checkCuda(operation) { \
  cudaError_t result = operation; \
  if (result != cudaSuccess) { \
    fprintf(stderr, "%s:%d: CUDA runtime error: %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
    exit(1); \
  } \
}

void postprocess(const float *ref, const float *res, int n, float ms) {
  bool passed = true;
  for (int i = 0; i < n; i++) {
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  }
  if (passed) {
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
  }
}

__global__ void copy(float *odata, const float *idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y + j) * width + x] = idata[(y + j) * width + x];
  }
}

// ? Why this is a bit faster than copy on a100?
__global__ void copySharedMem(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM * TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];
  }

  __syncthreads(); // technically not needed. Keep to minic the kernel for transpose
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
  }
}

__global__ void transposeNaive(float *odata, const float *idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[x * width + (y + j)] = idata[(y + j) * width + x];
  }
}

// Tile width == #banks causes shared memory bank conflicts
__global__ void transposeCoalesced(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
  }

  __syncthreads();
  int tx = blockIdx.y * TILE_DIM + threadIdx.x;
  int ty = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(ty + j) * width + tx] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transposeNoBankConflicts(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
  }

  __syncthreads();
  int tx = blockIdx.y * TILE_DIM + threadIdx.x;
  int ty = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(ty + j) * width + tx] = tile[threadIdx.x][threadIdx.y + j];
  }
}

int main(int argc, char **argv) {
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx * ny * sizeof(float);

  dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  if (argc > 1) {
    devId = atoi(argv[1]);
  }

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("Device: %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
      nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
      dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  checkCuda(cudaSetDevice(devId));

  float *h_idata = (float *) malloc(mem_size);
  float *h_cdata = (float *) malloc(mem_size);
  float *h_tdata = (float *) malloc(mem_size);
  float *gold = (float *) malloc(mem_size);

  float *d_idata, *d_cdata, *d_tdata;
  checkCuda(cudaMalloc(&d_idata, mem_size));
  checkCuda(cudaMalloc(&d_cdata, mem_size));
  checkCuda(cudaMalloc(&d_tdata, mem_size));

  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    exit(1);
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    exit(1);
  }


  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      h_idata[j * nx + i] = j * nx + i;
    }
  }

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      gold[j * nx + i] = h_idata[i * nx + j];
    }
  }

  checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  float ms;

  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
  printf("%25s", "copy");
  checkCuda(cudaMemset(d_cdata, 0, mem_size));
  // warm up
  copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);

  // According to chatgpt: cuda event is a marker you can put in a stream of cuda operations
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < NUM_REPS; i++) {
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(h_idata, h_cdata, nx * ny, ms);

  printf("%25s", "shared memory copy");
  checkCuda(cudaMemset(d_cdata, 0, mem_size));
  // warm up
  copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < NUM_REPS; i++) {
    copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(h_idata, h_cdata, nx * ny, ms);

  printf("%25s", "naive transpose");
  checkCuda(cudaMemset(d_tdata, 0, mem_size));
  // warm up
  transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < NUM_REPS; i++) {
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  printf("%25s", "coalesced transpose");
  checkCuda(cudaMemset(d_tdata, 0, mem_size));
  // warm up
  transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < NUM_REPS; i++) {
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  printf("%25s", "conflict-free transpose");
  checkCuda(cudaMemset(d_tdata, 0, mem_size));
  // warm up
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < NUM_REPS; i++) {
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  checkCuda(cudaFree(d_idata));
  checkCuda(cudaFree(d_cdata));
  checkCuda(cudaFree(d_tdata));

  free(h_idata);
  free(h_cdata);
  free(h_tdata);
  free(gold);
  return 0;
}
