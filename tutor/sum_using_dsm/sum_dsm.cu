#include "cooperative_groups.h"
#include "cuda_bf16.h"

using bfloat16 = __nv_bfloat16;

namespace cg = cooperative_groups;
extern __shared__ char _buffer[];


/// begin kernels for float
__device__ float _sumrow(int N) {
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();
  int cluster_size = cluster.size() / blockDim.x;

  #if 0
  if (cluster_rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    printf("cluster_size %d\n", cluster_size);
  }
  #endif

  int num_warps = blockDim.x / 32;
  float accum = 0.0;
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  float *buffer = (float *) _buffer;
  for (int i = threadIdx.x; i < N / 8; i += blockDim.x) {
    float val = buffer[i];
    accum += val;
  }

  // warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  float *redbuf = buffer + (N / 8);

  // block shuffle
  if (num_warps > 1) {
    // multi-warp reduction
    if (laneId == 0) {
      redbuf[warpId] = accum;
    }
    __syncthreads();
  
    if (warpId == 0) {
      if (laneId < num_warps) {
        accum = redbuf[laneId];
      } else {
        accum = 0;
      }
      for (int mask = 1; mask < num_warps; mask *= 2) {
        accum += __shfl_xor_sync(0xffffffff, accum, mask);
      }
      if (laneId == 0) {
        redbuf[0] = accum;
      }
    }
  
    __syncthreads();
  }

  // block cluster shuffle
  if (cluster_size > 1) {
    cluster.sync();

    if (cluster_rank != 0) {
      float *rank0addr = cluster.map_shared_rank(&redbuf[cluster_rank], 0);
      *rank0addr = redbuf[0];
    }
    cluster.sync();
    if (cluster_rank == 0 && warpId == 0) {
      if (laneId < cluster_size) {
        accum = redbuf[laneId];
      } else {
        accum = 0;
      }
      for (int mask = 1; mask < cluster_size; mask *= 2) {
        accum += __shfl_xor_sync(0xffffffff, accum, mask);
      }
      if (laneId == 0) {
        redbuf[0] = accum;
      }
    }
    cluster.sync();
    if (cluster_rank != 0) {
      float *rank0addr = cluster.map_shared_rank(&redbuf[0], 0);
      redbuf[0] = *rank0addr;
    }
    cluster.sync();
  }

  return redbuf[0];
}

__device__ void fullfill_sm(float *rowptr, int N) {
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();
  assert(N % 8 == 0);

  rowptr += (N / 8) * cluster_rank;
  assert(N % 32 == 0);
  float *buffer = (float *) _buffer;
  for (int i = threadIdx.x * 4; i < N / 8; i += blockDim.x * 4) {
    float4 vec = __ldg((float4*) &rowptr[i]);
    float *sub = (float *) &vec;
    for (int j = 0; j < 4; ++j) {
      buffer[i + j] = sub[j];
    }
  }

  cluster.sync();
}

__device__ void sum_dsm_kernel_float(float *x, float *y, int M, int N) {
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();

  int num_warps = blockDim.x / 32;
  assert(num_warps > 0 && ((num_warps & (num_warps - 1)) == 0));

  int rowIdx = blockIdx.x / 8;
  assert(rowIdx < M);

  float *rowptr = x + rowIdx * N;
  fullfill_sm(rowptr, N);
  float sum = _sumrow(N);

  y += rowIdx * N + (N / 8) * cluster_rank;

  float *buffer = (float *) _buffer;
  assert(N % 32 == 0);
  for (int i = threadIdx.x * 4; i < N / 8; i += blockDim.x * 4) {
    float out4[4];
    for (int j = 0; j < 4; ++j) {
      out4[j] = buffer[i + j] + sum;
    }
    *(int4*) (y + i) = *(int4 *) out4;
  }
}

// begin kernels for bfloat16

__device__ float _sumrow_bf16(int N) {
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();
  int cluster_size = cluster.size() / blockDim.x;

  #if 0
  if (cluster_rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    printf("cluster_size %d\n", cluster_size);
  }
  #endif

  int num_warps = blockDim.x / 32;
  float accum = 0.0;
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  bfloat16 *buffer = (bfloat16 *) _buffer;
  for (int i = threadIdx.x; i < N / 8; i += blockDim.x) {
    float val = buffer[i];
    accum += val;
  }

  // warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  float *redbuf = (float *) (buffer + (N / 8));

  // block shuffle
  if (num_warps > 1) {
    // multi-warp reduction
    if (laneId == 0) {
      redbuf[warpId] = accum;
    }
    __syncthreads();
  
    if (warpId == 0) {
      if (laneId < num_warps) {
        accum = redbuf[laneId];
      } else {
        accum = 0;
      }
      for (int mask = 1; mask < num_warps; mask *= 2) {
        accum += __shfl_xor_sync(0xffffffff, accum, mask);
      }
      if (laneId == 0) {
        redbuf[0] = accum;
      }
    }
  
    __syncthreads();
  }

  // block cluster shuffle
  if (cluster_size > 1) {
    cluster.sync();

    if (cluster_rank != 0) {
      float *rank0addr = cluster.map_shared_rank(&redbuf[cluster_rank], 0);
      *rank0addr = redbuf[0];
    }
    cluster.sync();
    if (cluster_rank == 0 && warpId == 0) {
      if (laneId < cluster_size) {
        accum = redbuf[laneId];
      } else {
        accum = 0;
      }
      for (int mask = 1; mask < cluster_size; mask *= 2) {
        accum += __shfl_xor_sync(0xffffffff, accum, mask);
      }
      if (laneId == 0) {
        redbuf[0] = accum;
      }
    }
    cluster.sync();
    if (cluster_rank != 0) {
      float *rank0addr = cluster.map_shared_rank(&redbuf[0], 0);
      redbuf[0] = *rank0addr;
    }
    cluster.sync();
  }

  return redbuf[0];
}

__device__ void fullfill_sm_bf16(bfloat16 *rowptr, int N) {
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();
  assert(N % 8 == 0);

  rowptr += (N / 8) * cluster_rank;
  assert(N % 64 == 0);
  bfloat16 *buffer = (bfloat16 *) _buffer;
  for (int i = threadIdx.x * 8; i < N / 8; i += blockDim.x * 8) {
    int4 vec = __ldg((int4*) &rowptr[i]);
    bfloat16 *sub = (bfloat16 *) &vec;
    for (int j = 0; j < 8; ++j) {
      buffer[i + j] = sub[j];
    }
  }

  cluster.sync();
}

__device__ void sum_dsm_kernel_bf16(bfloat16 *x, bfloat16 *y, int M, int N) {
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_rank = cluster.block_rank();

  int num_warps = blockDim.x / 32;
  assert(num_warps > 0 && ((num_warps & (num_warps - 1)) == 0));

  int rowIdx = blockIdx.x / 8;
  assert(rowIdx < M);

  bfloat16 *rowptr = x + rowIdx * N;
  fullfill_sm_bf16(rowptr, N);
  float sum = _sumrow_bf16(N);

  y += rowIdx * N + (N / 8) * cluster_rank;

  bfloat16 *buffer = (bfloat16 *) _buffer;
  assert(N % 64 == 0);
  for (int i = threadIdx.x * 8; i < N / 8; i += blockDim.x * 8) {
    bfloat16 out4[8];
    for (int j = 0; j < 8; ++j) {
      out4[j] = (bfloat16) ((float) buffer[i + j] + sum);
    }
    *(int4*) (y + i) = *(int4 *) out4;
  }
}

extern "C" __global__ void
__cluster_dims__(8, 1, 1)
sum_dsm_kernel(void* x, void* y, int M, int N, bool is_float) {
  if (is_float) {
    sum_dsm_kernel_float((float *) x, (float *) y, M, N);
  } else {
    sum_dsm_kernel_bf16((bfloat16*) x, (bfloat16*) y, M, N);
  }
}

