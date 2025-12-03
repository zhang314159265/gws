__device__ float _sumrow(float * __restrict__ rowptr, int N) {
  int num_warps = blockDim.x / 32;
  float accum = 0.0;
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  assert(N % 4 == 0);
  for (int i = threadIdx.x * 4; i < N; i += blockDim.x * 4) {
    int4 vec = __ldg((int4*) (&rowptr[i]));
    float *sub = (float *) &vec;
    for (int j = 0; j < 4; ++j) {
      accum += (float) sub[j];
    }
  }

  // warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  if (blockDim.x <= 32) {
    return accum;
  }

  // multi-warp reduction
  #if 1
  extern __shared__ float buffer[];
  #else
  __shared__ float buffer[32];
  #endif
  if (laneId == 0) {
    buffer[warpId] = accum;
  }
  __syncthreads();

  if (warpId == 0) {
    if (laneId < num_warps) {
      accum = buffer[laneId];
    } else {
      accum = 0;
    }
    // warp shuffle
    for (int mask = 1; mask < num_warps; mask *= 2) {
      accum += __shfl_xor_sync(0xffffffff, accum, mask);
    }
    if (laneId == 0) {
      buffer[0] = accum;
    }
  }

  __syncthreads();
  return buffer[0];
}

extern "C" __global__ void sum_kernel(float* x, float* y, int M, int N) {
  int rowIdx = blockIdx.x;
  float *rowptr = x + rowIdx * N;
  float *outptr = y + rowIdx * N;
  int num_warps = blockDim.x / 32;
  assert(num_warps > 0 && ((num_warps & (num_warps - 1)) == 0));

  float sum = _sumrow(rowptr, N);
  for (int i = threadIdx.x * 4; i < N; i += blockDim.x * 4) {
    int4 vec = __ldg((int4*) (&rowptr[i]));
    float *sub = (float *) &vec;
    float out4[4];
    for (int j = 0; j < 4; ++j) {
      out4[j] = sub[j] + sum;
    }
    *(int4*) (outptr + i) = *(int4 *) out4;
  }
}
