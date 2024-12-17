#include <stdio.h>
#include <chrono>
#include <cuda_bf16.h>

#define WARP_SIZE 32

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}



typedef __nv_bfloat16 floatX;
typedef Packed128<floatX> x128;

// warp-level reduction for summing values
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
// warp-level reduction for finding the maximum value
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*) (float);
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}



// This kernel is mostly copied from llm.c and then slightly updated.
__global__ void prepare_softmax_kernel(__nv_bfloat16 *inp, __nv_bfloat16 *scale, __nv_bfloat16 *offset, int BT, int V, int Vp
) {
  int64_t idx = blockIdx.x;
  int P = Vp;

  const floatX* x = inp + idx * P;
  float thread_maxval = -INFINITY;
  float thread_sumval = 0.0f;
  int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

  // special-case loop to handle the unaligned elements at the end of the array
  // this lets us skip the bounds check in the main loop below, which improves performance
  while ((i+1)*x128::size > V) {
    for(int k = 0; k < x128::size; ++k) {
      if (i*x128::size+k >= V) {
        break; // bounds checking against real V (rather than padded P)
      }
      float v = (float)x[i*x128::size+k];
      float old_maxval = thread_maxval;
      thread_maxval = fmaxf(thread_maxval, v);
      thread_sumval *= expf((old_maxval - thread_maxval));
      thread_sumval += expf(v - thread_maxval);
    }
    i -= blockDim.x;
  }

  // main loop for the bulk of the iterations (no bounds checking required!)
  for (; i >= 0; i -= blockDim.x) {
    x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
    for(int k = 0; k < x128::size; ++k) {
      float v = (float)packed_x[k];
      float old_maxval = thread_maxval;
      thread_maxval = fmaxf(thread_maxval, v);
      thread_sumval *= expf((old_maxval - thread_maxval));
      thread_sumval += expf(v - thread_maxval);
    }
  }

  // Block Max Reduction -> Maths -> Block Sum Reduction
  float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
  thread_sumval *= expf(thread_maxval - block_maxval);
  float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

  // only thread0 do the write
  if (threadIdx.x == 0) {
  #if 0
    scale[idx] = 1.f / block_sumval;
  #else
    scale[idx] = logf(block_sumval);
  #endif
    offset[idx] = block_maxval;
  }
}

void prepare_softmax(__nv_bfloat16 *inp, __nv_bfloat16 *scale, __nv_bfloat16 *offset, int BT, int V, int Vp) {
  const int block_size = 1024;
  const int grid_size = BT;
  prepare_softmax_kernel<<<grid_size, block_size>>>(inp, scale, offset, BT, V, Vp);
}
   
using namespace std::chrono;

int main(void) {
  int BT = 32768;
  int V = 50257;
  int Vp = 50304;

  __nv_bfloat16 *inp, *scale, *offset; 
  cudaMalloc(&inp, BT * Vp * sizeof(__nv_bfloat16));
  cudaMalloc(&scale, BT * sizeof(float));
  cudaMalloc(&offset, BT * sizeof(float));

  for (int i = 0; i < 5; ++i) {
    // reset input and clear L2 cache
    cudaMemset(inp, 0x01, BT * Vp * sizeof(__nv_bfloat16));

    // Run prepare_softmax kernel
    cudaDeviceSynchronize();


    auto start = high_resolution_clock::now();
    prepare_softmax(inp, scale, offset, BT, V, Vp);
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    auto elapse_us = duration_cast<microseconds>(end - start);
    printf("latency %.3fms\n", (elapse_us.count() + 0.0) / 1000);
  }

  cudaDeviceSynchronize();

  printf("Bye\n");
  return 0;
}
