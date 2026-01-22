// follows cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu
// Compile command need specify '-arch=sm_100a'

#include <iostream>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;

template <
  class AccType,
  class TensorA, class TensorB,
  class TensorC, class TensorD,
  class Alpha, class Beta
>
void reference_gemm(const TensorA &tensor_A, const TensorB &tensor_B,
    const TensorC& tensor_C, TensorD &tensor_D,
    Alpha alpha, Beta beta) {
  for (int m = 0; m < size<0>(tensor_D); ++m) {
    for (int n = 0; n < size<1>(tensor_D); ++n) {
      AccType acc{};
      for (int k = 0; k < size<1>(tensor_A); ++k) {
        acc += tensor_A(m, k) * tensor_B(n, k);
      }
      tensor_D(m, n) = alpha * acc + beta * tensor_C(m, n);
    }
  }
}

// generate integral inputs to ease accuracy check
template <class Tensor>
void initialize_tensor(Tensor &tensor, cute::tuple<int, int> value_range = {-2, 2}) {
  using DataType = typename Tensor::element_type;
  auto [min, max] = value_range;
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = DataType((int)((max - min) * (rand() / double(RAND_MAX)) + min));
  }
}

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) ArrayEngine<TypeA, cosize_v<ASmemLayout>> A;
  alignas(128) ArrayEngine<TypeB, cosize_v<BSmemLayout>> B;
  alignas(16) uint64_t mma_barrier;
  alignas(16) uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

template <
  class SharedStorage,
  class ATensor, class BTensor, class CTensor, class DTensor,
  class MmaTiler_MNK, class TiledMMA,
  class Alpha, class Beta
>
__global__ void
gemm_device(
  ATensor mA, BTensor mB, CTensor mC, DTensor mD, 
  MmaTiler_MNK mma_tiler,
  TiledMMA tiled_mma,
  Alpha alpha, Beta beta
) {
  // step 1: the prologue
  auto mma_coord_vmnk = make_coord(0, blockIdx.x, blockIdx.y, _);
  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);
  auto gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  auto gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  auto gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  auto gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  // the smem tensors
  extern __shared__ char shared_memory[];
  SharedStorage& shared_storage = *(SharedStorage*) (shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  // XXX tXgY -> the partition pattern tX applied to tensor gY
  
  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  // MMA fragment allocation
  // XXX Allocate 'fragments' which are SMEM descriptors that serve as inputs
  // to cute::gemm operations
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);

  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
  }

  __syncthreads();
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  // barrier initialization
  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ 1);
    // XXX why use a explicit barrier? Does __syncthreads work instead?
  }
  int mma_barrier_phase_bit = 0;
  __syncthreads(); // make sure all threads observe barrier initialization

  // step 2: the main loop
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    cooperative_copy<128>(threadIdx.x, tCgA(_, _, _, k_tile), tCsA);
    cooperative_copy<128>(threadIdx.x, tCgB(_, _, _, k_tile), tCsB);

    __syncthreads();

    // tcgen05.mma instructions require single-thread execution
    // - only one warp performs the MMA-related loop operations
    // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
    // - no explicit elect_one_sync region is needed from the user
    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // ensure MMAs are completed, only then we can reuse the A and B SMEM.
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }


  // step 3: the epilogue
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
  Tensor tDrC = make_fragment_like(tDgC);
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));

  // Load TMEM -> RMEM
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  axpby(alpha, tDrAcc, beta, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    // for some reason, release lock before free
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeA, class LayoutA,
    class TypeB, class LayoutB,
    class TypeC, class LayoutC,
    class TypeD, class LayoutD,
    class Alpha, class Beta>
void gemm_host(TypeA const *device_ptr_A, LayoutA layout_A,
    TypeB const *device_ptr_B, LayoutB layout_B,
    TypeC const *device_ptr_C, LayoutC layout_C,
    TypeD *device_ptr_D, LayoutD layout_D,
    Alpha const alpha, Beta const beta) {
  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);

  auto Gemm_M = shape<0>(layout_A);
  auto Gemm_N = shape<0>(layout_B);
  auto Gemm_K = shape<1>(layout_A);

  TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<
      TypeA, TypeB, TypeC, // A/B/Acc types
      128, 256, // Mma M/N dimensions
      UMMA::Major::K, UMMA::Major::K>{}); // A/B layout
  print(tiled_mma);
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{}; // use 4 MMAs per MMA Tile K.
  auto mma_tiler = make_shape(bM, bN, bK);

  // In SM90, the MMAs are CTA-local and perform thread-level partitioning.
  // In SM100, the MMAs are Cluster-local and perform CTA-level partitioning.
  //   The MMA's partitioning yileds the CTA-local work.
  if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
    std::cerr << "The MMA shape should evenly divide the MMA tiler." << std::endl;
    return;
  }

  if (not evenly_divides(make_shape(Gemm_M, Gemm_N, Gemm_K), mma_tiler)) {
    std::cerr << "OOB accesses are not supported. MmaTiler_MNK should evenly divide ProblemShape_MNK." << std::endl;
    return;
  }
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  // apply swizzling
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
  print("sA_layout:\t"); print(sA_layout); print("\n");
  print("sB_layout:\t"); print(sB_layout); print("\n");

  using SMEMStorage = SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  dim3 dimBlock(128);
  dim3 dimGrid(ceil_div(Gemm_M, bM),
      ceil_div(Gemm_N, bN));
  int smemBytes = sizeof(SMEMStorage);

  auto *kernel_ptr = &gemm_device<SMEMStorage,
    decltype(mA), decltype(mB), decltype(mC), decltype(mD),
    decltype(mma_tiler), decltype(tiled_mma),
    Alpha, Beta>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smemBytes));
  kernel_ptr<<<
    dimGrid,
    dimBlock,
    smemBytes
  >>>(mA, mB, mC, mD, mma_tiler, tiled_mma, alpha, beta);
  CUTE_CHECK_LAST();
}

// copied from the first cute blackwell gemm tutorial
int main(int argc, char** argv)
{
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if ((props.major != 10) || (props.major == 10 && props.minor > 1)) {
    std::cerr << "This example requires NVIDIA's Blackwell Architecture GPU with compute capability 100a." << std::endl;
    std::cerr << "  Found " << props.major << "." << props.minor << std::endl;
    return -1;
  }

  int Gemm_M = 512;
  if (argc >= 2)
    sscanf(argv[1], "%d", &Gemm_M);

  int Gemm_N = 1024;
  if (argc >= 3)
    sscanf(argv[2], "%d", &Gemm_N);

  int Gemm_K = 256;
  if (argc >= 4)
    sscanf(argv[3], "%d", &Gemm_K);

  // Define the data types. A and B types are same for MMA instruction.
  using TypeA = cutlass::bfloat16_t; // MMA A Data Type
  auto type_str_a = "bf16";
  using TypeB = cutlass::bfloat16_t; // MMA B Data Type
  auto type_str_b = "bf16";
  using TypeC = float;           // MMA C Data Type
  [[maybe_unused]] auto type_str_c = "float";
  using TypeD = float;           // MMA D Data Type
  auto type_str_d = "float";
  using TypeAccumulator = float; // Both TypeC and TypeD are float, use float accumulator type.

  // A tensor MxK K-major (Layout T = Row-Major)
  Layout layout_A = make_layout(make_shape (Gemm_M,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
  // B tensor NxK K-major (Layout N = Column-Major)
  Layout layout_B = make_layout(make_shape (Gemm_N,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
  // C tensor MxN N-major (Layout T = Row-Major)
  Layout layout_C = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  // D tensor MxN N-major (Layout T = Row-Major)
  Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)

  // Host allocations and host CuTe tensors for A, B, and C tensors.
  thrust::host_vector<TypeA>   host_A(Gemm_M * Gemm_K);
  Tensor host_tensor_A = make_tensor(host_A.data(), layout_A);
  print("host_tensor_A:\t"); print(host_tensor_A); print("\n"); // host_tensor_A:	ptr[16b](ADDR_A) o (512,256):(256,_1)

  thrust::host_vector<TypeB>   host_B(Gemm_N * Gemm_K);
  Tensor host_tensor_B = make_tensor(host_B.data(), layout_B);
  print("host_tensor_B:\t"); print(host_tensor_B); print("\n"); // host_tensor_B:	ptr[16b](ADDR_B) o (1024,256):(256,_1)

  thrust::host_vector<TypeC>   host_C(Gemm_M * Gemm_N);
  Tensor host_tensor_C = make_tensor(host_C.data(), layout_C);
  print("host_tensor_C:\t"); print(host_tensor_C); print("\n"); // host_tensor_C:	ptr[32b](ADDR_C) o (512,1024):(1024,_1)

  // Note that we don't need a host_tensor for D yet.
  thrust::device_vector<TypeD> device_D(Gemm_M * Gemm_N);

  // Initialize A, B, and C tensors with random values.
  initialize_tensor(host_tensor_A);
  initialize_tensor(host_tensor_B);
  initialize_tensor(host_tensor_C);

  // Copy A, B, and C tensors from host memory to device memory
  thrust::device_vector<TypeA> device_A = host_A;
  thrust::device_vector<TypeB> device_B = host_B;
  thrust::device_vector<TypeC> device_C = host_C;

  using Alpha = float;
  using Beta = float;
  Alpha alpha = 1.0f;
  Beta beta = 0.0f;
  // Setup input and output tensors, and the kernel parameters; and execute the kernel on device
  gemm_host(device_A.data().get(), layout_A,
                                device_B.data().get(), layout_B,
                                device_C.data().get(), layout_C,
                                device_D.data().get(), layout_D,
                                alpha, beta);
  // Host allocation for D tensor and transfer D tensor from device to host
  thrust::host_vector<TypeD> host_D = device_D;
  // Create a non-owning CuTe tensor for D tensor
  Tensor host_tensor_D = make_tensor(host_D.data(), layout_D);

  thrust::host_vector<TypeD> host_reference_D(Gemm_M*Gemm_N);
  auto host_reference_tensor_D = make_tensor(host_reference_D.data(), layout_D);
  reference_gemm<TypeAccumulator>(host_tensor_A, host_tensor_B, host_tensor_C, host_reference_tensor_D, alpha, beta);

  auto relative_error = print_matrix_multiply_mollified_relative_error(type_str_a, host_tensor_A,
                                                                       type_str_b, host_tensor_B,
                                                                       type_str_d, host_tensor_D, host_reference_tensor_D);
  bool success = relative_error <= 0.0;
  std::cout << "Execution is " << ((success) ? "successful." : "failed.") << std::endl;

  return 0;
}
