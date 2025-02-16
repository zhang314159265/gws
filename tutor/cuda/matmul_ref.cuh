// TODO use static int lie cute::Int<16>{}
#define BLK_M 16
#define BLK_N 16
#define BLK_K 16

template <typename T>
__global__ void matmul_ref_kernel(
  int M, int N, int K,
  const T *A, int stride_am, int stride_ak,
  const T *B, int stride_bk, int stride_bn,
  T *C, int stride_cm, int stride_cn) {
  __shared__ T sA[BLK_M][BLK_K];
  __shared__ T sB[BLK_K][BLK_N];
  __shared__ T sC[BLK_M][BLK_N];

  int blkx = blockIdx.x, blky = blockIdx.y;
  int trdx = threadIdx.x, trdy = threadIdx.y;

  // clear sC
  sC[trdx][trdy] = 0.0f;
  __syncthreads();

  // TODO; only work when K is multiple of BLK_K for now
  for (int kidx = 0; kidx < K; kidx += BLK_K) {
    // load next block of data to sA, sB
    // TODO: this relis on the fact that BLK_M==BLK_N==BLK_K
    sA[trdx][trdy] = A[(blkx * BLK_M + trdx) * stride_am + (kidx + trdy) * stride_ak];
    sB[trdx][trdy] = B[(kidx + trdx) * stride_bk + (blky * BLK_N + trdy) * stride_bn];
    __syncthreads();

    // each thread compute one output element
    for (int i = 0; i < BLK_K; ++i) {
      sC[trdx][trdy] += sA[trdx][i] * sB[i][trdy];
    }

    __syncthreads();
  }
 
  // write sC out
  __syncthreads();
  C[(blkx * BLK_M + trdx) * stride_cm + (blky * BLK_N + trdy) * stride_cn] = sC[trdx][trdy];
}

template <typename T>
void matmul_ref(
  int M, int N, int K,
  const T *A, int stride_am, int stride_ak,
  const T *B, int stride_bk, int stride_bn,
  T *C, int stride_cm, int stride_cn) {

  // TODO this can be improved
  assert(BLK_M == BLK_N);
  assert(BLK_N == BLK_K);
  assert(M % BLK_M == 0);
  assert(N % BLK_N == 0);
  assert(K % BLK_K == 0);

  int cnt_blk_m = (M + BLK_M - 1) / BLK_M;
  int cnt_blk_n = (N + BLK_N - 1) / BLK_N;

  dim3 grid_dims(cnt_blk_m, cnt_blk_n);
  dim3 block_dims(BLK_M, BLK_N);
  matmul_ref_kernel<<<
    grid_dims,
    block_dims
  >>>(
    M, N, K,
    A, stride_am, stride_ak,
    B, stride_bk, stride_bn,
    C, stride_cm, stride_cn
  );
}

#undef BLK_M
#undef BLK_N
#undef BLK_K
