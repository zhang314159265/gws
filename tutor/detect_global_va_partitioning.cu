#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <vector>

using namespace std;

int main(void) {
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("unifiedAddressing %d\n", prop.unifiedAddressing);
  }

  int64_t N = 1024LL * 1024 * 1024 * 4;
  int K = 8;
  vector<int64_t> all_addr;
  for (int dev_id = 0; dev_id < 8; ++dev_id) {
    cudaSetDevice(dev_id);
    printf("Device: %d\n", dev_id);
    for (int i = 0; i < K; ++i) {
      void *ptr = nullptr;
      cudaMalloc(&ptr, N);
      printf("  ptr%d: %p\n", i, ptr);
      all_addr.push_back((int64_t) ptr);
    }
  }
  sort(all_addr.begin(), all_addr.end());
  for (int i = 0; i + 1 < all_addr.size(); ++i) {
    int64_t dif = all_addr[i + 1] - all_addr[i];
    if (dif < N) {
      printf("Detected address overlapping! %p v.s. %p\n", (void *) all_addr[i], (void *) all_addr[i + 1]);
      return -1;
    }
  }
  printf("No address overlapping detected across GPUs\n");
  // sleep(3600);
  return 0;
}
