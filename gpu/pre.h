#include "cuda_bf16.h"

using bfloat16 = __nv_bfloat16;

enum L1EVICT {
  L1EVICT_NONE,
  L1EVICT_FIRST,
  L1EVICT_LAST,
};

__device__ int4 loadInt4(void *ptr, int l1evict) {
  switch (l1evict) {
  case L1EVICT_NONE:
    return *(int4*) ptr;
  case L1EVICT_FIRST: {
    int4 ans;
    asm(
      "ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
      : "=r"(ans.x), "=r"(ans.y), "=r"(ans.z), "=r"(ans.w)
      : "l"(ptr)
    );
    return ans;
  }
  case L1EVICT_LAST: {
    int4 ans;
    asm(
      "ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
      : "=r"(ans.x), "=r"(ans.y), "=r"(ans.z), "=r"(ans.w)
      : "l"(ptr)
    );
    return ans;
  }
  default:
    assert(false);
  }
}

__device__ void storeInt4(void *ptr, int4 content) {
  *(int4*) ptr = content;
}
