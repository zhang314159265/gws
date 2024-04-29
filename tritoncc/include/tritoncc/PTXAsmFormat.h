#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 1
#if USE_TRITON

#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#else

namespace tritoncc {

}
#endif
