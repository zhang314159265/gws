#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"

def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;

#endif
