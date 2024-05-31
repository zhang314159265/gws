#ifndef NVGPU_ATTRDEFS
#define NVGPU_ATTRDEFS

include "tritoncc/dialect/NVGPU/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class NVGPU_Attr<string name, list<Trait> traits = [],
    string baseCppClass = "::mlir::Attribute">
  : AttrDef<NVGPU_Dialect, name, traits, baseCppClass> {
}

#endif
