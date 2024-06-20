#ifndef TRITON_ATTR_DEFS
#define TRITON_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def TT_CacheModifierAttr : I32EnumAttr<
  "CacheModifier", "",
  [
    I32EnumAttrCase<"NONE", 1, "none">,
    I32EnumAttrCase<"CA", 2, "ca">,
    I32EnumAttrCase<"CG", 3, "cg">,
    I32EnumAttrCase<"WB", 4, "wb">,
    I32EnumAttrCase<"CS", 5, "cs">,
    I32EnumAttrCase<"WT", 6, "wt">,
  ]> {
  let cppNamespace = "::mlir::_tritoncc";
}

def TT_EvictionPolicyAttr : I32EnumAttr<
  "EvictionPolicy", "",
  [
    I32EnumAttrCase<"NORMAL", 1, "evict_normal">,
    I32EnumAttrCase<"EVICT_FIRST", 2, "evict_first">,
    I32EnumAttrCase<"EVICT_LAST", 3, "evict_last">
  ]> {
  let cppNamespace = "::mlir::_tritoncc";
}

def TT_PaddingOptionAttr : I32EnumAttr<
  "PaddingOption", "",
  [
    I32EnumAttrCase<"PAD_ZERO", 1, "zero">,
    // We can not set the string value to "NAN" because it is a keyword in C++
    I32EnumAttrCase<"PAD_NAN", 2, "nan">
  ]> {
  let cppNamespace = "::mlir::_tritoncc";
}

def TT_ProgramDim : I32EnumAttr<
  "ProgramIDDim", "",
  [
    I32EnumAttrCase<"X", 0, "x">,
    I32EnumAttrCase<"Y", 1, "y">,
    I32EnumAttrCase<"Z", 2, "z">,
  ]> {
  let cppNamespace = "::mlir::_tritoncc";
}

#endif
