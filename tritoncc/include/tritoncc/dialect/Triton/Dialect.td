#ifndef TRITON_DIALECT
#define TRITON_DIALECT

include "mlir/IR/OpBase.td"

def Triton_Dialect : Dialect {
  let name = "tt";
  let cppNamespace = "::mlir::_tritoncc";

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif
