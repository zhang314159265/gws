set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Pattern.td)
mlir_tablegen(Pattern.inc -gen-rewriters)

add_public_tablegen_target(tblgen_target)
