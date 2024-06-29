# The enclosing makefile should properly setup LLVM_ROOT and TRITONCC_ROOT

TBLGEN := @${LLVM_ROOT}/bin/mlir-tblgen
TRITONCC_INC_DIR := ${TRITONCC_ROOT}/include
TRITONCC_OUT_DIR := ${TRITONCC_ROOT}/out
TBLGEN_INC_FLAGS := -I${LLVM_ROOT}/include -I${TRITONCC_ROOT}/include

run_tblgen:
	@mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU
	@mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU
	@mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU
	@mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonNvidiaGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonNvidiaGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU/Dialect.cpp.inc
	${TBLGEN} -gen-attrdef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonNvidiaGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU/AttrDefs.cpp.inc
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Dialect.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Ops.cpp.inc
	${TBLGEN} -gen-attrdef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/AttrDefs.cpp.inc
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Dialect.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Ops.cpp.inc
	${TBLGEN} -gen-attrdef-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrDefs.h.inc
	${TBLGEN} -gen-attrdef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrDefs.cpp.inc
	${TBLGEN} -gen-attr-interface-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrInterfaces.h.inc
	${TBLGEN} -gen-attr-interface-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrInterfaces.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Ops.cpp.inc
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Dialect.cpp.inc
	${TBLGEN} -gen-enum-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/OpsEnums.h.inc
	${TBLGEN} -gen-enum-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/OpsEnums.cpp.inc
	${TBLGEN} -gen-typedef-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Types.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Types.h.inc
	${TBLGEN} -gen-typedef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Types.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Types.cpp.inc
