# the make file fragment handles building toyc manually
LLVM_ROOT := ${HOME}/.triton/llvm/llvm-4017f04e-ubuntu-x64
TBLGEN := ${LLVM_ROOT}/bin/mlir-tblgen

TBLGEN_INC_FLAGS := -I${LLVM_ROOT}/include -Iinclude

CFLAGS := -I${LLVM_ROOT}/include -Iinclude -Iout/include -fno-rtti

include ../../llvm_or_mlir_libs.make
LDFLAGS := -lz -Wl,--start-group ${LLVM_OR_MLIR_LIBS} -Wl,--end-group -L ${LLVM_ROOT}/lib

build_toyc_via_make:
	mkdir -p out/include/toy/
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} include/toy/Ops.td -o out/include/toy/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} include/toy/Ops.td -o out/include/toy/Dialect.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} include/toy/Ops.td -o out/include/toy/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} include/toy/Ops.td -o out/include/toy/Ops.cpp.inc
	${TBLGEN} -gen-rewriters ${TBLGEN_INC_FLAGS} include/toy/Pattern.td -o out/include/toy/Pattern.inc
	${TBLGEN} -gen-op-interface-decls ${TBLGEN_INC_FLAGS} include/toy/ShapeInferenceInterface.td -o out/include/toy/ShapeInferenceOpInterfaces.h.inc
	${TBLGEN} -gen-op-interface-defs ${TBLGEN_INC_FLAGS} include/toy/ShapeInferenceInterface.td -o out/include/toy/ShapeInferenceOpInterfaces.cpp.inc
	g++ ${CFLAGS} toyc.cpp Dialect.cpp ${LDFLAGS} -o out/toyc
	rm -rf build/ && mkdir build
	cp out/toyc build/toyc
