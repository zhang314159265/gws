# A makefile that has partial functionality of the CMake counterpart.
# Eventually use this to understand the CMake counterpart.

all: struct

nested_struct: build_toyc
	build/toyc toy-prog/nested_struct.toy -emit=jit -opt

struct: build_toyc
	build/toyc toy-prog/struct.toy -emit=mlir 2>&1 | tee /tmp/out.mlir
	build/toyc /tmp/out.mlir -emit=mlir -opt
	build/toyc toy-prog/struct.toy -emit=llvm -opt
	build/toyc toy-prog/struct.toy -emit=jit -opt

jit: build_toyc
	build/toyc toy-prog/ast.toy -emit=jit -opt
	echo 'def main() { print([[1, 2], [3, 4]]); }' | build/toyc -emit=jit -opt

trivial_reshape: build_toyc
	build/toyc toy-prog/trivial_reshape.toy -emit=mlir -opt

transpose_transpose: build_toyc
	build/toyc toy-prog/transpose_transpose.toy -emit=mlir -opt

round_trip: build_toyc
	build/toyc toy-prog/ast.toy -emit=mlir 2>&1 | tee /tmp/out.mlir
	build/toyc /tmp/out.mlir -emit=mlir -opt

build_toyc_via_cmake:
	mkdir -p build
	cd build && cmake .. && make

include build_toyc.make

build_toyc: build_toyc_via_make
