# A makefile that has partial functionality of the CMake counterpart.
# Eventually use this to understand the CMake counterpart.

shape_inference_and_partial_lowering: build_toyc
	build/toyc toy-prog/ast.toy -emit=mlir -opt

trivial_reshape: build_toyc
	build/toyc toy-prog/trivial_reshape.toy -emit=mlir -opt

transpose_transpose: build_toyc
	build/toyc toy-prog/transpose_transpose.toy -emit=mlir -opt

round_trip: build_toyc
	build/toyc toy-prog/ast.toy -emit=mlir 2>&1 | tee /tmp/out.mlir
	build/toyc /tmp/out.mlir -emit=mlir -opt

build_toyc:
	cd build && cmake .. && make

# XXX this does not work yet.
run:
	mkdir -p out
	g++ toyc.cpp -o out/toyc
	out/toyc toy-prog/ast.toy -emit=ast
