# A makefile that has partial functionality of the CMake counterpart.
# Eventually use this to understand the CMake counterpart.

with_cmake:
	cd build && cmake .. && make
	build/toyc toy-prog/ast.toy -emit=ast

# XXX this does not work yet.
run:
	mkdir -p out
	g++ toyc.cpp -o out/toyc
	out/toyc toy-prog/ast.toy -emit=ast
