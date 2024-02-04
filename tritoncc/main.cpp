#include <iostream>
#include "mlir/IR/MLIRContext.h"

int main(void) {
  mlir::MLIRContext ctx;
  std::cout << "constructed an mlir::MLIRContext object " << &ctx << std::endl;
  std::cout << "bye" << std::endl;
  return 0;
}
