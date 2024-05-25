#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
        "void", "inferShapes">
  ];
}

#endif
