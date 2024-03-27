import triton
import triton.language as tl
import torch
import ptxrunner
import ttirrunner

torch.set_default_device("cuda")

ttir_code = """
module {
  tt.func public @kernel_0d1d2d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2048_i32 = arith.constant 2048 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xi32>
    %7 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %8 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xi32>
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %11 = tt.cat %7, %10 : tensor<1024xf32> -> tensor<2048xf32>
    %12 = arith.muli %0, %c2048_i32 : i32
    %13 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %14 = tt.splat %12 : i32 -> tensor<2048xi32>
    %15 = arith.addi %14, %13 : tensor<2048xi32>
    %16 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<2048x!tt.ptr<f32, 1>>
    %17 = tt.addptr %16, %15 : tensor<2048x!tt.ptr<f32, 1>>, tensor<2048xi32>
    tt.store %17, %11 {cache = 1 : i32, evict = 1 : i32} : tensor<2048xf32>
    tt.return
  }
}
"""

M, N = 1024, 1024

x = torch.randn(M, N)
y = torch.randn(M, N)

ref = torch.cat([x, y], dim=1)
act = torch.empty_like(ref);
ptx_code = ttirrunner.ttir_to_ptx(ttir_code)
ptxrunner.load_and_run(ptx_code, args=(x, y, act), gridDim=M, blockDim=4*32, shared=2048)

print("tl.cat may reorder elements. So torch.allclose should return false here")
tol = {"atol": 1e-3, "rtol": 1e-3}
assert not torch.allclose(ref, act, **tol), f"ref\n{ref}\nact\n{act}"

print("However rowsum should match")
ref2 = ref.sum(dim=1)
act2 = act.sum(dim=1)
assert torch.allclose(ref2, act2, **tol), f"ref\n{ref2}\nact\n{act2}"
print("Pass!")

