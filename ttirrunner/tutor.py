import triton
import triton.language as tl
import torch
import ptxrunner
import ttirrunner

torch.set_default_device("cuda")

ttir_code = """
module {
  tt.func public @my_ttir_func(
    %a: !tt.ptr<f32, 1> {tt.divisibility = 16: i32},
    %b: !tt.ptr<f32, 1> {tt.divisibility = 16: i32},
    %o: !tt.ptr<f32, 1> {tt.divisibility = 16: i32},
    %numel: i32 {tt.divisibility = 16: i32}
  ) attributes {noinline = false} {
    %c32 = arith.constant 32: i32
    %rng = tt.make_range {end = 32: i32, start = 0: i32}: tensor<32xi32>
    %blk_id = tt.get_program_id x: i32
    %base = arith.muli %blk_id, %c32: i32
    %base_blk = tt.splat %base: i32 -> tensor<32xi32>
    %idx = arith.addi %rng, %base_blk: tensor<32xi32>
    %numel_blk = tt.splat %numel: i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %idx, %numel_blk: tensor<32xi32>
    %a_blk = tt.splat %a: !tt.ptr<f32, 1> -> tensor<32x!tt.ptr<f32, 1>>
    %aptr_blk = tt.addptr %a_blk, %idx: tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>

    %aval = tt.load %aptr_blk, %mask {cache = 1: i32, evict = 1: i32, isVolatile = false}: tensor<32xf32>

    %b_blk = tt.splat %b: !tt.ptr<f32, 1> -> tensor<32x!tt.ptr<f32, 1>>
    %bptr_blk = tt.addptr %b_blk, %idx: tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>

    %bval = tt.load %bptr_blk, %mask {cache=1: i32, evict=1: i32, isVolatile=false}: tensor<32xf32>
    %oval = arith.addf %aval, %bval: tensor<32xf32>
    %o_blk = tt.splat %o: !tt.ptr<f32, 1> -> tensor<32x!tt.ptr<f32, 1>>
    
    %optr_blk = tt.addptr %o_blk, %idx: tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    tt.store %optr_blk, %oval, %mask {cache=1: i32, evict=1: i32}: tensor<32xf32>
    tt.return
  }
}
"""

a = torch.randn(1024)
b = torch.randn(1024)
ref = a + b
act = torch.empty_like(ref)

ptx_code = ttirrunner.ttir_to_ptx(ttir_code)
ptxrunner.load_and_run(ptx_code, args=(a, b, act, a.numel()), gridDim=32, blockDim=32*4, shared=0)

assert torch.allclose(ref, act), f"ref\n{ref}\nact\n{act}"
print("Pass!")
