import torch
import ptxrunner
from torch._inductor import config
from torch._inductor.utils import do_bench

config.benchmark_kernel = True
config.compile_threads = 1

torch.set_default_device("cuda")

ptx_code = """
.version 8.0
.target sm_80
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry triton__0d1d2de3de(
	.param .u64 triton__0d1d2de3de_param_0,
	.param .u64 triton__0d1d2de3de_param_1,
	.param .u32 triton__0d1d2de3de_param_2,
	.param .u32 triton__0d1d2de3de_param_3
)
.maxntid 256, 1, 1
{
	.reg .pred 	%p<27>;
	.reg .b32 	%r<57>;
	.reg .f32 	%f<74>;
	.reg .b64 	%rd<8>;

    .reg .b64 %iptr, %optr;
	ld.param.u64 	%optr, [triton__0d1d2de3de_param_1];
	ld.param.u64 	%iptr, [triton__0d1d2de3de_param_0];
	mov.u32 %r7, %ctaid.x; // blkid
	mov.u32 	%r2, %tid.x; // thread_id

    .reg .b32 %thread_id_x4;
	shl.b32 	%thread_id_x4, %r2, 2;

    .reg .b32 %blkid_x_rs;
	shl.b32 	%blkid_x_rs, %r7, 16;
	or.b32  	%r4, %blkid_x_rs, %thread_id_x4;
	mov.b32 	%r56, -2048;

	mov.f32 	%f66, 0f00000000;
	mov.f32 	%f67, %f66;
	mov.f32 	%f68, %f66;
	mov.f32 	%f69, %f66;
	mov.f32 	%f70, %f66;
	mov.f32 	%f71, %f66;
	mov.f32 	%f72, %f66;
	mov.f32 	%f73, %f66;
LOOP_ENTRY:
	add.s32 	%r56, %r56, 2048;
	or.b32  	%r27, %r4, %r56;
	mul.wide.s32 	%rd5, %r27, 4;
	add.s64 	%rd3, %iptr, %rd5;
	add.s64 	%rd4, %rd3, 4096;

    ld.global.L1::evict_first.v4.b32 {%r11, %r12, %r13, %r14}, [%rd3];
    ld.global.L1::evict_first.v4.b32 {%r19, %r20, %r21, %r22}, [%rd4];

    // loaded 8 floats
	mov.b32 	%f25, %r14;
	mov.b32 	%f26, %r13;
	mov.b32 	%f27, %r12;
	mov.b32 	%f28, %r11;
	mov.b32 	%f29, %r22;
	mov.b32 	%f30, %r21;
	mov.b32 	%f31, %r20;
	mov.b32 	%f32, %r19;

    // accumulate the 8 floats to the accumulator
	add.f32 	%f69, %f69, %f25;
	add.f32 	%f68, %f68, %f26;
	add.f32 	%f67, %f67, %f27;
	add.f32 	%f66, %f66, %f28;
	add.f32 	%f73, %f73, %f29;
	add.f32 	%f72, %f72, %f30;
	add.f32 	%f71, %f71, %f31;
	add.f32 	%f70, %f70, %f32;

	setp.lt.u32 	%p19, %r56, 63488;
	@%p19 bra 	LOOP_ENTRY;

	add.f32 	%f41, %f66, %f67;
	add.f32 	%f42, %f68, %f41;
	add.f32 	%f43, %f69, %f42;
	add.f32 	%f44, %f70, %f43;
	add.f32 	%f45, %f71, %f44;
	add.f32 	%f46, %f72, %f45;
	add.f32 	%f47, %f73, %f46;

    // shuffle within warp
    // rnd1
	mov.b32 	%r36, %f47;
	shfl.sync.bfly.b32	%r37, %r36, 16, 31, -1;
	mov.b32 	%f48, %r37;
	add.f32 	%f49, %f47, %f48;

    // rnd2
	mov.b32 	%r38, %f49;
	shfl.sync.bfly.b32	%r39, %r38, 8, 31, -1;
	mov.b32 	%f50, %r39;
	add.f32 	%f51, %f49, %f50;

    // rnd3
	mov.b32 	%r40, %f51;
	shfl.sync.bfly.b32	%r41, %r40, 4, 31, -1;
	mov.b32 	%f52, %r41;
	add.f32 	%f53, %f51, %f52;

    // rnd4
	mov.b32 	%r42, %f53;
	shfl.sync.bfly.b32	%r43, %r42, 2, 31, -1;
	mov.b32 	%f54, %r43;
	add.f32 	%f55, %f53, %f54;

    // rnd5
	mov.b32 	%r44, %f55;
	shfl.sync.bfly.b32	%r45, %r44, 1, 31, -1;
	mov.b32 	%f56, %r45;
	add.f32 	%f57, %f55, %f56;

    // mask for lane0
	and.b32  	%r35, %r2, 31;
	setp.eq.s32 	%p20, %r35, 0;

	shr.u32 	%r46, %r2, 3; // warp_id * 4
	mov.u32 	%r48, global_smem;
	add.s32 	%r28, %r48, %r46;
	mov.b32 	%r29, %f57;
	@%p20 st.shared.b32 [%r28], %r29;
	bar.sync 	0;

    // only the first 8 threads in a block participate in the inter-warp
    // shuffle
	setp.lt.s32 	%p21, %r2, 8;

	shl.b32 	%r49, %r2, 2;
	add.s32 	%r31, %r48, %r49;
	@%p21 ld.shared.b32 %r30, [%r31];
	mov.b32 	%f58, %r30;

    // there are 8 elements, so we only need 3 rounds of warp shuffle
    // round 1
	shfl.sync.bfly.b32	%r50, %r30, 4, 31, -1;
	mov.b32 	%f59, %r50;
	add.f32 	%f60, %f58, %f59;

    // round 2
	mov.b32 	%r51, %f60;
	shfl.sync.bfly.b32	%r52, %r51, 2, 31, -1;
	mov.b32 	%f61, %r52;
	add.f32 	%f62, %f60, %f61;

    // round 3
	mov.b32 	%r53, %f62;
	shfl.sync.bfly.b32	%r54, %r53, 1, 31, -1;
	mov.b32 	%f63, %r54;
	add.f32 	%f64, %f62, %f63;

    // 0th of the 8 threads writes the output to shared memory
    // But why not write to the global memory directly?
    // It's to make sure every thread in the block get the sum loaded
    // into it's register?

	and.b32  	%r55, %r2, 7;
	setp.eq.s32 	%p25, %r55, 0;
	and.pred  	%p22, %p21, %p25;
	mov.b32 	%r33, %f64;
	@%p22 st.shared.b32 [global_smem], %r33;
	bar.sync 	0;
	ld.shared.f32 	%f65, [global_smem];
	bar.sync 	0;
    mov.b32 %r34, %f64;

	mul.wide.s32 	%rd7, %r7, 4;
	add.s64 	%rd6, %optr, %rd7;
	setp.eq.s32 	%p26, %r2, 0;
	@%p26 st.global.b32 [%rd6], { %r34 };
	ret;
}
"""

@torch.compile
def f(x):
    return x.sum(dim=-1)

x = torch.rand(1024, 2 ** 16)
ref = f(x)
act = torch.zeros_like(ref)

ptxrunner.load_and_run(ptx_code, args=(x, act, x.size(0), x.size(1)), gridDim=1024, blockDim=32*8, shared=32)

assert torch.allclose(ref, act), f"ref:\n{ref}\nact:\n{act}"

ms_torch_compile = do_bench(lambda: f(x))
cu_func = ptxrunner.load_cu_func_from_ptx_code(ptx_code)
ms_ptx_runner = do_bench(lambda: ptxrunner.launch(cu_func, args=(x, act, x.size(0), x.size(1)), gridDim=1024, blockDim=32*8, shared=32))
print(f"ms_torch_compile {ms_torch_compile:.3f}ms v.s. ms_ptx_runner {ms_ptx_runner:.3f}ms")

print("bye")
