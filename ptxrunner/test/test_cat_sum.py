import torch
import ptxrunner
from torch._inductor import config
from triton.testing import do_bench

config.benchmark_kernel = True
config.compile_threads = 1

torch.set_default_device("cuda")

# the corresponding triton code is in tutor/tl_cat.py
ptx_code = """
.version 8.0
.target sm_90
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry kernel_0d1d2d(
	.param .u64 kernel_0d1d2d_param_0,
	.param .u64 kernel_0d1d2d_param_1,
	.param .u64 kernel_0d1d2d_param_2
)
.maxntid 128, 1, 1
{
	.reg .pred 	%p<10>;
	.reg .b32 	%r<52>;
	.reg .f32 	%f<47>;
	.reg .b64 	%rd<11>;

    .reg.b64 %aptr, %bptr, %optr;
	ld.param.u64 	%aptr, [kernel_0d1d2d_param_0];
	ld.param.u64 	%bptr, [kernel_0d1d2d_param_1];
	mov.u32 %r1, %ctaid.x;

	shl.b32 	%r25, %r1, 10;

	ld.param.u64 	%optr, [kernel_0d1d2d_param_2];

    .reg.b32 %thread_id;
	mov.u32 	%thread_id, %tid.x;

    // load A
    .reg.b32 %lane_id;
	and.b32  	%lane_id, %thread_id, 31;
	shl.b32 	%r29, %thread_id, 2;  // each thread handle 4 elements
	or.b32  	%r30, %r25, %r29;
	mul.wide.s32 	%rd9, %r30, 4;
	add.s64 	%rd1, %aptr, %rd9;
	add.s64 	%rd2, %rd1, 2048;
	ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd1 + 0 ];
	ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];

    // convert sizePerThread from 4 to 1 for tensor A
	shl.b32 	%r31, %thread_id, 4;
	mov.u32 	%r32, global_smem;
	add.s32 	%r33, %r32, %r31;
	st.shared.v4.u32 	[%r33], {%r2, %r3, %r4, %r5};
	bar.sync 	0;
	add.s32 	%r34, %r32, %r29;
	ld.shared.f32 	%f1, [%r34];
	ld.shared.f32 	%f2, [%r34+512];
	ld.shared.f32 	%f3, [%r34+1024];
	ld.shared.f32 	%f4, [%r34+1536];
	bar.sync 	0;
	st.shared.v4.u32 	[%r33], {%r6, %r7, %r8, %r9};
	bar.sync 	0;
	ld.shared.f32 	%f5, [%r34];
	ld.shared.f32 	%f6, [%r34+512];
	ld.shared.f32 	%f7, [%r34+1024];
	ld.shared.f32 	%f8, [%r34+1536];

    // load tensor B
	add.s64 	%rd3, %bptr, %rd9;
	add.s64 	%rd4, %rd3, 2048;
	ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];

    // shuffle tneosr B
	bar.sync 	0;
	st.shared.v4.u32 	[%r33], {%r10, %r11, %r12, %r13};
	bar.sync 	0;
	ld.shared.f32 	%f9, [%r34];
	ld.shared.f32 	%f10, [%r34+512];
	ld.shared.f32 	%f11, [%r34+1024];
	ld.shared.f32 	%f12, [%r34+1536];
	bar.sync 	0;
	st.shared.v4.u32 	[%r33], {%r14, %r15, %r16, %r17};
	bar.sync 	0;
	ld.shared.f32 	%f13, [%r34];
	ld.shared.f32 	%f14, [%r34+512];
	ld.shared.f32 	%f15, [%r34+1024];
	ld.shared.f32 	%f16, [%r34+1536];
	bar.sync 	0;

	add.f32 	%f17, %f1, %f2;
	add.f32 	%f18, %f17, %f3;
	add.f32 	%f19, %f18, %f4;
	add.f32 	%f20, %f19, %f5;
	add.f32 	%f21, %f20, %f6;
	add.f32 	%f22, %f21, %f7;
	add.f32 	%f23, %f22, %f8;
	add.f32 	%f24, %f23, %f9;
	add.f32 	%f25, %f24, %f10;
	add.f32 	%f26, %f25, %f11;
	add.f32 	%f27, %f26, %f12;
	add.f32 	%f28, %f27, %f13;
	add.f32 	%f29, %f28, %f14;
	add.f32 	%f30, %f29, %f15;
	add.f32 	%f31, %f30, %f16;

    // intra warp shuffle
	mov.b32 	%r35, %f31;
	shfl.sync.bfly.b32	%r36, %r35, 16, 31, -1;
	mov.b32 	%f32, %r36;
	add.f32 	%f33, %f31, %f32;

	mov.b32 	%r37, %f33;
	shfl.sync.bfly.b32	%r38, %r37, 8, 31, -1;
	mov.b32 	%f34, %r38;
	add.f32 	%f35, %f33, %f34;

	mov.b32 	%r39, %f35;
	shfl.sync.bfly.b32	%r40, %r39, 4, 31, -1;
	mov.b32 	%f36, %r40;
	add.f32 	%f37, %f35, %f36;

	mov.b32 	%r41, %f37;
	shfl.sync.bfly.b32	%r42, %r41, 2, 31, -1;
	mov.b32 	%f38, %r42;
	add.f32 	%f39, %f37, %f38;

	mov.b32 	%r43, %f39;
	shfl.sync.bfly.b32	%r44, %r43, 1, 31, -1;
	mov.b32 	%f40, %r44;
	add.f32 	%f41, %f39, %f40;

    // lane0 write intra wrap shuffle result to smem
	setp.eq.s32 	%p5, %lane_id, 0;
	shr.u32 	%r45, %thread_id, 3;
	add.s32 	%r18, %r32, %r45;
	mov.b32 	%r19, %f41;
	@%p5 st.shared.b32 [ %r18 + 0 ], %r19;
	bar.sync 	0;

    // inter warp shuffle
	setp.lt.s32 	%p6, %thread_id, 4;
	shl.b32 	%r47, %thread_id, 2;
	add.s32 	%r21, %r32, %r47;
	@%p6 ld.shared.b32 %r20, [ %r21 + 0 ];

	mov.b32 	%f42, %r20;
	shfl.sync.bfly.b32	%r48, %r20, 2, 31, -1;
	mov.b32 	%f43, %r48;
	add.f32 	%f44, %f42, %f43;

	mov.b32 	%r49, %f44;
	shfl.sync.bfly.b32	%r50, %r49, 1, 31, -1;
	mov.b32 	%f45, %r50;
	add.f32 	%f46, %f44, %f45;

    // broadcast reduciton result to all threads
	and.b32  	%r51, %thread_id, 3;
	setp.eq.s32 	%p9, %r51, 0;
	and.pred  	%p7, %p6, %p9;
	mov.b32 	%r23, %f46;
	@%p7 st.shared.b32 [ %r21 + 0 ], %r23;
	bar.sync 	0;
	ld.shared.u32 	%r24, [global_smem];

	mul.wide.s32 	%rd10, %r1, 4;
	add.s64 	%rd5, %optr, %rd10;
	setp.eq.s32 	%p8, %thread_id, 0;
	@%p8 st.global.b32 [ %rd5 + 0 ], { %r24 };
	ret;
}
"""

@torch.compile
def f(x, y):
    return torch.cat([x, y], dim=1).sum(dim=1) 

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)

ref = f(x, y)
act = torch.zeros_like(ref)

ptxrunner.load_and_run(ptx_code, args=(x, y, act), gridDim=1024, blockDim=32*4, shared=2048)

tol = {"atol": 1e-3, "rtol": 1e-3}
assert torch.allclose(ref, act, **tol), f"ref:\n{ref}\nact:\n{act}"
print("bye")
