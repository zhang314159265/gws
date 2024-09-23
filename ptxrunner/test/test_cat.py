"""
Test the ptx code for tl.cat. The concat may reorder elements.
"""
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
	.reg .pred 	%p<9>;
	.reg .b32 	%r<45>;
	.reg .b64 	%rd<24>;

    .reg.b64 %aptr, %bptr, %optr;
	ld.param.u64 	%aptr, [kernel_0d1d2d_param_0];
	ld.param.u64 	%bptr, [kernel_0d1d2d_param_1];
	mov.u32 %r1, %ctaid.x;
	shl.b32 	%r34, %r1, 10;
	ld.param.u64 	%optr, [kernel_0d1d2d_param_2];

    .reg.b32 %thread_id;
	mov.u32 	%thread_id, %tid.x;
	shl.b32 	%r37, %thread_id, 2;
	or.b32  	%r38, %r34, %r37;
	mul.wide.s32 	%rd12, %r38, 4;
	add.s64 	%rd1, %aptr, %rd12;
	add.s64 	%rd2, %rd1, 2048;

    // load tensor A
	ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd1 + 0 ];
	ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];

    // convert layout for tensor A
    .reg.b32 %smbase;
	shl.b32 	%r39, %thread_id, 4;
	mov.u32 	%smbase, global_smem;
	add.s32 	%r41, %smbase, %r39;
	st.shared.v4.u32 	[%r41], {%r2, %r3, %r4, %r5};
	bar.sync 	0;
	add.s32 	%r42, %smbase, %r37;
	ld.shared.u32 	%r18, [%r42];
	ld.shared.u32 	%r19, [%r42+512];
	ld.shared.u32 	%r20, [%r42+1024];
	ld.shared.u32 	%r21, [%r42+1536];
	bar.sync 	0;
	st.shared.v4.u32 	[%r41], {%r6, %r7, %r8, %r9};
	bar.sync 	0;
	ld.shared.u32 	%r22, [%r42];
	ld.shared.u32 	%r23, [%r42+512];
	ld.shared.u32 	%r24, [%r42+1024];
	ld.shared.u32 	%r25, [%r42+1536];

    // load tensor B 
	add.s64 	%rd3, %bptr, %rd12;
	add.s64 	%rd4, %rd3, 2048;
	ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];

    // shuffle tensor B
	bar.sync 	0;
	st.shared.v4.u32 	[%r41], {%r10, %r11, %r12, %r13};
	bar.sync 	0;
	ld.shared.u32 	%r26, [%r42];
	ld.shared.u32 	%r27, [%r42+512];
	ld.shared.u32 	%r28, [%r42+1024];
	ld.shared.u32 	%r29, [%r42+1536];
	bar.sync 	0;
	st.shared.v4.u32 	[%r41], {%r14, %r15, %r16, %r17};
	bar.sync 	0;
	ld.shared.u32 	%r30, [%r42];
	ld.shared.u32 	%r31, [%r42+512];
	ld.shared.u32 	%r32, [%r42+1024];
	ld.shared.u32 	%r33, [%r42+1536];

    // store result to tensor C
	shl.b32 	%r43, %r1, 11;
	or.b32  	%r44, %r43, %r37;
	mul.wide.s32 	%rd19, %r44, 4;
	add.s64 	%rd5, %optr, %rd19;
	add.s64 	%rd6, %rd5, 2048;
	add.s64 	%rd7, %rd5, 4096;
	add.s64 	%rd8, %rd5, 6144;
	st.global.v4.b32 [ %rd5 + 0 ], { %r18, %r19, %r20, %r21 };
	st.global.v4.b32 [ %rd6 + 0 ], { %r22, %r23, %r24, %r25 };
	st.global.v4.b32 [ %rd7 + 0 ], { %r26, %r27, %r28, %r29 };
	st.global.v4.b32 [ %rd8 + 0 ], { %r30, %r31, %r32, %r33 };
	ret;
}
"""

@torch.compile
def f(x, y):
    return torch.cat([x, y], dim=1)

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)

ref = f(x, y)
act = torch.zeros_like(ref)

ptxrunner.load_and_run(ptx_code, args=(x, y, act), gridDim=1024, blockDim=32*4, shared=2048)

tol = {"atol": 4 * 1e-3, "rtol": 4 * 1e-3}
print("tl.cat may reorder elements. So torch.allclose should return false here")
assert not torch.allclose(ref, act, **tol), f"ref:\n{ref}\nact:\n{act}"

print("However rowsum should match")
ref2 = ref.sum(dim=1)
act2 = act.sum(dim=1)
assert torch.allclose(ref2, act2, **tol), f"ref\n{ref2}\nact\n{act2}"

print("bye")
