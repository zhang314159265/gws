nvcc ~/gws/tutor/cutlass/standalone/copied_attn_from_ex41/x.cu -I. -I ~/ws/cutlass/include/ -I ~/ws/cutlass/tools/util/include/ --expt-relaxed-constexpr -arch=sm_100a && ./a.out
