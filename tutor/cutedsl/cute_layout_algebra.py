import cutlass
from cutlass import cute

@cute.jit
def coalesce_example():
    layout = cute.make_layout((2, (1, 6)), stride=(1, (cutlass.Int32(6), 2)))
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    cute.printf(">?? Orignal: {}", layout)
    print(">>> Coalesced:", result)
    cute.printf(">?? Coalesced: {}", result)
    print(f"Depth of coalesced layout: {cute.depth(result)}, original depth {cute.depth(layout)}")

coalesce_example()

@cute.jit
def coalesce_post_conditions():
    layout = cute.make_layout(
        ((2, (3, 4)), (3, 2), 1),
        stride=((4, (8, 24)), (2, 6), 12)
    )
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    print(">>> Coalesced:", result)

    print(">>> Checking post-conditions:")
    print(">>> 1. Checking size remains the same after the coalesce operation:")
    original_size = cute.size(layout)
    coalesced_size = cute.size(result)
    print(f"Original size: {original_size}, Coalesced size: {coalesced_size}")
    assert coalesced_size == original_size, \
        f"Size mismatch: original {original_size}, coalseced {coalesced_size}"

    print(">>> 2. Checking depth of coalesced layout <= 1:")
    original_depth = cute.depth(layout)
    depth = cute.depth(result)
    print(f"Depth of coalesced layout: {depth}, original depth {original_depth}")
    assert depth <= 1

    print(">>> 3. Checking layout functionality remains the same after the coalesce operation:")
    for i in cutless.range_constexpr(original_size):
        original_value = layout(i)
        coalesced_value = result(i)
        print(f"Index {i}: original {original_value}, coalesced {coalesced_value}")
        assert coalesced_value == original_value

coalesce_post_conditions()

@cute.jit
def bymode_coalsec_example():
    layout = cute.make_layout((2, (1, 6)), stride=(1, (6, 2)))
    result = cute.coalesce(layout, target_profile=(1, 1))

    print(">>> Original: ", layout)
    print(">>> Coalesced Result: ", result)

bymode_coalsec_example()

@cute.jit
def composition_example():
    A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2))
    B = cute.make_layout((4, 3), stride=(3, 1))
    R = cute.composition(A, B)

    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Layout B:", B)
    cute.printf(">?? Layout B: {}", B)
    print(">>> Composition R = A o B:", R)
    cute.printf(">?? Composition R: {}", R)

composition_example()

@cute.jit
def composition_static_vs_dynamic_layout():
    A_static = cute.make_layout(
        (10, 2),
        stride=(16, 4)
    )
    B_static = cute.make_layout(
        (5, 4),
        stride=(1, 5)
    )
    R_static = cute.composition(A_static, B_static)

    print(">>> Static composition:")
    print(">>> A_static: ", A_static)
    print(">>> B_static: ", B_static)
    print(">>> R_static: ", R_static)

    A_dynamic = cute.make_layout(
        (cutlass.Int32(10), cutlass.Int32(2)),
        stride=(cutlass.Int32(16), cutlass.Int32(4))
    )
    B_dynamic = cute.make_layout(
        (cutlass.Int32(5), cutlass.Int32(4)),
        stride=(cutlass.Int32(1), cutlass.Int32(5))
    )
    R_dynamic = cute.composition(A_dynamic, B_dynamic)
    cute.printf(">?? Dynamic composition:")
    cute.printf(">?? A_dynamic: {}", A_dynamic)
    cute.printf(">?? B_dynamic: {}", B_dynamic)
    cute.printf(">?? R_dynamic: {}", R_dynamic)

composition_static_vs_dynamic_layout()

@cute.jit
def bymode_composition_example():
    A = cute.make_layout(
        (cutlass.Int32(12), (cutlass.Int32(4), cutlass.Int32(8))),
        stride=(cutlass.Int32(59), (cutlass.Int32(13), cutlass.Int32(1)))
    )

    tiler = (3, 8)

    # apply by-mode composition
    result = cute.composition(A, tiler)
    print(">>> Layout A:", A)
    cute.printf(">??? Layout A: {}", A)
    print(">>> Tiler:", tiler)
    cute.printf(">?? Tiler: {}", tiler)
    print(">>> By-mode composition result:", result)
    cute.printf(">?? By-mode composition result: {}", result)

bymode_composition_example()

@cute.jit
def logical_divide_1d_example():
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))
    tiler = cute.make_layout(4, stride=2)

    result = cute.logical_divide(layout, tiler=tiler)
    print(">>> layout:", layout)
    print(">>> tiler :", tiler)
    print(">>> logical divide result:", result)
    cute.printf(">?? Logical Divide result: {}", result)


logical_divide_1d_example()

@cute.jit
def logical_divide_2d_example():
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))

    tiler = (cute.make_layout(3, stride=3),
        cute.make_layout((2, 4), stride=(1, 8)))
    result = cute.logical_divide(layout, tiler=tiler)

    print(">>> layout:", layout)
    print(">>> tiler :", tiler)
    print(">>> Logical divide result:", result)
    cute.printf(">?? Logical divide result: {}", result)

logical_divide_2d_example()

@cute.jit
def logical_product_1d_example():
    layout = cute.make_layout((2, 2), stride=(4, 1))

    tiler = cute.make_layout(6, stride=1)

    result = cute.logical_product(layout, tiler=tiler)

    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical product result:", result)
    cute.printf(">?? Logical product result: {}", result)

logical_product_1d_example()

exit()


