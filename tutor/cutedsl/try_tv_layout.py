import cutlass
from cutlass import cute

import tabulate

def print_layout(layout, T=None):
    """
    When T is not None, treat layout as layout_mn and print the
    mapping btw (m,n) to (t,v)
    """
    assert len(layout.shape) == 2, "Only support 2D layout"
    table = []
    M = cute.size(layout, mode=(0,))
    N = cute.size(layout, mode=(1,))
    header = [*map(str, range(N))]
    for i in range(M):
        table.append([i])
        for j in range(N):
            v = layout((i, j))
            # print(f"{i} {j} -> {v}")
            if T is None:
                table[-1].append(v)
            else:
                table[-1].append((v % T, v // T))
    table_str = tabulate.tabulate(table, headers=header)
    print(table_str)

@cute.jit
def f():
    a = cute.make_layout((3, 4)); b = cute.make_layout((5, 6))
    a = cute.make_layout((2, 3), stride=(3, 1)); b = cute.make_layout((2, 2), stride=(2, 1))
    a = cute.make_layout((2, 3), stride=(1, 2)); b = cute.make_layout((2, 2), stride=(1, 2))
    a = cute.make_layout((4, 8), stride=(1, 4)); b = cute.make_layout((2, 2), stride=(1, 2))
    a = cute.make_layout((3,)); b = cute.make_layout((5,))
    a = cute.make_layout((2, 2), stride=(1, 2)); b = cute.make_layout((2, 2), stride=(1, 2))
    print(a)
    print(b)
    lp = cute.logical_product(a, b)
    print("lp", lp)
    rp = cute.raked_product(a, b)
    print("rp", rp)
    bp = cute.blocked_product(a, b)
    print("bp", bp)

    tv_out = cute.make_layout_tv(a, b)
    print("tv", tv_out)

    print("layout mn")
    if cutlass.const_expr(cute.rank(a) != 2):
        print(rp)
    else:
        print_layout(rp, T=cute.size(a))
    print("layout tv")
    if cutlass.const_expr(cute.rank(a) != 2):
        print(tv_out[1])
    else:
        M = cute.size(rp, mode=(0,))
        print_layout(tv_out[1], T=M)

f()
