from tritoncc._C import ir

def cast(input, dst_ty, builder):
    """
    dst_ty can be an element type (rather than block)
    """
    src_ty = input.type()
    if src_ty.is_block():
        dst_ty = builder.get_block_ty(dst_ty.scalar(), src_ty.get_block_shape())

    if str(src_ty) == str(dst_ty):  # src_ty == dst_ty does not work as expected
        return input

    breakpoint()
    assert False

def _to_tensor(x, builder):
    if isinstance(x, int):
        return builder.get_int32(x)
    elif isinstance(x, float):
        return builder.get_fp32(x)
    elif isinstance(x, ir.value):
        return x
    assert False

def broadcast(lhs, rhs, builder):
    lhs_ty = lhs.get_type()
    rhs_ty = rhs.get_type()

    if lhs_ty.is_block() and not rhs_ty.is_block():
        rhs = builder.create_splat(rhs, lhs.get_type().get_block_shape())
    elif not lhs_ty.is_block() and rhs_ty.is_block():
        lhs = builder.create_splat(lhs, rhs.get_type().get_block_shape())
    elif lhs_ty.is_block() and rhs_ty.is_block():
        # TODO: don't support broadcasting for 2 block types yet
        assert lhs_ty.get_block_shape() == rhs_ty.get_block_shape()

    return lhs, rhs

def broadcast_and_type_promotion(input, other, builder):
    input = _to_tensor(input, builder)
    other = _to_tensor(other, builder)

    # do broadcast
    input, other = broadcast(input, other, builder)

    # TODO do type promotion
    return input, other

def get_module_op_from_value(value: ir.value) -> ir.module:
    return value.get_defining_op().get_containing_module()

# Begin builtin functions
def load(ptr, mask, builder):
    """
    Don't support block ptr. Only legacy load.
    """
    cache = ir.CACHE_MODIFIER.NONE
    eviction = ir.EVICTION_POLICY.NORMAL
    is_volatile = False

    assert ptr.get_type().is_block()
    assert ptr.get_type().scalar().is_ptr()

    # Don't support broadcasting yet
    assert mask.type().get_block_shape() == ptr.type().get_block_shape()
    ptr_ty =  ptr.type().scalar()
    elt_ty = ptr_ty.element_ty()

    if elt_ty.is_bool():
        assert False

    if not mask:
        assert False
    else:
        return builder.create_masked_load(ptr, mask, None, cache, eviction, is_volatile)

def store(ptr, val, mask, builder):
    cache = ir.CACHE_MODIFIER.NONE
    eviction = ir.EVICTION_POLICY.NORMAL
    assert ptr.get_type().is_block()
    assert ptr.get_type().scalar().is_ptr()

    assert val.type().get_block_shape() == ptr.type().get_block_shape()
    if mask:
        assert mask.type().get_block_shape() == ptr.type().get_block_shape()

    ptr_ty =  ptr.type().scalar()
    elt_ty = ptr_ty.element_ty()

    if elt_ty.is_bool():
        assert False

    # Cast to target data type
    val = cast(val, elt_ty, builder)

    # Build IR
    if not mask:
        assert False

    if not mask.type().scalar().is_bool():
        assert False

    return builder.create_masked_store(ptr, val, mask, cache, eviction)

def arange(start, end, builder):
    return builder.create_make_range(start, end)

def program_id(axis, builder):
    out = builder.create_get_program_id(axis) 
    return out

def mul(input, other, builder):
    input, other = broadcast_and_type_promotion(input, other, builder)
    scalar_ty = input.get_type().scalar()
    if scalar_ty.is_floating():
        return builder.create_fmul(input, other)
    elif scalar_ty.is_int():
        return builder.create_mul(input, other)
    assert False

def add(input, other, builder):
    input, other = broadcast_and_type_promotion(input, other, builder)
    input_scalar_ty = input.get_type().scalar()
    other_scalar_ty = other.get_type().scalar()
    if input_scalar_ty.is_ptr() and other_scalar_ty.is_ptr():
        raise ValueError("cannot add pointers together")

    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        assert False

    if input_scalar_ty.is_ptr():
        return builder.create_addptr(input, other)
    elif input_scalar_ty.is_floating():
        return builder.create_fadd(input, other)
    else:
        return builder.create_add(input, other)
    assert False

def less_than(input, other, builder):
    input, other = broadcast_and_type_promotion(input, other, builder)
    scalar_ty = input.get_type().scalar()
    if scalar_ty.is_floating():
        assert False
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            assert False
        else:
            return builder.create_icmpULT(input, other)
    assert False

def get_builtin(triton_builtin):
    if not isinstance(triton_builtin, str):
        assert getattr(triton_builtin, "__triton_builtin__", False), triton_builtin
        triton_builtin = triton_builtin.__name__
    return globals()[triton_builtin]
