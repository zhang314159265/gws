# XXX: don't import triton in this file!
import ast
import inspect
from typing import Tuple, Dict, Any
from tritoncc._C import ir
import torch

from frontend.builtin import get_builtin
import ttirrunner
import ptxrunner

def _is_list_like(o: Any) -> bool:
    return isinstance(o, (list, tuple))

# TODO: I don't want to define a tensor class like tl.tensor
class tensor:
    pass

class CodeGenerator(ast.NodeVisitor):
    def __init__(self, context, builder, func_mlir_type, gscope):
        self.context = context
        self.builder = builder
        self.func_mlir_type = func_mlir_type
        self.module = self.builder.create_module()
        self.gscope = gscope
        self.lscope = {}
        self.local_defs: Dict[str, tensor] = {}
        self.global_uses: Dict[str, tensor] = {}

        self.fn = None
        self.function_name = "triton_kernel"
        self.is_kernel = True
        self.noinline = False

    builtin_namespace: Dict[str, Any] = {}

    def generic_visit(self, node):
        print(f"Unsupported AST node type: {type(node).__name__}\n{ast.dump(node) if node is not None else 'None'}")
        breakpoint()
        assert False

    def visit_NoneType(self, node):
        return None

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def dereference_name(self, name):
        def local_lookup(name: str, absent):
            value = self.lscope.get(name, absent)
            if value is not absent and name not in self.local_defs:
                self.global_uses[name] = value
            return value

        absent = object()

        for lookup_function in local_lookup, self.gscope.get, self.builtin_namespace.get:
            value = lookup_function(name, absent)
            if value is not absent:
                return value
        raise NameError(f"{name} is not defined")

    def set_value(self, name: str, value) -> None:
        self.lscope[name] = value
        self.local_defs[name] = value

    def visit_Compare(self, node):
        if len(node.comparators) != 1 or len(node.ops) != 1:
            assert False
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        if type(node.ops[0]) == ast.Is:
            assert False
        if type(node.ops[0]) == ast.IsNot:
            assert False
        method_name = self._method_name_for_comp_op.get(type(node.ops[0]))
        assert method_name, ast.dump(node)
        return self._apply_binary_method(method_name, lhs, rhs)

    _method_name_for_comp_op = {
        ast.Lt: "less_than",
    }

    def visit_keyword(self, node) -> Tuple[str, Any]:
        return node.arg, self.visit(node.value)

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            return node.id
        return self.dereference_name(node.id)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        return getattr(lhs, node.attr)

    def visit_arguments(self, node):
        arg_names = []
        for arg in node.args:
            arg_names += [self.visit(arg)]
        kwarg_names = self.visit(node.kwarg)
        return arg_names, kwarg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def _unsupported(self, node, message):
        return RuntimeError(f"Unsupported: {node=}, {message=}")

    def _apply_binary_method(self, method_name, lhs, rhs):
        return get_builtin(method_name)(lhs, rhs, builder=self.builder)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self._method_name_for_bin_op.get(type(node.op))
        assert method_name, node.op
        return self._apply_binary_method(method_name, lhs, rhs)

    _method_name_for_bin_op = {
        ast.Add: "add",
        ast.Mult: "mul",
    }

    def visit_Constant(self, node):
        return node.value

    def visit_Call(self, node):
        fn = self.visit(node.func)
        kws = dict(self.visit(keyword) for keyword in node.keywords)
        args = [self.visit(arg) for arg in node.args]

        return get_builtin(fn)(*args, builder=self.builder, **kws)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        _names = []
        for target in node.targets:
            _names += [self.visit(target)]

        if len(_names) > 1:
            raise self._unsupported(node, "simultaneous multiple assignment is not supported.")
        names = _names[0]
        values = self.visit(node.value)
        if not _is_list_like(names):
            names = [names]
        if not _is_list_like(values):
            values = [values]
        for name, value in zip(names, values):
            self.set_value(name, value)

    def visit_compound_statement(self, stmts):
        if not _is_list_like(stmts):
            stmts = [stmts]
        for stmt in stmts:
            self.visit(stmt)

            if isinstance(stmt, ast.Return):
                break

    def visit_FunctionDef(self, node: ast.FunctionDef) -> bool:
        arg_names, kwarg_names = self.visit(node.args)
        print(f"{arg_names=} {kwarg_names=}")
        assert not self.fn, "nested function definition is not supported."

        # initialize defaults
        for i, default_value in enumerate(node.args.defaults):
            assert False, "Not support default value yet"

        # initialize function
        visibility = "public" if self.is_kernel else "private"
        self.fn = self.builder.get_or_insert_function(self.module, self.function_name, self.func_mlir_type, visibility, self.noinline)
        self.module.push_back(self.fn)
        entry = self.fn.add_entry_block()
        arg_values = []

        for i, arg_name in zip(range(self.fn.num_args()), arg_names):
            arg_values.append(self.fn.args(i))
       
        insert_pt = self.builder.get_insertion_block()
        for arg_name, arg_value in zip(arg_names, arg_values):
            self.set_value(arg_name, arg_value)
        self.builder.set_insertion_point_to_start(entry)
        # visit function body
        self.visit_compound_statement(node.body)

        # finalize function
        # return void
        self.builder.ret([])
        if insert_pt:
            self.builder.set_insertion_point_to_end(insert_pt)
        # Remove dead code
        self.fn.finalize()

def gen_ttir(fn, func_mlir_type, context, builder):
    gscope = fn.__globals__.copy()
    generator = CodeGenerator(context, builder, func_mlir_type, gscope=gscope) 
    src = inspect.getsource(fn)
    ast_mod = ast.parse(src)
    # print(ast.dump(ast_mod, indent=2))
    generator.visit(ast_mod)
    return generator.module

def _get_arg_mlir_type(builder, arg):
    if isinstance(arg, torch.Tensor):
        elem_ty = builder.get_float_ty()
        ptr_ty = builder.get_ptr_ty(elem_ty, 1)
        return ptr_ty
    elif isinstance(arg, int):
        return builder.get_int32_ty()
    else:
        assert False, f"{type(arg).__name__}"

def compile(fn):
    def compiled_fn(*args, **kwargs):
        # TODO cache for compiled_fn. It's critical to have reasonable perf
        # number for do_bench

        # XXX context need to be live before we are fully done with MLIR
        context = ir.context()
        ir.load_dialects(context)
        builder = ir.builder(context)
        arg_mlir_types = [
            _get_arg_mlir_type(builder, arg) for arg in args
        ]
        func_mlir_type = builder.get_function_ty(arg_mlir_types, [])
        print(f"{func_mlir_type=}")
        ir_mod = gen_ttir(fn, func_mlir_type, context, builder)
        ptx_code = ttirrunner.ttir_to_ptx(str(ir_mod))
        # TODO can we compute these automatically?
        gridDim = kwargs["gridDim"]
        blockDim = kwargs["blockDim"]
        shared = kwargs["shared"]
        ptxrunner.load_and_run(ptx_code, args=args, gridDim=gridDim, blockDim=blockDim, shared=shared)

    return compiled_fn
