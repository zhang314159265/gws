import triton
from triton._C.libtriton import ir
from triton.runtime.driver import driver
from triton.compiler.compiler import make_backend

ttir_path = "skip-checkin//add_ref.ttir"
target = driver.active.get_current_target()
backend = make_backend(target)
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)
module = ir.parse_mlir_module(ttir_path, context)
print(module)
print("bye")
