import runpy
from . import layernorm_bwd

# works when runing "python -m" in tutor folder
# doen't work when do that in gws/ folder.
print("Layernorm bwd:")
runpy.run_module("norm_bwd.layernorm_bwd", run_name="__main__")
print("rmsnorm bwd:")
runpy.run_module("norm_bwd.rmsnorm_bwd", run_name="__main__")
