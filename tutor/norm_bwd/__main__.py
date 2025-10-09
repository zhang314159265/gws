import runpy
from . import layernorm_bwd

# works when runing "python -m" in tutor folder
# doen't work when do that in gws/ folder.
runpy.run_module("norm_bwd.layernorm_bwd", run_name="__main__")
runpy.run_module("norm_bwd.rmsnorm_bwd", run_name="__main__")
