from ._curun import open as openCubin, sym as findSym, run as runKernel
import torch
from .compile_utils import compile_cuda, compile_ptx

def padDim3(dims):
    if isinstance(dims, int):
        dims = [dims]
    dims = list(dims)
    while len(dims) < 3:
        dims.append(1)
    return dims

class CuFunction:
    def __init__(self, cu_function):
        self.cu_function = cu_function

    def __getitem__(self, launch_meta):
        """
        Handle launching metadata like #block, #thread, #shared, stream
        """
        assert len(launch_meta) >= 2 and len(launch_meta) <= 4
        grid, block = launch_meta[:2]
        shared = stream = 0
        if len(launch_meta) >= 3:
            shared = launch_meta[2]

        if len(launch_meta) >= 4:
            stream = launch_meta[3]

        return lambda *args: self.run(args, grid=grid, block=block, shared=shared, stream=stream)

    def run(self, params, *, grid, block, shared, stream):
        gridX, gridY, gridZ = padDim3(grid)
        blockX, blockY, blockZ = padDim3(block)

        params = list(params)
        for i, x in enumerate(params):
            if isinstance(x, torch.Tensor):
                params[i] = x.data_ptr()

        runKernel(self.cu_function, gridX, gridY, gridZ, blockX, blockY, blockZ, shared, stream, params)

class CuModule:
    def __init__(self, cu_module):
        self.cu_module = cu_module

    def sym(self, name):
        return CuFunction(findSym(self.cu_module, name))

def open(path, use_cutlass=False):
    if path.endswith(".cubin"):
        cubinpath = path
    elif path.endswith(".cu"):
        cubinpath = compile_cuda(path, use_cutlass=use_cutlass)
    elif path.endswith(".ptx"):
        cubinpath = compile_ptx(path)
    else:
        raise RuntimeError(f"Unrecognized path for curun.open {path}")
    return CuModule(openCubin(cubinpath))

def run(*args, **kwargs):
    return runKernel(*args, **kwargs)
