import shutil
import tempfile
import os
import subprocess
import torch
import re

from . import _C

def get_ptxas_path():
    """
    Use the default ptx found in PATH.
    """
    return os.environ.get("PTXAS_PATH", shutil.which("ptxas"))

def ptx_to_cubin(ptx_code, arch=None):
    if arch is None:
        arch = parse_arch_from_ptx_code(ptx_code)
    assert arch is not None
    with tempfile.TemporaryDirectory() as tmpdir:
        ptx_path = os.path.join(tmpdir, "input.ptx")
        cubin_path = os.path.join(tmpdir, "output.cubin")
        log_path = os.path.join(tmpdir, "ptxas.log")

        with open(ptx_path, "w") as f:
            f.write(ptx_code)

        # run ptxas command
        cmd = f"{get_ptxas_path()} -arch={arch} {ptx_path} -o {cubin_path} >{log_path} 2>&1"
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            with open(log_path) as f:
                print(f"ptxas log:\n{f.read()}")
            raise

        # read the cubin bytes
        with open(cubin_path, "rb") as f:
            return f.read()

def launch(cu_func, gridDim, blockDim, args, shared=0):
    def _process_dim(dim):
        if isinstance(dim, int):
            dim = [dim]

        dim = list(dim)
        while len(dim) < 3:
            dim.append(1)
        return dim

    gridDim = _process_dim(gridDim)
    blockDim = _process_dim(blockDim)
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            args[i] = arg.data_ptr()
        else:
            assert isinstance(arg, int), "Argument must be a torch.Tensor or int"

    _C.launch(cu_func, gridDim, blockDim, args, shared)

def parse_arch_from_ptx_code(ptx_code):
    """
    Expect the ptx code specify the arch with a line like
        .target sm_80
    """
    all_matches = list(re.finditer(r"\.target\s+(\w+)", ptx_code))
    assert len(all_matches) == 1
    m = all_matches[0]
    assert m
    return m.group(1)

def parse_entry_from_ptx_code(ptx_code):
    """
    This function parse the .entry function name from the ptx code so users
    don't need to specify it.

    The function has the following limitations:
    1. the ptx code must contain a single .entry function
    2. the function name must follow .entry immediately. Other stuffs between
       .entry directive and the function name will make it not work.
       E.g. this function does not handle the following case:

       .entry
       // arbitrary comments
       func_name {
         ...
       }

    In case this function can not correctly find the .entry function name,
    one can always pass in the function name explictly.
    """
    all_matches = list(re.finditer(r"\.entry\s+(\w+)", ptx_code))
    assert len(all_matches) == 1
    m = all_matches[0]
    assert m
    return m.group(1)

def load_cu_func_from_ptx_code(ptx_code, func_name=None):
    if func_name is None:
        func_name = parse_entry_from_ptx_code(ptx_code)
    assert func_name is not None
    cubin_bytes = ptx_to_cubin(ptx_code)
    cu_func = _C.load_cufunc_from_bytes(cubin_bytes, func_name)
    return cu_func

def load_and_run(ptx_code, gridDim, blockDim, args, func_name=None, shared=0):
    """
    Main API. Load the kernel from ptx code and launch it with the provided
    arguments.

    This API is not suitable for perf test since the `load_cu_func_from_ptx_code`
    function takes a long time to run.

    For perf testing, save the cu_func and measure the execution time of `launch`
    function only.
    """
    cu_func = load_cu_func_from_ptx_code(ptx_code, func_name)
    launch(cu_func, gridDim=gridDim, blockDim=blockDim, shared=shared, args=args)
