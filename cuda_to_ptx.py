"""
A small tool to show the compiled ptx code by nvcc for the passed in cuda kernel.

This tool basically makes it convenient to do what's done in this numba GH issue: https://github.com/numba/numba/issues/6183
"""

import tempfile
import os
import shutil
import subprocess

KEEP_TMP_DIR = os.environ.get("KEEP_TMP_DIR") == "1"

def write_to_file(content, path):
    with open(path, "w") as f:
        f.write(content)

def compile_cuda(cu_path, ptx_path, use_fast_math):
    use_fast_math_option = "--use_fast_math" if use_fast_math else ""
    cmd = f"nvcc {cu_path} --ptx {use_fast_math_option} -arch=sm_80 -o {ptx_path}"
    subprocess.check_call(cmd.split())

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def cuda_to_ptx(cuda_str, use_fast_math):
    try:
        tmpdir = tempfile.mkdtemp()
        if KEEP_TMP_DIR:
            print(f"Create a temporary directory at: {tmpdir}")

        cu_path = os.path.join(tmpdir, "inp.cu")
        ptx_path = os.path.join(tmpdir, "output.ptx")

        write_to_file(cuda_str, cu_path)
        compile_cuda(cu_path, ptx_path, use_fast_math)
        return read_file(ptx_path)
    finally:
        if not KEEP_TMP_DIR:
            shutil.rmtree(tmpdir)

if __name__ == "__main__":
    cuda_str = """
    __global__ void kernel_exp(float* p, float x) {
        p[0] = exp(x);
    }
    """

    ptx_str = cuda_to_ptx(cuda_str, use_fast_math=False)
    print(f"ptx generated:\n{ptx_str}")
