import os
import subprocess

H100_ARCH = "sm_90a"

def _compile_file(inpath, suffix, arch):
    assert inpath.endswith(suffix)
    cubinpath = os.path.basename(inpath).removesuffix(suffix) + ".cubin"
    cubinpath = os.path.join("/tmp", cubinpath)
    # print(f"{cubinpath=}")

    cmd = f"""
        nvcc -cubin -I. {inpath} -o {cubinpath} -arch={arch}
    """.strip()

    subprocess.run(cmd.split(), check=True)
    return cubinpath


def compile_cuda(cudapath, arch=H100_ARCH):
    return _compile_file(cudapath, ".cu", arch)

def compile_ptx(ptxpath, arch=H100_ARCH):
    return _compile_file(ptxpath, ".ptx", arch)
