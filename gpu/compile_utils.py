import os
import subprocess
import torch

H100_ARCH = "sm_90a"
B200_ARCH = "sm_100"

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

def detect_arch(arch):
    if arch:
        return arch

    device_name = torch.cuda.get_device_properties(0).name
    if device_name == "NVIDIA B200":
        return B200_ARCH
    elif device_name == "NVIDIA H100":
        return H100_ARCH
    else:
        raise RuntimeError(f"Unrecognized device name {device_name}")

def compile_cuda(cudapath, arch=None):
    arch = detect_arch(arch)
    return _compile_file(cudapath, ".cu", arch)

def compile_ptx(ptxpath, arch=None):
    arch = detect_arch(arch)
    return _compile_file(ptxpath, ".ptx", arch)
