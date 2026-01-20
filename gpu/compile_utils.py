import os
import subprocess
import torch

H100_ARCH = "sm_90a"
B200_ARCH = "sm_100"

def _compile_file(inpath, suffix, arch, use_cutlass=False):
    """
    Need add the cutlass headers if use_cutlass is True
    """
    assert inpath.endswith(suffix)
    cubinpath = os.path.basename(inpath).removesuffix(suffix) + ".cubin"
    cubinpath = os.path.join("/tmp", cubinpath)
    # print(f"{cubinpath=}")

    extra_args = []
    if use_cutlass:
        cutlass_home = os.getenv("CUTLASS_HOME")
        if cutlass_home is None:
            raise RuntimeError("Please specify the home directory of cutalss with CUTLASS_HOME environment variable")
        extra_args.extend([
            f"-I {cutlass_home}/include",
            f"-I {cutlass_home}/tools/util/include"
        ])

    cmd = f"""
        nvcc -cubin -I. {inpath} {" ".join(extra_args)} -o {cubinpath} -arch={arch}
    """.strip()

    # print(cmd)

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

def compile_cuda(cudapath, arch=None, use_cutlass=False):
    arch = detect_arch(arch)
    return _compile_file(cudapath, ".cu", arch, use_cutlass=use_cutlass)

def compile_ptx(ptxpath, arch=None):
    arch = detect_arch(arch)
    return _compile_file(ptxpath, ".ptx", arch)
