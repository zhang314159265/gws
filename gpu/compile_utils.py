import os
import subprocess

H100_ARCH = "sm_90a"

def compile_cuda(cudapath, arch=H100_ARCH):
    assert cudapath.endswith(".cu")
    cubinpath = os.path.basename(cudapath).removesuffix(".cu") + ".cubin"
    cubinpath = os.path.join("/tmp", cubinpath)
    # print(f"{cubinpath=}")

    cmd = f"""
        nvcc -cubin -I. {cudapath} -o {cubinpath} -arch={arch}
    """.strip()

    subprocess.run(cmd.split(), check=True)
    return cubinpath
