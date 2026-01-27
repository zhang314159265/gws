from setuptools import setup, Extension
import os
import shutil
import pybind11

nvcc_path = shutil.which("nvcc")
CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))
CUDA_INCLUDE = os.path.join(CUDA_HOME, "include")

ext_modules = [
    Extension(
        name="curun._curun",
        sources=["curun.cpp"],
        include_dirs=[CUDA_INCLUDE, pybind11.get_include()],
        libraries=["cuda"],
    ),
]

setup(
    ext_modules=ext_modules
)
