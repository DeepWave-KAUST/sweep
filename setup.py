from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

def get_sources():
    return (
        glob.glob("src/sweep/csrc/common/**/*.cu", recursive=True) +
        glob.glob("src/sweep/csrc/equations/**/*.cu", recursive=True) +
        ["src/sweep/csrc/bindings.cpp"]
    )

setup(
    name="sweep",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["sweep", "sweep.*"],
        exclude=["sweep.csrc", "sweep.csrc.*"],
    ),
    ext_modules=[
        CUDAExtension(
            name="sweep._C",
            sources=get_sources(),
            include_dirs=[
                os.path.join(this_dir, "src/sweep/csrc"),
                os.path.join(this_dir, "src/sweep/csrc/common"),
                os.path.join(this_dir, "src/sweep/csrc/equations"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-attributes"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--threads=16",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True)
    }
)