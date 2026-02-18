from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
this_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Collect all C++ / CUDA files
# ----------------------------
sources = (
    glob.glob("src/sweep/csrc/**/*.cpp", recursive=True) +
    glob.glob("src/sweep/csrc/**/*.cu", recursive=True) +
    glob.glob("src/sweep/equations/**/*.cu", recursive=True)
)

print("Compiling sources:")
for s in sources:
    print("  ", s)

# ----------------------------
# Compile flags
# ----------------------------
extra_compile_args = {"cxx": ["-O3"]}

if os.name != "nt":
    extra_compile_args["cxx"] += ["-Wno-sign-compare"]

nvcc_flags = os.getenv("NVCC_FLAGS", "")
nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
nvcc_flags += ["-O3", "--use_fast_math"]

extra_compile_args["nvcc"] = nvcc_flags

# ----------------------------
# Setup
# ----------------------------
setup(
    name="sweep",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[
        CUDAExtension(
            name="sweep._C",
            sources=sources,
            include_dirs=[
                os.path.join(this_dir, "src/sweep"),
                os.path.join(this_dir, "src/sweep/csrc"),
                os.path.join(this_dir, "src/sweep/operators"),
                os.path.join(this_dir, "src/sweep/equations"),
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)}
)