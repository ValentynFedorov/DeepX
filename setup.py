"""
Setup script for building the C++ extension module
"""
import sys
from setuptools import setup, Extension

# Define M_PI for MSVC compatibility
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/D_USE_MATH_DEFINES"]
    extra_link_args = []
else:  # Linux
    extra_compile_args = ["-O3", "-std=c++17"]
    extra_link_args = []

# Try to import pybind11, if not available provide helpful error
try:
    import pybind11
    include_dirs = [pybind11.get_include()]
except ImportError:
    print("ERROR: pybind11 is required to build this module")
    sys.exit(1)

ext_modules = [
    Extension(
        "deepx_core",
        sources=["cpp_module/geometry_engine.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="deepx_core",
    version="1.0.0",
    author="DeepX Candidate",
    description="C++ accelerated geometry operations for 3D reconstruction",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)