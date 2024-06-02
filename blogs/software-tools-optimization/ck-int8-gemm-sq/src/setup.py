import os
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension

os.environ["CC"] = "hipcc"
os.environ["CXX"] = "hipcc"

sources = [
    'torch_int/kernels/linear.cpp',
    'torch_int/kernels/bmm.cpp',
    'torch_int/kernels/pybind.cpp', 
]

include_dirs = ['torch_int/kernels/include']
extra_link_args = ['libutility.a']
extra_compile_args = ['-O3','-DNDEBUG', '-std=c++17', '--offload-arch=gfx942', '-DCK_ENABLE_FP32', '-DCK_ENABLE_FP16', '-DCK_ENABLE_INT8', '-D__HIP_PLATFORM_AMD__=1']

setup(
    name='torch_int',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='torch_int.rocm',
            sources=sources,
            include_dirs=include_dirs,
            extra_link_args=extra_link_args,
            extra_compile_args=extra_compile_args
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)
