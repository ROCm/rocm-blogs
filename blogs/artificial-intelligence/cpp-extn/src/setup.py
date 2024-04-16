from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='mlp_cpp',
    ext_modules=[
        CppExtension('mlp_cpp', ['mlp.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })