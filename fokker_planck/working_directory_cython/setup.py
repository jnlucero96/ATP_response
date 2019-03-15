#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:00:44 2016

@author: JLucero
"""

from distutils.core import setup, Extension
from Cython.Build import build_ext


ext = [
    Extension(
        'fpe',['fpe.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        ),
    Extension(
        'utilities',['utilities.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        )
    ]

ext_parallel = [
    Extension(
        'fpe', ['fpe.pyx'],
        extra_compile_args=["-Ofast", "-march=native", "-Wall", "-fopenmp"],
        extra_link_args=['-fopenmp', '-lm']
    )
    ]

setup(
    name="FPE",
    version="1.0",
    ext_modules=ext,
    cmdclass={'build_ext': build_ext}
    )
