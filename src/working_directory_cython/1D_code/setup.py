#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:00:44 2016

@author: JLucero
"""

from distutils.core import setup, Extension
from Cython.Build import build_ext
import numpy.distutils.misc_util

ext = [
    Extension(
        'fpe_1d',['fpe_1d.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        ),
    Extension(
        'utilities_1d',['utilities_1d.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        )
    ]

ext_parallel = [
    Extension(
        'fpe', ['fpe.pyx'],
        extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
        extra_link_args=['-fopenmp', '-lm']
    )
    ]

setup(
    name="FPE_1D",
    version="0.1",
    ext_modules=ext,
    cmdclass={'build_ext': build_ext}
    )
