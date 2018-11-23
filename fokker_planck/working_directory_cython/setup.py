# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:00:44 2016

@author: JLucero
"""

from distutils.core import setup, Extension
from Cython.Build import build_ext
import numpy.distutils.misc_util

include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()

ext = [
    Extension(
        'fpe',['fpe.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=["-O3","-ffast-math", "-v", "-march=native"]
        )
    ]

ext_parallel = [
    Extension(
        'fpe', ['fpe.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math", "-fopenmp", "-v"],
        extra_link_args=['-fopenmp', '-lm']
    )
    ]

setup(
    name="FPE",
    version="0.2",
    ext_modules=ext,
    cmdclass={'build_ext': build_ext}
    )