#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy
import subprocess
import os
import numpy  # noqa

# compile kernel/libG6K.so if not done already
subprocess.check_call("make -C kernel",shell=True)

# read actual values of all build variables from kernel/Makefile
makefile_defs = subprocess.getoutput("make -C kernel printvariables | grep '='").splitlines()

def read_from_makefile(field):
    global makefile_defs
    data = [line for line in makefile_defs if line.startswith(field)][0]
    data = '=' .join(data.split('=')[1:])
    data = data.strip()
    data = [arg for arg in data.split(' ') if arg.strip()]
    return data

extra_compile_args = read_from_makefile("CXXFLAGS")
extra_link_args = read_from_makefile("LDFLAGS") + read_from_makefile("LIBADD")

kwds = {
    "language": "c++",
    "extra_compile_args": extra_compile_args,
    "extra_link_args": extra_link_args,
    "libraries": ["gmp", "pthread", "G6K"],
    "include_dirs": [numpy.get_include(), "parallel-hashmap"]
    }

extensions = [
    Extension("g6k.siever", ["g6k/siever.pyx"], **kwds),
    Extension("g6k.siever_params", ["g6k/siever_params.pyx"], **kwds)
]

setup(
    name="G6K",
    version="0.0.1",
    ext_modules=cythonize(extensions, compiler_directives={'binding': True,
                                                           'embedsignature': True,
                                                           'language_level': 2}),
    packages=[],
)
