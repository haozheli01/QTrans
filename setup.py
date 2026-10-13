import os
import numpy as np
from setuptools import Extension, setup
import sysconfig

def native_extension():
    sources = [
        os.path.join("c", "numpy_extension.c"),
        os.path.join("c", "utils_matrix.c"),
    ]

    include_dirs = [np.get_include(), sysconfig.get_path("include")]
    # Match the optimized QTrans20260721 native build.  ``-march=native`` is
    # evaluated by the compiler on the build host; callers can remove it when
    # building a portable wheel for heterogeneous nodes.
    compile_args = ["-std=gnu99", "-O3", "-march=native", "-funroll-loops", "-fopenmp"]
    link_args = ["-fopenmp", "-lgomp", "-lpthread"]

    return Extension(
        "qtrans._native",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )


setup(ext_modules=[native_extension()])
