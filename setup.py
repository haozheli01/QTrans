import os
import sys

import numpy as np
from setuptools import Extension, setup


def native_extension():
    sources = [
        os.path.join("c", "numpy_extension.c"),
        os.path.join("c", "utils_matrix.c"),
    ]

    if sys.platform == "win32":
        compile_args = ["/O2", "/openmp"]
        link_args = []
    else:
        compile_args = ["-O3", "-fopenmp"]
        link_args = ["-fopenmp", "-lpthread"]

    return Extension(
        "qtrans._native",
        sources=sources,
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )


setup(ext_modules=[native_extension()])

