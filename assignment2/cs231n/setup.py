from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Build import cythonize

    use_cython = True
except ModuleNotFoundError:
    use_cython = False

source = "im2col_cython.pyx" if use_cython else "im2col_cython.c"

extensions = [
    Extension(
        "im2col_cython", [source], include_dirs=[numpy.get_include()]
    ),
]

ext_modules = cythonize(extensions) if use_cython else extensions
setup(ext_modules=ext_modules)
