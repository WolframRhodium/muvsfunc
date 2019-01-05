# To build, run `python setup.py build_ext --inplace`

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [Extension(name="sigma_filter", sources=["sigma_filter.pyx"])]

setup(name='sigma_filter', ext_modules=cythonize(module_list=ext_modules, language_level=3))
