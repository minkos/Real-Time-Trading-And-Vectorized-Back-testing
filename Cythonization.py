from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('tpqoa_trading_cython.pyx', language_level = "3"))