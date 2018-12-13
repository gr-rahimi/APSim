from distutils.core import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension


setup(
    name="My hello app",
    ext_modules=cythonize("point_comperator.pyx", include_path=['-I ~/anaconda2/include/python2.7']),
)