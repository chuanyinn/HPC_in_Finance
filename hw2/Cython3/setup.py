from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="assignment2",
        sources=["assignment2.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-g"],
        extra_link_args=["-fopenmp", "-g"]
    )
]

setup(
    name='Tree with Cython 3',
    ext_modules = cythonize(extensions)
)