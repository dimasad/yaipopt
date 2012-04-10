from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

ipopt_ext = Extension("ipopt", ["ipopt.pyx"], libraries=["ipopt"])
setup(name="ipopt",
      version='0.1',
      description='Python bindings for IPOPT nonlinear optimization solver.',
      author='Dimas Abreu Dutra',
      author_email='dimasadutra@gmail.com',
      url='http://github.com/dimasad/python-ipopt',
      cmdclass={"build_ext": build_ext}, ext_modules=[ipopt_ext])
