import commands

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext


# The function bellow was based in the following recipe:
# http://code.activestate.com/recipes/502261-python-distutils-pkg-config/
def pkgconfig(*pakages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    pkg_names = ' '.join(pakages)
    output = commands.getoutput("pkg-config --libs --cflags %s" % pkg_names)
    for token in output.split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


try:
    lipipopt_cfg = pkgconfig('ipopt')
except:
    lipipopt_cfg = {}

ipopt_ext = Extension("ipopt", ["ipopt.pyx"], **lipipopt_cfg)

setup(name="ipopt",
      version='0.1',
      description='Python bindings for IPOPT nonlinear optimization solver.',
      author='Dimas Abreu Dutra',
      author_email='dimasadutra@gmail.com',
      url='http://github.com/dimasad/python-ipopt',
      install_requires='distribute',
      cmdclass={"build_ext": build_ext}, ext_modules=[ipopt_ext])
