import os
import subprocess

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext


def flag_dict(flags):
    d = {}
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in flags.split():
        d.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return d


if 'IPOPT_CFLAGS' in os.environ:
    flags = os.environ.get('IPOPT_CFLAGS')
    print("Using compilation flags given in IPOPT_CFLAGS environment variable:")
    print("\t", flags)
else:
    try:
        cmd = ['pkg-config', '--libs', '--cflags', 'ipopt']
        flags = subprocess.check_output(cmd).decode()
        print("Using compilation flags given by pkg-config:")
        print("\t", flags)
    except subprocess.CalledProcessError:
        print("No compilation flags found.")
        flags = ''


ipopt_cfg = flag_dict(flags)
wrapper_ext = Extension("ipopt.wrapper", ["ipopt/wrapper.pyx"], **ipopt_cfg)

setup(name="ipopt",
      version='0.1',
      description='Python bindings for IPOPT nonlinear optimization solver.',
      author='Dimas Abreu Dutra',
      author_email='dimasadutra@gmail.com',
      url='http://github.com/dimasad/python-ipopt',
      test_suite='nose.collector',
      tests_require=['nose>=1.0'],
      install_requires='distribute',
      packages=find_packages(),
      cmdclass={"build_ext": build_ext}, ext_modules=[wrapper_ext])
