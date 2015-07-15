import os
import subprocess

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext


DESCRIPTION = open("README.rst", encoding="utf-8").read()

CLASSIFIERS = '''\
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development'''


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


def flag_dict(flags):
    d = {}
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in flags.split():
        d.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return d


ipopt_cfg = flag_dict(flags)
wrapper_ext = Extension("yaipopt.wrapper", ["yaipopt/wrapper.pyx"], **ipopt_cfg)


setup(
    name="yaipopt",
    version='0.1.dev1',
    test_suite='nose.collector',
    tests_require=['nose>=1.0'],
    install_requires=['cython', 'distribute', 'numpy'],
    packages=find_packages(),
    cmdclass={"build_ext": build_ext}, ext_modules=[wrapper_ext],
    
    # metadata for upload to PyPI
    author='Dimas Abreu Dutra',
    author_email='dimasad@ufmg.br',
    description='Python bindings for IPOPT nonlinear optimization solver.',
    long_description=DESCRIPTION,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=CLASSIFIERS.split('\n'),
    license="MIT",
    url='http://github.com/dimasad/yaipopt',
)
