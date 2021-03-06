import os
import subprocess

from setuptools import Extension, setup, find_packages


DESCRIPTION = open("README.rst").read()

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
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering
Topic :: Software Development'''


def ipopt_opts():
    # Based on code from
    # http://code.activestate.com/recipes/502261-python-distutils-pkg-config/
    if 'IPOPT_FLAGS' in os.environ:
        flags = os.environ.get('IPOPT_FLAGS')
        print("Using compilation flags given in environment variable:")
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
    opts = {}
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in flags.split():
        opt_name = flag_map.get(token[:2])
        if opt_name:
            opts.setdefault(opt_name, []).append(token[2:])
    return opts


wrapper = Extension("yaipopt.wrapper", ["yaipopt/wrapper.pyx"], **ipopt_opts())

try:
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
except ImportError:
    cmdclass = {}


try:
    import numpy
    include_dirs = [numpy.get_include()]
except ImportError:
    include_dirs = []


setup(
    name="yaipopt",
    version='0.1.dev2',
    test_suite='nose.collector',
    tests_require=['nose>=1.0'],
    install_requires=['cython', 'numpy', 'setuptools'],
    packages=find_packages(),
    cmdclass=cmdclass,
    include_dirs=include_dirs,
    ext_modules=[wrapper],
    
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
