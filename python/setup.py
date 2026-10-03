"""pip install ./python  (from the BRISK source directory)

Builds the shared library libbrisk with the BRISK Makefile (`make libbrisk`, in the source
directory one level up) and puts a copy inside the package, so that the installed package does
not depend on the source tree. The metadata is in pyproject.toml.

Environment:
  BRISK_MAKEFLAGS   extra make arguments, e.g. "BLAS=scipy" or "BLASLIB=-lopenblas" or "OMP=0"
  BRISK_NO_BUILD=1  do not run make: use the libbrisk already built in the source directory
                    (or none: the package then needs BRISK_LIBRARY at run time)
  MAKE              the make program (default: make)

An editable install (pip install -e ./python) does not copy the library: the package finds
libbrisk in the source directory (run `make libbrisk` there first) or at BRISK_LIBRARY.
"""
import os
import shlex
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
LIBNAME = "libbrisk." + ("dylib" if sys.platform == "darwin" else "so")


class BuildPyWithLibbrisk(build_py):
    def run(self):
        super().run()
        lib = os.path.join(SRC, LIBNAME)
        if not os.environ.get("BRISK_NO_BUILD"):
            if not os.path.exists(os.path.join(SRC, "capi.c")):
                raise RuntimeError(f"the BRISK sources are not in {SRC}: install from the BRISK "
                                   "source directory (pip install ./python), or set BRISK_NO_BUILD=1 "
                                   "and BRISK_LIBRARY")
            cmd = [os.environ.get("MAKE", "make"), "-C", SRC, "libbrisk"]
            cmd += shlex.split(os.environ.get("BRISK_MAKEFLAGS", ""))
            print("building libbrisk:", " ".join(cmd), flush=True)
            subprocess.check_call(cmd)
        if os.path.exists(lib):
            dest = os.path.join(self.build_lib, "brisk")
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(lib, os.path.join(dest, LIBNAME))
        elif not os.environ.get("BRISK_NO_BUILD"):
            raise RuntimeError(f"make libbrisk did not produce {lib}")


class BinaryDistribution(Distribution):
    """The package carries a compiled library: platform-specific wheels."""

    def has_ext_modules(self):
        return True


setup(cmdclass={"build_py": BuildPyWithLibbrisk}, distclass=BinaryDistribution)
