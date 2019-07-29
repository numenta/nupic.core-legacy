# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2015, Numenta, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

"""This file builds and installs the HTM Core Python bindings."""
 
import glob
import os
import subprocess
import sys
import tempfile
import distutils.dir_util

from setuptools import Command, find_packages, setup
from setuptools.command.test import test as BaseTestCommand
from distutils.core import Extension


PY_BINDINGS = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(PY_BINDINGS, os.pardir, os.pardir, os.pardir))
DISTR_DIR = os.path.join(REPO_DIR, "build", "Release", "distr")
DARWIN_PLATFORM = "darwin"
LINUX_PLATFORM = "linux"
UNIX_PLATFORMS = [LINUX_PLATFORM, DARWIN_PLATFORM]
WINDOWS_PLATFORMS = ["windows"]



def getExtensionVersion():
  """
  Get version from local file.
  """
  with open(os.path.join(REPO_DIR, "VERSION"), "r") as versionFile:
    return versionFile.read().strip()



class CleanCommand(Command):
  """Command for cleaning up intermediate build files."""

  description = "Command for cleaning up generated extension files."
  user_options = []


  def initialize_options(self):
    pass


  def finalize_options(self):
    pass


  def run(self):
    platform = getPlatformInfo()
    files = getExtensionFileNames(platform)
    for f in files:
      try:
        os.remove(f)
      except OSError:
        pass



def fixPath(path):
  """
  Ensures paths are correct for linux and windows
  """
  path = os.path.abspath(os.path.expanduser(path))
  if path.startswith("\\"):
    return "C:" + path

  return path



def findRequirements(platform):
  """
  Read the requirements.txt file and parse into requirements for setup's
  install_requirements option.
  """
  requirementsPath = fixPath(os.path.join(PY_BINDINGS, "requirements.txt"))
  return [
    line.strip()
    for line in open(requirementsPath).readlines()
    if not line.startswith("#") 
  ]



class TestCommand(BaseTestCommand):
  user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]


  def initialize_options(self):
    BaseTestCommand.initialize_options(self)
    self.pytest_args = [] # pylint: disable=W0201


  def finalize_options(self):
    BaseTestCommand.finalize_options(self)
    self.test_args = []
    self.test_suite = True


  def run_tests(self):
    import pytest
    cwd = os.getcwd()
    # run python bindings tests (in /bindings/py/tests/)
    try:
      os.chdir(os.path.join(REPO_DIR, "bindings", "py", "tests"))
      errno = pytest.main(self.pytest_args)
    finally:
      os.chdir(cwd)
    if errno != 0:
      sys.exit(errno)
    
    # python tests (in /py/tests/)
    try:
      os.chdir(os.path.join(REPO_DIR, "py", "tests"))
      errno = pytest.main(self.pytest_args)
    finally:
      os.chdir(cwd)
    sys.exit(errno)



def getPlatformInfo():
  """Identify platform."""
  if "linux" in sys.platform:
    platform = "linux"
  elif "darwin" in sys.platform:
    platform = "darwin"
  # win32
  elif sys.platform.startswith("win"):
    platform = "windows"
  else:
    raise Exception("Platform '%s' is unsupported!" % sys.platform)
  return platform



def getExtensionFileNames(platform):
  # look for extension libraries in Repository/build/Release/distr/src/htm/bindings
  # library filenames:  
  #     htm.core.algorithms.so
  #     htm.core.engine.so
  #     htm.core.math.so
  #     htm.core.encoders.so
  #     htm.core.sdr.so
  if platform in WINDOWS_PLATFORMS:
    libExtension = "pyd"
  else:
    libExtension = "so"
  libNames = ("sdr", "encoders", "algorithms", "engine_internal", "math")
  libFiles = ["htm.bindings.{}.{}".format(name, libExtension) for name in libNames]
  files = [os.path.join(DISTR_DIR, "src", "htm", "bindings", name)
           for name in list(libFiles)]
  return files


def getExtensionFiles(platform):
  files = getExtensionFileNames(platform)
  for f in files:
    if not os.path.exists(f):
      generateExtensions(platform)
      break

  return files



def generateExtensions(platform):
  """
  This will perform a full Release build with default arguments.
  The CMake build will copy everything in the Repository/bindings/py/packaging 
  directory to the distr directory (Repository/build/Release/distr)
  and then create the extension libraries in Repository/build/Release/distr/src/nupic/bindings.
  Note: for Windows it will force a X64 build.
  """
  cwd = os.getcwd()
  
  from sys import version_info
  if version_info > (3, 0):
    # Build a Python 3.x library
    PY_VER = "-DBINDING_BUILD=Python3"
  else:
    # Build a Python 2.7 library
    PY_VER = "-DBINDING_BUILD=Python2"

  print("Python version: {}\n".format(sys.version))

  scriptsDir = os.path.join(REPO_DIR, "build", "scripts")
  try:
    if not os.path.isdir(scriptsDir):
      os.makedirs(scriptsDir)
    os.chdir(scriptsDir)
    if platform == WINDOWS_PLATFORMS:
      subprocess.check_call(["cmake", PY_VER, REPO_DIR, "-A", "x64"])
    else:
      subprocess.check_call(["cmake", PY_VER, REPO_DIR])
    subprocess.check_call(["cmake", "--build", ".", "--target", "install", "--config", "Release"])
  finally:
    os.chdir(cwd)



if __name__ == "__main__":
  platform = getPlatformInfo()

  if platform == DARWIN_PLATFORM and not "ARCHFLAGS" in os.environ:
    raise Exception("To build HTM.Core bindings in OS X, you must "
                    "`export ARCHFLAGS=\"-arch x86_64\"`.")

  # Run CMake if extension files are missing.
  getExtensionFiles(platform)

  # Copy the python code into place. (from /py/htm/)
  distutils.dir_util.copy_tree(
            os.path.join(REPO_DIR, "py", "htm"), os.path.join(DISTR_DIR, "src", "htm"))
  """
  set the default directory to the distr, and package it.
  """
  os.chdir(DISTR_DIR)

  print("\nbindings/py/setup.py: Setup Pybind11 Python module in " + DISTR_DIR+ "\n")
  setup(
    # See https://docs.python.org/2/distutils/apiref.html for descriptions of arguments.
    #     https://docs.python.org/2/distutils/setupscript.html
    #     https://opensourceforu.com/2010/OS/extending-python-via-shared-libraries
    #     https://docs.python.org/3/library/ctypes.html
    #     https://docs.python.org/2/library/imp.html
    name="htm.core",
    version=getExtensionVersion(),
    # This distribution contains platform-specific C++ libraries, but they are not
    # built with distutils. So we must create a dummy Extension object so when we
    # create a binary file it knows to make it platform-specific.
    ext_modules=[Extension('htm.dummy', sources = ['dummy.c'])],
    package_dir = {"": "src"},
    packages=find_packages("src"),
    namespace_packages=["htm"],
    install_requires=findRequirements(platform),
    package_data={
        "htm.bindings": ["*.so", "*.pyd"],
        "htm.examples": ["*.csv"],
    },
    extras_require = {},
    zip_safe=False,
    cmdclass={
      "clean": CleanCommand,
      "test": TestCommand,
    },
    description="Python package for htm.core.",
    author="Numenta & HTM Community",
    author_email="help@numenta.org",
    url="https://github.com/htm-community/htm.core",
    long_description = "HTM Community Edition of Numenta's Platform for Intelligent Computing (NuPIC)",
    license = "GNU Affero General Public License v3 or later (AGPLv3+)",
    classifiers=[
      "Programming Language :: Python",
      "Programming Language :: Python :: 2",
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: POSIX :: Linux",
      "Operating System :: Microsoft :: Windows",
      # It has to be "5 - Production/Stable" or else pypi rejects it!
      "Development Status :: 5 - Production/Stable",
      "Environment :: Console",
      "Intended Audience :: Science/Research",
      "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    entry_points = {
      "console_scripts": [
        "htm-bindings-check = htm.bindings.check:checkMain",
      ],
    },
  )
  print("\nbindings/py/setup.py: Setup complete.\n")

