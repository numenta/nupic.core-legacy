# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""This file builds and installs the NuPIC Core Python bindings."""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import Command, find_packages, setup
from setuptools.command.test import test as BaseTestCommand


PY_BINDINGS = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(PY_BINDINGS, os.pardir, os.pardir))
DARWIN_PLATFORM = "darwin"
LINUX_PLATFORM = "linux"
UNIX_PLATFORMS = [LINUX_PLATFORM, DARWIN_PLATFORM]
WINDOWS_PLATFORMS = ["windows"]



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
  includePycapnp = platform not in WINDOWS_PLATFORMS
  requirementsPath = fixPath(os.path.join(PY_BINDINGS, "requirements.txt"))
  return [
    line.strip()
    for line in open(requirementsPath).readlines()
    if not line.startswith("#") and (not line.startswith("pycapnp") or includePycapnp)
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
    try:
      os.chdir("tests")
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
  if platform in WINDOWS_PLATFORMS:
    libExtension = "dll"
  else:
    libExtension = "so"
  libNames = ("algorithms", "engine_internal", "math")
  swigPythonFiles = ["{}.py".format(name) for name in libNames]
  swigLibFiles = ["_{}.{}".format(name, libExtension) for name in libNames]
  files = [os.path.join(PY_BINDINGS, "nupic", "bindings", name)
           for name in list(swigPythonFiles + swigLibFiles)]
  return files



def getExtensionFiles(platform):
  files = getExtensionFileNames(platform)
  for f in files:
    if not os.path.exists(f):
      generateExtensions()
      break

  return files



def generateExtensions():
  tmpDir = tempfile.mkdtemp()
  cwd = os.getcwd()
  try:
    scriptsDir = os.path.join(tmpDir, "scripts")
    releaseDir = os.path.join(tmpDir, "release")
    pyExtensionsDir = os.path.join(PY_BINDINGS, "nupic", "bindings")
    os.mkdir(scriptsDir)
    os.chdir(scriptsDir)
    subprocess.check_call(
        ["cmake", REPO_DIR, "-DCMAKE_INSTALL_PREFIX={}".format(releaseDir),
         "-DPY_EXTENSIONS_DIR={}".format(pyExtensionsDir)])
    subprocess.check_call(["make", "-j3"])
    subprocess.check_call(["make", "install"])
  finally:
    shutil.rmtree(tmpDir, ignore_errors=True)
    os.chdir(cwd)



if __name__ == "__main__":
  platform = getPlatformInfo()

  if platform == DARWIN_PLATFORM and not "ARCHFLAGS" in os.environ:
    raise Exception("To build NuPIC Core bindings in OS X, you must "
                    "`export ARCHFLAGS=\"-arch x86_64\"`.")

  # Run CMake if extension files are missing.
  getExtensionFiles(platform)

  # Copy the proto files into the proto Python package.
  destDir = os.path.relpath(os.path.join("nupic", "proto"))
  for protoPath in glob.glob(os.path.relpath(os.path.join(
      "..", "..", "src", "nupic", "proto", "*.capnp"))):
    shutil.copy(protoPath, destDir)

  print "\nSetup SWIG Python module"
  setup(
    name="nupic.bindings",
    version="0.2.2",
    namespace_packages=["nupic"],
    install_requires=findRequirements(platform),
    packages=find_packages(),
    package_data={
        "nupic.proto": ["*.capnp"],
        "nupic.bindings": ["*.so", "*.dll"],
    },
    extras_require = {"capnp": ["pycapnp==0.5.5"]},
    zip_safe=False,
    cmdclass={
      "clean": CleanCommand,
      "test": TestCommand,
    },
    description="Numenta Platform for Intelligent Computing - bindings",
    author="Numenta",
    author_email="help@numenta.org",
    url="https://github.com/numenta/nupic.core",
    long_description = "Python bindings for nupic core.",
    classifiers=[
      "Programming Language :: Python",
      "Programming Language :: Python :: 2",
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
  )
