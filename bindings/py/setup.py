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

import argparse
import glob
import numpy
import os
import shutil
import subprocess
import sys
import tempfile

from distutils import ccompiler
from setuptools import setup, find_packages, Extension

"""
This file builds and installs the NuPIC Core bindings
"""



PY_BINDINGS = os.path.dirname(os.path.realpath(__file__))
DARWIN_PLATFORM = "darwin"
LINUX_PLATFORM = "linux"
UNIX_PLATFORMS = [LINUX_PLATFORM, DARWIN_PLATFORM]
WINDOWS_PLATFORMS = ["windows"]



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



def printOptions(optionsDesc):
  """
  Print command line options.
  """

  print "Options:\n"

  for option in optionsDesc:
    optionUsage = "--" + option[0]
    if option[1] != "":
      optionUsage += "=[" + option[1] + "]"

    optionDesc = option[2]
    print "    " + optionUsage.ljust(30) + " = " + optionDesc



def getCommandLineOptions():

  # optionDesc = [name, value, description]
  optionsDesc = []
  optionsDesc.append(
    ["nupic-core-dir",
     "dir",
     "Absolute path to nupic.core binary release directory"]
  )
  optionsDesc.append(
    ["compiler",
    "value",
    "(optional) compiler name to use"]
  )
  optionsDesc.append(
    ["optimizations-native",
    "value",
    "(optional) enable aggressive compiler optimizations"]
  )
  optionsDesc.append(
    ["optimizations-lto",
    "value",
    "(optional) enable link-time optimizations (LTO); currently only for gcc and linker ld.gold"]
  )
  optionsDesc.append(
    ["debug",
    "value",
    "(optional) compile in mode suitable for debugging; overrides any optimizations"]
  )

  # Read command line options looking for extra options
  # For example, an user could type:
  #   python setup.py install --nupic-core-dir="path/to/release"
  # which will set the nupic.core release dir
  optionsValues = dict()
  for arg in sys.argv[:]:
    optionFound = False
    for option in optionsDesc:
      name = option[0]
      if "--" + name in arg:
        value = None
        hasValue = (option[1] != "")
        if hasValue:
          value = arg.partition("=")[2]

        optionsValues[name] = value
        sys.argv.remove(arg)
        optionFound = True
        break

    if not optionFound:
      if ("--help-nupic" in arg):
        printOptions(optionsDesc)
        sys.exit()

  return optionsValues



def getCommandLineOption(name, options):
  if name is None or options is None:
    return False
  if name in options:
    return options[name]



def getExtensionFiles(platform):
  if platform in WINDOWS_PLATFORMS:
    libExtension = "dll"
  else:
    libExtension = "so"
  libNames = ("algorithms", "engine_internal", "math")
  swigPythonFiles = ["{}.py".format(name) for name in libNames]
  swigLibFiles = ["_{}.{}".format(name, libExtension) for name in libNames]
  files = [os.path.join(PY_BINDINGS, "nupic", "bindings", name)
           for name in list(swigPythonFiles + swigLibFiles)]

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
  cwd = os.getcwd()
  os.chdir(PY_BINDINGS)

  options = getCommandLineOptions()
  platform = getPlatformInfo()

  print "NumPy version: {}".format(numpy.__version__)
  print "Bindings directory: {}".format(PY_BINDINGS)

  try:
    if platform == DARWIN_PLATFORM and not "ARCHFLAGS" in os.environ:
      raise Exception("To build NuPIC Core bindings in OS X, you must "
                      "`export ARCHFLAGS=\"-arch x86_64\"`.")

    buildEgg = False
    for arg in sys.argv[:]:
      if arg == "bdist_egg":
        buildEgg = True

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
  finally:
    os.chdir(cwd)
