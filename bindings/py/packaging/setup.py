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
import json

from setuptools import Command, find_packages, setup
from setuptools.command.test import test as BaseTestCommand
from distutils.core import Extension

# NOTE:  To debug the python bindings in a debugger, use the procedure
#        described here: https://pythonextensionpatterns.readthedocs.io/en/latest/debugging/debug_in_ide.html
#

# NOTE:  CMake usually is able to determine the tool chain based on the platform.
#        However, if you would like CMake to use a different generator, (perhaps an 
#        alternative compiler) you can set the environment variable NC_CMAKE_GENERATOR
#        to the generator you wish to use.  See CMake docs for generators avaiable.
#         
#         On Windows, CMake will try to use the newest Visual Studio installed
#         on your machine. You many choose an older version as follows:
#            set NC_CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
#            python setup.py install --user --force
#         This script will override the default 32bit bitness such that a 64bit build is created.
#


# bindings cannot be compiled in Debug mode, unless a special python library also in
# Debug is used, which is quite unlikely. So for any CMAKE_BUILD_TYPE setting, override 
# to Release mode. 
build_type = 'Release'

PY_BINDINGS = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(PY_BINDINGS, os.pardir, os.pardir, os.pardir))
DISTR_DIR = os.path.join(REPO_DIR, "build", build_type, "distr")
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



def findRequirements(platform, fileName="requirements.txt"):
  """
  Read the requirements.txt file and parse into requirements for setup's
  install_requirements option.
  """
  requirementsPath = fixPath(os.path.join(REPO_DIR, fileName))
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
    errno = 0
    # run c++ tests (from python)
    cpp_tests = os.path.join(REPO_DIR, "build", "Release", "bin", "unit_tests")
    subprocess.check_call([cpp_tests])
    os.chdir(cwd)

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



def getExtensionFileNames(platform, build_type):
  # look for extension libraries in Repository/build/Release/distr/src/htm/bindings
  # library filenames:  
  #     htm.core.algorithms.so
  #     htm.core.engine.so
  #     htm.core.math.so
  #     htm.core.encoders.so
  #     htm.core.sdr.so
  # (or on windows x64 with Python3.7:)
  #     algorithms.cp37-win_amd64.pyd
  #     engine_internal.cp37-win_amd64.pyd
  #     math.cp37-win_amd64.pyd
  #     encoders.cp37-win_amd64.pyd
  #     sdr.cp37-win_amd64.pyd
  if platform in WINDOWS_PLATFORMS:
    libExtension = "pyd"
  else:
    libExtension = "so"
  libNames = ("sdr", "encoders", "algorithms", "engine_internal", "math")
  libFiles = ["{}.*.{}".format(name, libExtension) for name in libNames]
  files = [os.path.join(DISTR_DIR, "src", "htm", "bindings", name)
           for name in list(libFiles)]
  return files


def getExtensionFiles(platform, build_type):
  files = getExtensionFileNames(platform, build_type)
  for f in files:
    if not glob.glob(f):
      generateExtensions(platform, build_type)
      break

  return files
  
def isMSVC_installed(ver):
  """
  For windows we need to know the most recent version of Visual Studio that is installed.
  This is because the calling arguments for setting x64 is different between 2017 and 2019.
  
  Run vswhere to get Visual Studio info.  (only available in MSVC 2017 and later)
  Parse the json and look in displayName for "2017" or "2019"
  return true if ver is found.
  """
  vswhere = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
  output = subprocess.check_output([vswhere, "-legacy", "-prerelease", "-format", "json"], universal_newlines=True)
  data = json.loads(output);
  for vs in data:
    if 'displayName' in vs and ver in vs['displayName']: return True
  return False


def generateExtensions(platform, build_type):
  """
  This will perform a full Release build with default arguments.
  The CMake build will copy everything in the Repository/bindings/py/packaging 
  directory to the distr directory (Repository/build/Release/distr)
  and then create the extension libraries in Repository/build/Release/distr/src/nupic/bindings.
  Note: for Windows it will force a X64 build.
  """
  cwd = os.getcwd()
  
  print("Python version: {}\n".format(sys.version))
  from sys import version_info
  if version_info > (3, 0):
    # Build a Python 3.x library
    PY_VER = "-DBINDING_BUILD=Python3"
  else:
    # Build a Python 2.7 library
    PY_VER = "-DBINDING_BUILD=Python2"
    if platform == "windows":
        raise Exception("Python2 is not supported on Windows.")
        

  BUILD_TYPE = "-DCMAKE_BUILD_TYPE="+build_type


  scriptsDir = os.path.join(REPO_DIR, "build", "scripts")
  try:
    if not os.path.isdir(scriptsDir):
      os.makedirs(scriptsDir)
    os.chdir(scriptsDir)
    
    # Call CMake to setup the cache for the build.
    # Normally we would let CMake figure out the generator based on the platform.
    # But Visual Studio gets it wrong.  By default it uses 32 bit and we support only x64.  
    # Also Visual Studio 2019 now wants a new argument -A to specify that we want x64.
    # Using -A on 2017 causes an error.  So we have to manually specify each.
    generator = os.environ.get('NC_CMAKE_GENERATOR')
    if generator == None:
      # The generator is not specified, figure out which to use.
      if platform == "windows":
        # Check to see if the CMake cache already exists and defines BINDING_BUILD.  If it does, skip this step
        if not os.path.isfile('CMakeCache.txt') or not 'BINDING_BUILD:STRING=Python3' in open('CMakeCache.txt').read():
          # Note: the calling arguments for MSVC 2017 is not the same as for MSVC 2019
          if isMSVC_installed("2019"):
            subprocess.check_call(["cmake", "-G", "Visual Studio 16 2019", "-A", "x64", BUILD_TYPE, PY_VER, REPO_DIR])
          elif isMSVC_installed("2017"):
            subprocess.check_call(["cmake", "-G", "Visual Studio 15 2017 Win64", BUILD_TYPE, PY_VER, REPO_DIR])
          else:
            raise Exception("Did not find Microsoft Visual Studio 2017 or 2019.")
        #else 
        #   we can skip this step, the cache is already setup and we have the right binding specified.
      else:
        # For Linux and OSx we can let CMake figure it out.
        subprocess.check_call(["cmake",BUILD_TYPE , PY_VER, REPO_DIR])
        
    else:
      # The generator is specified.
      if platform == "windows":
        # Check to see if cache already exists.  If it does, skip this step
        if not os.path.isfile("CMakeCache.txt"):
          # Note: the calling arguments for MSVC 2017 is not the same as for MSVC 2019
          if '2019' in generator and isMSVC_installed("2019"):
            subprocess.check_call(["cmake", "-G", "Visual Studio 16 2019", "-A", "x64", BUILD_TYPE, PY_VER, REPO_DIR])
          elif '2017' in generator and isMSVC_installed("2017"):
            subprocess.check_call(["cmake", "-G", "Visual Studio 15 2017 Win64", BUILD_TYPE, PY_VER, REPO_DIR])
          else:
            raise Exception('Did not find Visual Studio for generator "'+generator+ '".')
      else:
        subprocess.check_call(["cmake", "-G", generator, BUILD_TYPE, PY_VER, REPO_DIR])
        
    # Now do `make install`
    subprocess.check_call(["cmake", "--build", ".", "--target", "install", "--config", build_type])
  finally:
    os.chdir(cwd)



if __name__ == "__main__":
  platform = getPlatformInfo()

  if platform == DARWIN_PLATFORM and not "ARCHFLAGS" in os.environ:
    os.system("export ARCHFLAGS=\"-arch x86_64\"")

  # Run CMake if extension files are missing.
  # CMake also copies all py files into place in DISTR_DIR
  getExtensionFiles(platform, build_type)

  with open(os.path.join(REPO_DIR, "README.md"), "r") as fh:
    long_description = fh.read()

  """
  set the default directory to the distr, and package it.
  """
  print("\nbindings/py/setup.py: Setup htm.core Python module in " + DISTR_DIR+ "\n")
  os.chdir(DISTR_DIR)

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
    #install extras by `pip install htm.core[examples]`
    extras_require={'scikit-image>0.15.0':'examples',
                    'sklearn':'examples',
                    'matplotlib':'examples',
                    'PIL':'examples',
                    'scipy':'examples'
                   },
    zip_safe=False,
    cmdclass={
      "clean": CleanCommand,
      "test": TestCommand,
    },
    author="Numenta & HTM Community",
    author_email="help@numenta.org",
    url="https://github.com/htm-community/htm.core",
    description = "HTM Community Edition of Numenta's Platform for Intelligent Computing (NuPIC) htm.core",
    long_description = long_description,
    long_description_content_type="text/markdown",
    license = "GNU Affero General Public License v3 or later (AGPLv3+)",
    classifiers=[
      "Programming Language :: Python",
      "Programming Language :: Python :: 2",
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: POSIX :: Linux",
      "Operating System :: POSIX :: BSD",
      "Operating System :: Microsoft :: Windows",
      "Operating System :: OS Independent",
      # It has to be "5 - Production/Stable" or else pypi rejects it!
      "Development Status :: 5 - Production/Stable",
      "Environment :: Console",
      "Intended Audience :: Science/Research",
      "Intended Audience :: Developers",
      "Intended Audience :: Education",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
      "Natural Language :: English",
      "Programming Language :: C++",
      "Programming Language :: Python"
    ],
    entry_points = {
      "console_scripts": [
        "htm-bindings-check = htm.bindings.check:checkMain",
      ],
    },
  )
  print("\nbindings/py/setup.py: Setup complete.\n")

