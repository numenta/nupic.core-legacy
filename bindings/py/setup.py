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



def getPlatformInfo():
  """
  Identify platform
  """
  if "linux" in sys.platform:
    platform = "linux"
  elif "darwin" in sys.platform:
    platform = "darwin"
  # win32
  elif sys.platform.startswith("win"):
    platform = "windows"
  else:
    raise Exception("Platform '%s' is unsupported!" % sys.platform)

  # Python 32-bits doesn't detect Windows 64-bits so the workaround is
  # check whether "ProgramFiles (x86)" environment variable exists.
  is64bits = (sys.maxsize > 2**32 or
     (platform in WINDOWS_PLATFORMS and 'PROGRAMFILES(X86)' in os.environ))
  if is64bits:
    bitness = "64"
  else:
    bitness = "32"

  return platform, bitness



def getCompilerInfo(cxxCompiler):
  """
  Identify compiler
  """

  if cxxCompiler is None:
    cxxCompiler = ccompiler.get_default_compiler()
  
  if "clang" in cxxCompiler:
    cxxCompiler = "Clang"
  elif "gnu" in cxxCompiler:
    cxxCompiler = "GNU"
  # TODO: There is a problem here, because on OS X ccompiler.get_default_compiler()
  # returns "unix", not "clang" or "gnu". So we have to handle "unix" and we lose
  # the ability to decide which compiler is used.
  elif "unix" in cxxCompiler:
    cxxCompiler = "unix"
  elif "msvc" in cxxCompiler:
    cxxCompiler = "MSVC"
  elif "mingw" in cxxCompiler:
    cxxCompiler = "MinGW"
  else:
    raise Exception("C++ compiler '%s' is unsupported!" % cxxCompiler)

  print "CXX Compiler: {}".format(cxxCompiler)
  return cxxCompiler



def generateSwigWrap(swigExecutable, swigFlags, interfaceFile,
  language="python"):
  """
  Generate C++ code from the specified SWIG interface file.
  """
  wrap = interfaceFile.replace(".i", "_wrap.cxx")

  cmd = swigExecutable + " -c++ -{} ".format(language)
  for flag in swigFlags:
    cmd += flag + " "
  cmd += interfaceFile
  print cmd
  proc = subprocess.Popen(cmd, shell=True)
  proc.wait()

  return wrap



def getLibPrefix(platform):
  """
  Returns the default system prefix of a compiled library.
  """
  if platform in UNIX_PLATFORMS or cxxCompiler == "MinGW":
    return "lib"
  elif platform in WINDOWS_PLATFORMS:
    return ""



def getStaticLibExtension(platform):
  """
  Returns the default system extension of a compiled static library.
  """
  if platform in UNIX_PLATFORMS or cxxCompiler == "MinGW":
    return ".a"
  elif platform in WINDOWS_PLATFORMS:
    return ".lib"



def getExtensionModules(nupicCoreReleaseDir, platform, bitness, cxxCompiler, cmdOptions):
  #
  # Gives the version of Python necessary to get installation directories
  # for use with pythonVersion, etc.
  #
  if sys.version_info < (2, 7):
    raise Exception("Fatal Error: Python 2.7 or later is required.")

  pythonVersion = str(sys.version_info[0]) + '.' + str(sys.version_info[1])

  #
  # Find out where system installation of python is.
  #
  pythonPrefix = sys.prefix
  pythonPrefix = pythonPrefix.replace("\\", "/")
  if platform in WINDOWS_PLATFORMS:
    pythonIncludeDir = os.path.join(pythonPrefix, "include")
    pythonLib = "python" + pythonVersion.replace(".", "")
  else:
    pythonIncludeDir = os.path.join(
      pythonPrefix, "include", ("python" + pythonVersion)
    )
    pythonLib = "python" + pythonVersion

  #
  # Finds out version of Numpy and headers' path.
  #
  numpyIncludeDir = numpy.get_include()
  numpyIncludeDir = numpyIncludeDir.replace("\\", "/")

  commonDefines = [
    ("NTA_OS_" + platform.upper(), None),
    ("NTA_ARCH_" + bitness, None),
    ("NTA_PYTHON_SUPPORT", pythonVersion),
    ("HAVE_CONFIG_H", None),
    ("NTA_INTERNAL", None),
    ("BOOST_NO_WREGEX", None),
    ("NUPIC2", None),
    ("NTA_ASSERTIONS_ON", None),
    ("NTA_ASM", None)]

  if platform in WINDOWS_PLATFORMS:
    commonDefines.extend([
      ("PSAPI_VERSION", "1"),
      ("APR_DECLARE_STATIC", None),
      ("APU_DECLARE_STATIC", None),
      ("ZLIB_WINAPI", None),
      ("WIN32", None),
      ("_WINDOWS", None),
      ("_MBCS", None),
      ("_CRT_SECURE_NO_WARNINGS", None),
      ("NDEBUG", None),
      ("CAPNP_LITE", "CAPNP_LITE"),
      ("_VARIADIC_MAX", "10"),
      ("NOMINMAX", None)])
  else:
    commonDefines.append(("HAVE_UNISTD_H", None))
  
  if cxxCompiler == "GNU":
    commonDefines.append(("NTA_COMPILER_GNU", None))
  elif cxxCompiler == "Clang":
    commonDefines.append(("NTA_COMPILER_CLANG", None))
  elif cxxCompiler == "MSVC":
    commonDefines.append(("NTA_COMPILER_MSVC", None))
  elif cxxCompiler == "MinGW":
    commonDefines.append(("NTA_COMPILER_GNU", None))

  if cxxCompiler == "MinGW":
    commonDefines.append(("_hypot", "hypot"))
    commonDefines.append(("HAVE_UNISTD_H", None))

  commonIncludeDirs = [
    os.path.normpath(fixPath(PY_BINDINGS + "/../../external/" + platform + bitness + "/include")),
    os.path.normpath(fixPath(PY_BINDINGS + "/../../external/common/include")),
    fixPath(nupicCoreReleaseDir + "/include"),
    pythonIncludeDir,
    os.path.dirname(os.path.realpath(__file__)),
    numpyIncludeDir]

  if cxxCompiler == "MSVC":
    commonCompileFlags = [
      "/TP", "/Zc:wchar_t", "/Gm-", "/fp:precise", "/errorReport:prompt",
      "/W3", "/WX-", "/GR", "/Gd", "/GS-", "/Oy-", "/EHs", "/analyze-",
      "/nologo"]
    commonLinkFlags = [
      "/NOLOGO",
      "/SAFESEH:NO",
      "/NODEFAULTLIB:LIBCMT",
      "/LIBPATH:" + pythonPrefix + "/libs",
      "/LIBPATH:" + nupicCoreReleaseDir + "/lib"]
    if bitness == "32":
      commonLinkFlags.append("/MACHINE:X86")
    else:
      commonLinkFlags.append("/MACHINE:X" + bitness)
  else:
    commonCompileFlags = [
      # Adhere to c++11 spec
      "-std=c++11",
      # Generate 32 or 64 bit code
      "-m" + bitness,
      "-Wextra",
      "-Wreturn-type",
      "-Wunused",
      "-Wno-unused-parameter"]
    commonLinkFlags = [
      "-m" + bitness,
      "-L" + fixPath(nupicCoreReleaseDir + "/lib")]

    if cxxCompiler != "MinGW":
      # `Position Independent Code`, required for shared libraries
      commonCompileFlags.append("-fPIC")
      commonCompileFlags.append("-Wall")
      commonLinkFlags.append("-fPIC")

    if cxxCompiler == "MinGW":
      commonCompileFlags.append("-Wno-unused-local-typedefs")
      commonCompileFlags.append("-Wno-unused-variable")
      commonCompileFlags.append("-Wno-unused-function")
      # Apply these earlier in the link
      commonLinkFlags.append("-lkj")
      commonLinkFlags.append("-lcapnp")

  if platform == "darwin":
    commonCompileFlags.append("-stdlib=libc++")

  # Optimizations
  if getCommandLineOption("debug", cmdOptions) or cxxCompiler == "MinGW":
    commonCompileFlags.append("-Og")
    commonCompileFlags.append("-g")
    commonLinkFlags.append("-O0")
  else:
    if getCommandLineOption("optimizations-native", cmdOptions):
      commonCompileFlags.append("-march=native")
      commonCompileFlags.append("-O3")
      commonLinkFlags.append("-O3")
    else:
      commonCompileFlags.append("-mtune=generic")
      commonCompileFlags.append("-O2")
      commonLinkFlags.append("-O2")
    if getCommandLineOption("optimizations-lto", cmdOptions):
      commonCompileFlags.append("-fuse-linker-plugin")
      commonCompileFlags.append("-flto-report")
      commonCompileFlags.append("-fuse-ld=gold")
      commonCompileFlags.append("-flto")
      commonLinkFlags.append("-flto")

  commonLibraries = [
    pythonLib,
    "kj",
    "capnp"
  ]
  if platform == "linux":
    commonLibraries.extend(["capnpc","dl","pthread"])
  elif platform in WINDOWS_PLATFORMS:
    commonLibraries.extend([
      "psapi", "ws2_32", "shell32", "advapi32", "wsock32", "rpcrt4"])
    if cxxCompiler != "MinGW":
      commonLibraries.append("oldnames")

  commonObjects = [
    fixPath(nupicCoreReleaseDir + "/lib/" +
      getLibPrefix(platform) + "nupic_core" + getStaticLibExtension(platform))]

  supportFiles = [
    os.path.normpath(fixPath("../../src/nupic/py_support/NumpyVector.cpp")),
    os.path.normpath(fixPath("../../src/nupic/py_support/PyArray.cpp")),
    os.path.normpath(fixPath("../../src/nupic/py_support/PyHelpers.cpp")),
    os.path.normpath(fixPath("../../src/nupic/py_support/PythonStream.cpp"))]

  extensions = []

  #
  # SWIG
  #
  swigDir = os.path.normpath(fixPath(PY_BINDINGS + "/../../external/common/share/swig/3.0.2"))
  swigExecutable = (
    os.path.normpath(fixPath(PY_BINDINGS + "/../../external/" + platform + bitness + "/bin/swig"))
  )

  # SWIG options from:
  # https://github.com/swig/swig/blob/master/Source/Modules/python.cxx#L111
  swigFlags = [
    "-features",
    "autodoc=0,directors=0",
    "-noproxyimport",
    "-keyword",
    "-modern",
    "-modernargs",
    "-noproxydel",
    "-fvirtual",
    "-fastunpack",
    "-nofastproxy",
    "-fastquery",
    "-outputtuple",
    "-castmode",
    "-nosafecstrings",
    "-w402", #TODO silence warnings
    "-w503",
    "-w511",
    "-w302",
    "-w362",
    "-w312",
    "-w389",
    "-DSWIG_PYTHON_LEGACY_BOOL",
    "-I" + swigDir + "/python",
    "-I" + swigDir]
  for define in commonDefines:
    item = "-D" + define[0]
    if define[1]:
      item += "=" + define[1]
    swigFlags.append(item)
  for includeDir in commonIncludeDirs:
    item = "-I" + includeDir
    swigFlags.append(item)

  wrapAlgorithms = generateSwigWrap(swigExecutable,
                                    swigFlags,
                                    "nupic/bindings/algorithms.i")
  libModuleAlgorithms = Extension(
    "nupic.bindings._algorithms",
    extra_compile_args=commonCompileFlags,
    define_macros=commonDefines,
    extra_link_args=commonLinkFlags,
    include_dirs=commonIncludeDirs,
    libraries=commonLibraries,
    sources=supportFiles + [wrapAlgorithms],
    extra_objects=commonObjects)
  extensions.append(libModuleAlgorithms)

  wrapEngineInternal = generateSwigWrap(swigExecutable,
                                        swigFlags,
                                        "nupic/bindings/engine_internal.i")
  libModuleEngineInternal = Extension(
    "nupic.bindings._engine_internal",
    extra_compile_args=commonCompileFlags,
    define_macros=commonDefines,
    extra_link_args=commonLinkFlags,
    include_dirs=commonIncludeDirs,
    libraries=commonLibraries,
    sources=[wrapEngineInternal],
    extra_objects=commonObjects)
  extensions.append(libModuleEngineInternal)

  wrapMath = generateSwigWrap(swigExecutable,
                              swigFlags,
                              "nupic/bindings/math.i")
  libModuleMath = Extension(
    "nupic.bindings._math",
    extra_compile_args=commonCompileFlags,
    define_macros=commonDefines,
    extra_link_args=commonLinkFlags,
    include_dirs=commonIncludeDirs,
    libraries=commonLibraries,
    sources=supportFiles + [wrapMath, "nupic/bindings/PySparseTensor.cpp"],
    extra_objects=commonObjects)
  extensions.append(libModuleMath)

  return extensions



if __name__ == "__main__":
  cwd = os.getcwd()
  os.chdir(PY_BINDINGS)

  options = getCommandLineOptions()
  platform, bitness = getPlatformInfo()
  cxxCompiler = getCompilerInfo(getCommandLineOption("compiler", options))

  print "NumPy version: {}".format(numpy.__version__)
  print "Bindings directory: {}".format(PY_BINDINGS)

  try:
    nupicCoreReleaseDir = getCommandLineOption("nupic-core-dir", options)
    if nupicCoreReleaseDir is None:
      raise Exception("Must provide nupic core release directory. --nupic-core-dir")
    nupicCoreReleaseDir = fixPath(nupicCoreReleaseDir)
    print "Core directory: {}\n".format(nupicCoreReleaseDir)
    if not os.path.isdir(nupicCoreReleaseDir):
      raise Exception("{} does not exist".format(nupicCoreReleaseDir))

    if platform == DARWIN_PLATFORM and not "ARCHFLAGS" in os.environ:
      raise Exception("To build NuPIC Core bindings in OS X, you must "
                      "`export ARCHFLAGS=\"-arch x86_64\"`.")

    buildEgg = False
    for arg in sys.argv[:]:
      if arg == "bdist_egg":
        buildEgg = True

    # Build and setup NuPIC.Core Bindings
    print "Get SWIG C++ extensions"
    extensions = getExtensionModules(nupicCoreReleaseDir, platform, bitness, cxxCompiler,
      options)

    # Copy the proto files into the proto Python package.
    destDir = os.path.relpath(os.path.join("nupic", "proto"))
    for protoPath in glob.glob(os.path.relpath(os.path.join(
        "..", "..", "src", "nupic", "proto", "*.capnp"))):
      shutil.copy(protoPath, destDir)

    print "\nSetup SWIG Python module"
    setup(
      name="nupic.bindings",
      ext_modules=extensions,
      version="0.2.2",
      namespace_packages=["nupic"],
      install_requires=findRequirements(platform),
      description="Numenta Platform for Intelligent Computing - bindings",
      author="Numenta",
      author_email="help@numenta.org",
      url="https://github.com/numenta/nupic.core",
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
      long_description = "Python bindings for nupic core.",
      packages=find_packages(),
      package_data={"nupic.proto": ["*.capnp"]},
      zip_safe=False,
    )
  finally:
    os.chdir(cwd)
