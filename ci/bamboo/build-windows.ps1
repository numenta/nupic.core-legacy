# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

# Builds and tests the nupic.bindings python extension using Release build
# type.
#
# ASSUMPTIONS:
#          1. Building via Numenta Vagrant Task using Vagrant image
#             bamboo-vagrant-windows-1.x.x.
#
#          2. nupic.core root is the current directory on entry.
#
#          3. Expects a pristine nupic.core source tree without any remnant
#             build artifacts from prior build attempts. Otherwise, the
#             behavior is undefined.
#
# OUTPUTS:
#   nupic.bindings wheel: On success, the resulting wheel will be located in
#                         the subdirectory nupic_bindings_wheelhouse of the
#                         source tree's root directory.
#
#   Test results: nupic.bindings test results will be located in the
#                 subdirectory test_results of the source tree's root directory
#                 with the following content:
#
#                 junit-test-results.xml
#                 htmlcov/

# Stop and fail script if any command fails
$ErrorActionPreference = "Stop"

# Trace script lines as they run
Set-PsDebug -Trace 1



$NupicCoreRootDir = $(get-location).Path



# Use this function to wrap external commands for powershell error-checking.
#
# This is necessary so that `$ErrorActionPreference = "Stop"` will have the
# desired effect.
#
# Returns True if command's $LastExitCode was 0, False otherwise
#
# Usage: WrapCmd { cmd arg1 arg2 ... }
#
function WrapCmd
{
  [CmdletBinding()]

  param (
    [Parameter(Position=0, Mandatory=1)]
    [scriptblock]$Command,
    [Parameter(Position=1, Mandatory=0)]
    [string]$ErrorMessage = "ERROR: Command failed.`n$Command"
  )
  & $Command
  if ($LastExitCode -eq 0) {
    return $true
  }
  else {
    Write-Error "WrapCmd: $ErrorMessage"
    return $false
  }
}


#
# Make unix-compatible patch.exe available to the build by copying it from
# Git\usr\bin to another directory and adding it to PATH; the reason we copy it
# is that Git\usr\bin contains sh.exe that cmake doesn't like.
#

# Verify that patch command is not available yet.
&where.exe patch
if ($LastExitCode -eq 0) {
    throw "patch command was already available."
}

mkdir "C:\Program Files\PatchFromGit"
copy "C:\Program Files\Git\usr\bin\patch.exe" "C:\Program Files\PatchFromGit"
copy "C:\Program Files\Git\usr\bin\msys*.dll" "C:\Program Files\PatchFromGit"
$env:PATH = 'C:\Program Files\PatchFromGit;' + $env:PATH

# Verify that patch is now available
WrapCmd { where.exe patch }


#
# Remove sh.exe from the paths (CMake doesn't like it and fails in a subtle way)
#

# Validate expectation that sh.exe is in PATH prior to its removal
WrapCmd { where.exe sh }

$env:PATH = $env:PATH.Replace('C:\Program Files\OpenSSH\bin','')

# Verify that sh command was successfully removed from PATH"
&where.exe sh
if ($LastExitCode -eq 0) {
  throw "Failed to remove sh.exe from PATH."
}


# Verify that core toolchain components are available and log their versions
Write-Host "Checking tools."
WrapCmd { gcc --version }
WrapCmd { g++ --version }

WrapCmd { where.exe python }
WrapCmd { python --version }
Write-Host "PYTHONHOME=$env:PYTHONHOME"

WrapCmd { pip --version }

WrapCmd { wheel version }

# Log installed python packages
WrapCmd { pip list }


#
# Setup MinGW GCC as a valid distutils compiler
#
copy ".\external\windows64-gcc\bin\distutils.cfg" `
     "$($env:PYTHONHOME)\Lib\distutils\distutils.cfg"


#
# Build nupic.core
#

mkdir .\build\release
mkdir .\build\scripts

pushd .\build\scripts

$env:CC = "gcc"
$env:CXX = "g++"

WrapCmd {
  cmake `
    -G "MinGW Makefiles"  `
    -DCMAKE_BUILD_TYPE="Release" `
    -DCMAKE_INSTALL_PREFIX:PATH="..\release" `
    -DNUPIC_BUILD_PYEXT_MODULES="ON" `
    -DPY_EXTENSIONS_DIR:PATH="..\..\bindings\py\src\nupic\bindings" `
    "..\.."
}

# Make nupic.core from non-debug configuration
WrapCmd { cmake --build . --target install --config Release }

popd


# Create a python wheel in the destination wheelhouse
Write-Host "Building nupic.bindings python wheel."
WrapCmd {
  python setup.py bdist_wheel `
    --dist-dir "$($NupicCoreRootDir)\nupic_bindings_wheelhouse"
}


#
# Run tests
#

# Install nupic.bindings before running c++ tests; py_region_test depends on it
Write-Host "Installing from built nupic.bindings wheel."

dir -Filter *.whl -Recurse | Select Fullname

# Get path of nupic.bindings wheel
$NupicBindingsWheel = `
  (Get-ChildItem .\nupic_bindings_wheelhouse\nupic.bindings-*.whl)[0].FullName

WrapCmd { pip install $NupicBindingsWheel }


Write-Host "Running nupic.core C++ tests."

pushd .\build\release\bin

WrapCmd { .\py_region_test.exe }
WrapCmd { .\cpp_region_test.exe }
WrapCmd { .\unit_tests.exe }

# These executables aren't necessary for good test coverage. Leave them out
# of regular build to reduce build time.
#WrapCmd { .\helloregion.exe }
#WrapCmd { .\hello_sp_tp.exe }
#WrapCmd { .\prototest.exe }
#WrapCmd { .\connections_performance_test.exe }
popd


Write-Host "Running nupic.bindings python tests."

# So that py.test will deposit its artifacts in test_results
mkdir .\test_results
pushd .\test_results

# NOTE we use py.test directly instead of `python setup.py test`, because the
# latter changes current working directory, thus interfering with our ability to
# control the location of test artifacts.
WrapCmd { py.test ..\bindings\py\tests }

popd
