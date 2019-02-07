@echo off
rem // Runs CMake to configure Nupic.cpp for Visual Studio 2017
rem // Execute this file to start up CMake and configure Visual Studio.
rem //
rem // There is no need to use Developer Command Prompt for running CMake with 
rem // Visual Studio generators, corresponding environment will be loaded automatically by CMake.
rem // 
rem // Prerequisites:
rem //
rem //     Microsoft Visual Studio 2017 or newer (any flavor)
rem //
rem //     CMake: 3.7+ Download CMake from https://cmake.org/download/
rem //
rem //     Python 2.7 or 3.x Download from https://www.python.org/downloads/windows/  (optional)
rem // 
rem //
rem //   This script will create a Vsual Studio solution file at build/scripts/nupic_core.sln
rem //   Double click nupic_core.sln to start up Visual Studio.  Then perform a full build.
rem //
rem // Tricks for executing Visual Studio is Release or Build mode.
rem // https://stackoverflow.com/questions/24460486/cmake-build-type-not-being-used-in-cmakelists-txt

  
:CheckCMake
rem  // make sure CMake is installed.  (version 3.7 minimum)
cmake -version > NUL 2> NUL 
if %errorlevel% neq 0 (
  @echo startup.bat;  CMake was not found. 
  @echo Download and install from  https://cmake.org/download/
  @echo Make sure its path is in the system PATH environment variable.
  pause
  exit /B 1
)

rem // position to full path of NUPIC_BASE (the repository)
pushd %~dp0
set NUPIC_BASE=%CD%
@echo NUPIC_BASE=%NUPIC_BASE%
if not exist "%NUPIC_BASE%\CMakeLists.txt" (
  @echo CMakeList.txt not found in base folder.
  popd
  pause
  exit /B 1
)


:Build
set BUILDDIR=build\scripts
if not exist "%BUILDDIR%" (
  mkdir "%BUILDDIR%"
)
rem // Run CMake using the Visual Studio generator for VS 2017
rem //   -G "Visual Studio 15 2017 Win64"   set the generator toolset to use. Could also set ARM rather than Win64.
rem //   -Thost=x64                  Tell CMake to tell VS to use 64bit tools for compiler and linker
rem //   --config "Release"          Start out in Release mode
rem //   -DCMAKE_CONFIGURATION_TYPES="Debug;Release"   Specify the build types allowed.
rem //   ../..                       set the source directory (top of repository)

cd "%BUILDDIR%"
rem // cmake -G "Visual Studio 15 2017 Win64" -Thost=x64 --config "Release" -DCMAKE_CONFIGURATION_TYPES="Debug;Release" -DBINDING_BUILD=Python3 ../..
cmake -G "Visual Studio 15 2017 Win64" -Thost=x64 --config "Release" -DCMAKE_CONFIGURATION_TYPES="Debug;Release"  ../..
  
if exist "nupic_core.sln" (
    cmake --build . --target install --config "Release"
    @echo " "
    @echo You can now start Visual Studio using solution file %NUPIC_BASE%\build\scripts\nupic_core.sln
    @echo Press any key to start Visual Studio 
    pause >nul

    rem // %NUPIC_BASE%\build\scripts\nupic_core.sln
    nupic_core.sln

    popd
    exit /B 0
) else (
    @echo An error occured. Correct problem. Delete %NUPIC_BASE%\build before trying again.
    popd
    pause
    exit /B 1
)
  



