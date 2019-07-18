@echo off
rem // Runs CMake to configure htm.core for Visual Studio 2017 and 2019
rem // Execute this file to start up CMake and configure Visual Studio.
rem //
rem // There is no need to use Developer Command Prompt for running CMake with 
rem // Visual Studio generators, corresponding environment will be loaded automatically by CMake.
rem // 
rem // Prerequisites:
rem //
rem //     Microsoft Visual Studio 2017, 2019 or newer (any flavor)
rem //
rem //     CMake: 3.7+ Download CMake from https://cmake.org/download/
rem //            NOTE: CMake 3.14+ is required for MSCV 2019
rem //
rem //     Python 2.7 or 3.x Download from https://www.python.org/downloads/windows/  (optional)
rem // 
rem //
rem //   This script will create a Vsual Studio solution file at build/scripts/htm_core.sln
rem //   Double click htm_core.sln to start up Visual Studio.  Then perform a full build.
rem //
rem //   Note: if you were originally using this repository with VS 2017 and you now
rem //         want to use VS 2019, Do the following:
rem //           1) delete the build folder in the repository.  This will remove the .sln and other VS files.
rem //           2) run startupMSVC.bat again.   It will detect that VS 2019 is installed and configure for it but it will start VS 2017.
rem //           3) after .sln is re-created in build/scripts/, right click, on the .sln file, select "open with", click "choose another app", select VS 2019
rem //
rem // Tricks for executing Visual Studio in Release or Build mode.
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

rem // position to full path of HTM_BASE (the repository)
pushd %~dp0
set HTM_BASE=%CD%
@echo HTM_BASE=%HTM_BASE%
if not exist "%HTM_BASE%\CMakeLists.txt" (
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
cd "%BUILDDIR%"

rem // If htm_core.sln already exists, just startup Visual Studio
if exist "htm_core.sln" (
  htm_core.sln
  popd
  exit /B 0
)

rem // Run CMake using the Visual Studio generator.  The generator can be one of these.
rem //   cmake -G "Visual Studio 15 2017" -A x64
rem //   cmake -G "Visual Studio 16 2017" -A ARM
rem //
rem //   cmake -G "Visual Studio 16 2019" -A x64
rem //   cmake -G "Visual Studio 16 2019" -A ARM64
rem //      NOTE: MSVC 2019 tool set generator requires CMake V3.14 or greater)
rem //   
rem //  arguments:
rem //   -G "Visual Studio 16 2019"  Sets the generator toolset (compilers/linkers) to use. 
rem //   -A x64                      Sets the platform.  Note that 64bit only supported.
rem //   -Thost=x64                  Tell CMake to tell VS to use 64bit tools for compiler and linker
rem //   --config "Release"          Start out in Release mode
rem //   -DCMAKE_CONFIGURATION_TYPES="Debug;Release"   Specify the build types allowed.
rem //   ../..                       set the source directory (top of repository)

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
    set GENERATOR="Visual Studio 16 2019"
    set PLATFORM=x64
) else (
    set GENERATOR="Visual Studio 15 2017"
    set PLATFORM=x64
)

cmake -G %GENERATOR% -A %PLATFORM% -Thost=x64 --config "Release" -DCMAKE_CONFIGURATION_TYPES="Debug;Release"  ../..
  
if exist "htm_core.sln" (
    @echo " "
    @echo You can now start Visual Studio using solution file %HTM_BASE%\build\scripts\htm_core.sln
    @echo Dont forget to set your default Startup Project to unit_tests.
    @echo Press any key to start Visual Studio 
    pause >nul

    rem // Location is %HTM_BASE%\build\scripts\htm_core.sln
    htm_core.sln

    popd
    exit /B 0
) else (
    @echo An error occured. Correct problem. Delete %HTM_BASE%\build before trying again.
    popd
    pause
    exit /B 1
)
  



