@echo off
rem // Runs CMake to configure Nupic.cpp for Visual Studio 2017
rem //
rem // if not ran from "Developer Command Prompt for VS 2017" this will look for
rem // Visual Studio installation in the standard places and then setup 
rem // the tool chain for a X64 project. Other configurations can be constructed
rem // by running this script from within one of the special command prompts provided 
rem // with Visual Studio. see https://msdn.microsoft.com/en-us/library/f2ccy3wt.aspx
rem //
rem // Prerequisites:
rem //      Boost:  optionally requires modules filesystem and system. 
rem //                  See external/common/share/boost/README.md
rem //
rem //     CMake:  Download CMake from https://cmake.org/download/
rem //
rem //     Python 2.7 or 3.x   with numpy
rem //                   Download from https://www.python.org/downloads/windows/
rem // 
rem // Arguments: %1  - path to Boost Root (Optional)
rem //                             NOTE:  This is the path where a full Boost was installed.
rem //
rem //   This script will create a Vsual Studio solution file at build/scripts/nupic_core.sln
rem //   Double click nupic_core.sln to start up Visual Studio.  Then perform a full build.
rem //

if NOT [%1]==[] set BOOST_ROOT %1
goto CheckCMake

  
:CheckCMake
rem  // make sure CMake is installed.  (version 3.12 minimum)
cmake -version > NUL 2> NUL 
if %errorlevel% neq 0 (
  @echo startup.bat;  CMake was not found. 
  @echo Download and install from  https://cmake.org/download/
  @echo Make sure its path is in the system PATH environment variable.
  pause
  exit /B 1
)

rem // position to full path of NUPIC_BASE
pushd %~dp0\..\..
set NUPIC_BASE=%CD%
@echo NUPIC_BASE=%NUPIC_BASE%
if not exist "%NUPIC_BASE%\CMakeLists.txt" (
  @echo CMakeList.txt not found in base folder.
  pause
  exit /B 1
)


:Build
set BUILDDIR=build

rem // create the build folder
if not exist "%BUILDDIR%\" (
  mkdir %BUILDDIR%
)
if not exist "%BUILDDIR%\scripts" (
  mkdir %BUILDDIR%\scripts
)

if defined VS150COMNTOOLS goto CheckVS
  rem // Try to locate the tool chain for Visual Studio 2017 
  if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat" (
    set vsDev="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat"
  )
  if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\Tools\VsDevCmd.bat" (
    set vsDev="C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\Tools\VsDevCmd.bat"
  ) 
  if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsDevCmd.bat" ( 
    set vsDev="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsDevCmd.bat"
  )
  if not [%vsDev%]==[] call %vsDev%
  if %errorlevel% neq 0 (
  	@echo An error setting tool chain.
	popd
  	pause
	exit /B 0
  )

:CheckVS

rem // make sure Visual studio is installed
if defined VS150COMNTOOLS (
  cd %NUPIC_BASE%\%BUILDDIR%\scripts

  rem // Run CMake using the Visual Studio generator for VS 2017
  cmake ../.. -G "Visual Studio 15 2017 Win64"  -DCMAKE_INSTALL_PREFIX=%BUILDDIR% 
  
  if exist "nupic_core.sln" (
  	@echo You can now start Visual Studio using solution file %BUILDDIR%\scripts\nupic_core.sln
  	pause
	exit /B 0
  ) else (
    @echo An error occured.
    @echo  Try executing this command using "Developer Command Prompt for VS2017" to set the tool chain.
    popd
    pause
    exit /B 1
  )
  
) else (
    @echo build.bat
    @echo  Visual Studio 2017 not found
    @echo  "%%VS150COMNTOOLS%%" environment variable not defined
    @echo  Try executing this command using "Developer Command Prompt for VS2017" to set the tool chain.
    popd
    pause
    exit /B 1
)



