@echo off
rem Runs CMake to configure poppler for Visual Studio 2017.
rem https://github.com/nanodbc/nanodbc/blob/master/utility/build.bat

if not defined VS150COMNTOOLS goto :NoVS


:Build
set BUILDDIR=build

pushd ..\..

rem remove build folder
if exist ".\build\" (
    rd /s /q "build\"
)

mkdir %BUILDDIR%
pushd %BUILDDIR%

"C:/Program Files/CMake/bin/cmake.exe" ^
    -A x64 ^
    -DBOOST_ROOT:PATH=D:/boost ^
    -DBOOST_LIBRARYDIR:PATH=D:/boost/stage/lib ^
    -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg_2017_2/vcpkg/scripts/buildsystems/vcpkg.cmake ^
    -DVCPKG_TARGET_TRIPLET=x64-windows-static ^
    ..

rem Building
rem msbuild.exe nupic.base.sln /p:Configuration=Release /p:Platform=x64

popd
popd
goto :EOF

:NoVS
@echo build.bat
@echo  Visual Studio 2017 not found
@echo  "%%VS150COMNTOOLS%%" environment variable not defined
exit /B 1


