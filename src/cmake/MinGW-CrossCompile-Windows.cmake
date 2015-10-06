# http://www.cmake.org/Wiki/CMake_Cross_Compiling

# The name of the target operating system
SET(CMAKE_SYSTEM_NAME Windows)

# Autoset when CMAKE_SYSTEM_NAME is defined
SET(CMAKE_CROSSCOMPILING 1)

# Choose an appropriate compiler prefix

# for classical mingw32
# see http://www.mingw.org/
#set(COMPILER_PREFIX "i586-mingw32msvc")

# for 32 or 64 bits mingw-w64
# see http://mingw-w64.sourceforge.net/
#set(COMPILER_PREFIX "i686-w64-mingw32")
set(COMPILER_PREFIX "x86_64-w64-mingw32")

# which compilers to use for C and C++
set(CMAKE_RC_COMPILER windres)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_AR ar)
set(CMAKE_NM nm)
set(CMAKE_RANLIB ranlib)

# Target environment located
#SET(USER_ROOT_PATH /home/travis/build/rcrowder/travis-ci-test)
#SET(CMAKE_FIND_ROOT_PATH  /usr/${COMPILER_PREFIX} ${USER_ROOT_PATH})
SET(CMAKE_FIND_ROOT_PATH C:\\mingw-w64\\bin)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
