# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2016, Numenta, Inc.  Unless you have purchased from
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
# -----------------------------------------------------------------------------

# Configures common compiler/linker/loader settings for internal and external
# sources.

# INPUTS:
#
# PLATFORM: lowercase ${CMAKE_SYSTEM_NAME}

# OUTPUTS:
#
# BITNESS: Platform bitness: 32 or 64
#
# COMMON_COMPILER_DEFINITIONS: list of -D define flags for the compilation of
#                               source files; e.g., for cmake `add_definitions()`
# COMMON_COMPILER_DEFINITIONS_STR: string variant of COMMON_COMPILER_DEFINITIONS
#
# COMMON_C_FLAGS: string of common C flags
#
# COMMON_CXX_FLAGS: string of common C++ flags (COMMON_C_FLAGS + C++ flags)
#
# COMMON_LINK_FLAGS: string of linker flags
#
# CMAKE_LINKER: updated, if needed; see cmake documentation

if(NOT DEFINED PLATFORM)
    message(FATAL_ERROR "PLATFORM property not defined: PLATFORM=${PLATFORM}")
endif()

string(TOUPPER ${PLATFORM} PLATFORM_UPPERCASE)

# Identify platform "bitness".
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()

set(STDLIB_CXX "")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(STDLIB "")
  set(STDLIB_CXX "-stdlib=libc++")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(STDLIB "")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(STDLIB "")
endif()


# Compiler `-D*` definitions
set(COMMON_COMPILER_DEFINITIONS)

if(UNIX) # or UNIX like (i.e. APPLE and CYGWIN)
  set(COMMON_COMPILER_DEFINITIONS
      ${COMMON_COMPILER_DEFINITIONS}
      -DHAVE_UNISTD_H)
elseif(MSVC OR MSYS OR MINGW)
  set(COMMON_COMPILER_DEFINITIONS
      ${COMMON_COMPILER_DEFINITIONS}
      -DPSAPI_VERSION=1
      -DWIN32
      -D_WINDOWS
      -D_MBCS
      -D_CRT_SECURE_NO_WARNINGS
      -DNDEBUG
      -D_VARIADIC_MAX=10
      -DNOMINMAX
      -DCAPNP_LITE=1)
  if(MSYS OR MINGW)
    set(COMMON_COMPILER_DEFINITIONS
        ${COMMON_COMPILER_DEFINITIONS}
        -DHAVE_UNISTD_H)
  endif()
endif()


#
# Set linker (ld)
# use ld.gold if available
#
execute_process(COMMAND ld.gold --version RESULT_VARIABLE EXIT_CODE)
if(EXIT_CODE EQUAL 0)
  message("Using ld.gold as LINKER.")
  set(CMAKE_LINKER "ld.gold")
endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set(STDLIB "${STDLIB} -static-libgcc")
   set(STDLIB_CXX "${STDLIB_CXX} -static-libstdc++")
endif()


#
# Enable Optimization flags here
# These are quite aggresive flags, if your code misbehaves
# for strange reasons, try compiling without them.
#
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -mtune=generic -O2")
  set(OPTIMIZATION_FLAGS_CC "-pipe ${OPTIMIZATION_FLAGS_CC}") #TODO use -Ofast instead of -O3
  set(OPTIMIZATION_FLAGS_LT "-O2 ${OPTIMIZATION_FLAGS_LT}")

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND NOT MINGW)
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -fuse-linker-plugin -flto-report")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -flto -fuse-ld=gold") #TODO fix LTO for clang
    set(OPTIMIZATION_FLAGS_LT "${OPTIMIZATION_FLAGS_LT} -flto") #TODO LTO for clang too
  endif()
endif()


#
# compiler specific settings here
#
set(COMMON_CXX_FLAGS "")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  # MS Visual C
  set(SHARED_COMPILE_FLAGS "/TP /Zc:wchar_t /Gm- /fp:precise /errorReport:prompt /W1 /WX- /GR /Gd /GS /Oy- /EHs /analyze- /nologo")
  set(COMMON_LINK_FLAGS "/NOLOGO /SAFESEH:NO /NODEFAULTLIB:LIBCMT")
  if("${BITNESS}" STREQUAL "32")
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} /MACHINE:X86")
  else()
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} /MACHINE:X${BITNESS}")
  endif()

else()
  # LLVM Clang / Gnu GCC
  set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -std=c++11")

  set(SHARED_COMPILE_FLAGS "-m${BITNESS} ${STDLIB} -Wextra -Wreturn-type -Wunused -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers")
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(SHARED_COMPILE_FLAGS "-Wall -fPIC ${SHARED_COMPILE_FLAGS} -Wno-deprecated-register")
  endif()
  set(COMMON_LINK_FLAGS "-m${BITNESS} ${STDLIB} ${STDLIB_CXX}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(SHARED_COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} -g")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -O0")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR MINGW)
    set(SHARED_COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} -Og")
  endif()
else()
  set(SHARED_COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} ${OPTIMIZATION_FLAGS_CC}")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} ${OPTIMIZATION_FLAGS_LT}")
endif()

if(MINGW)
  # This is for GCC 4.8.x
  # http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0
  set(SHARED_COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} -include cmath")
endif()

# Set compiler-specific options.
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -Wl,--no-undefined")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -Wl,-undefined,error")
endif()

set(COMMON_C_FLAGS "${SHARED_COMPILE_FLAGS}")

set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} ${SHARED_COMPILE_FLAGS} ${STDLIB_CXX}")

# Provide a string variant of COMMON_COMPILER_DEFINITIONS list
string (REPLACE ";" " " COMMON_COMPILER_DEFINITIONS_STR "${COMMON_COMPILER_DEFINITIONS}")
