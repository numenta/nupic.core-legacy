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
#
# NOTE SETTINGS THAT ARE SPECIFIC TO THIS OR THAT MODULE DO NOT BELONG HERE.

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
# EXTERNAL_C_FLAGS_UNOPTIMIZED: string of C flags without explicit optimization flags for 3rd-party sources
#
# EXTERNAL_C_FLAGS_OPTIMIZED: EXTERNAL_C_FLAGS_UNOPTIMIZED plus optimizations
#
# EXTERNAL_CXX_FLAGS_UNOPTIMIZED: string of C++ flags without explicit optimization flags for 3rd-party sources.
#
# EXTERNAL_LINKER_FLAGS_UNOPTIMIZED: string of linker flags for linking 3rd-party executables
#                      and shared libraries (DLLs) without explicit optimizations
#                      settings. This property is for use with
#                      EXTERNAL_C_FLAGS_UNOPTIMIZED and EXTERNAL_CXX_FLAGS_UNOPTIMIZED
#
# EXTERNAL_LINKER_FLAGS_UNOPTIMIZED: string of linker flags for linking 3rd-party executables
#                      and shared libraries (DLLs) without optimizations that are
#                      compatible with EXTERNAL_C_FLAGS_UNOPTIMIZED and EXTERNAL_CXX_FLAGS_UNOPTIMIZED
#
# EXTERNAL_LINKER_FLAGS_OPTIMIZED: string of linker flags for linking 3rd-party executables
#                      and shared libraries (DLLs) with optimizations that are
#                      compatible with EXTERNAL_C_FLAGS_OPTIMIZED and EXTERNAL_CXX_FLAGS_OPTIMIZED
#
# INTERNAL_CXX_FLAGS_OPTIMIZED: string of C++ flags with explicit optimization flags for internal sources
#
# INTERNAL_LINKER_FLAGS_OPTIMIZED: string of linker flags for linking internal executables
#                      and shared libraries (DLLs) with optimizations that are
#                      compatible with INTERNAL_CXX_FLAGS_OPTIMIZED
#
# CMAKE_LINKER: updated, if needed; use ld.gold if available. See cmake
#               documentation
#
# NOTE The XXX_OPTIMIZED flags are quite aggresive - if your code misbehaves for
# strange reasons, try compiling without them.

# NOTE much of the code below was factored out from src/CMakeLists.txt

if(NOT DEFINED PLATFORM)
    message(FATAL_ERROR "PLATFORM property not defined: PLATFORM=${PLATFORM}")
endif()


# Init exported properties
set(COMMON_COMPILER_DEFINITIONS)
set(COMMON_COMPILER_DEFINITIONS_STR)

set(INTERNAL_CXX_FLAGS_OPTIMIZED)
set(INTERNAL_LINKER_FLAGS_OPTIMIZED)


set(EXTERNAL_C_FLAGS_UNOPTIMIZED)
set(EXTERNAL_C_FLAGS_OPTIMIZED)

set(EXTERNAL_CXX_FLAGS_UNOPTIMIZED)
set(EXTERNAL_CXX_FLAGS_OPTIMIZED)

set(EXTERNAL_LINKER_FLAGS_UNOPTIMIZED)
set(EXTERNAL_LINKER_FLAGS_OPTIMIZED)


# Identify platform "bitness".
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()

message(STATUS "CMAKE BITNESS=${BITNESS}")


# Compiler `-D*` definitions
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
      -DNOMINMAX)
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


#
# Determine stdlib settings
#
set(stdlib_cxx "")
set(stdlib_common "")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(stdlib_cxx "${stdlib_cxx} -stdlib=libc++")
endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set(stdlib_common "${stdlib_common} -static-libgcc")
   set(stdlib_cxx "${stdlib_cxx} -static-libstdc++")
endif()


#
# Determine Optimization flags here
# These are quite aggresive flags, if your code misbehaves for strange reasons,
# try compiling without them.
#
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(optimization_flags_cc "${optimization_flags_cc} -mtune=generic -O2")
  set(optimization_flags_cc "-pipe ${optimization_flags_cc}") #TODO use -Ofast instead of -O3
  set(optimization_flags_lt "-O2 ${optimization_flags_lt}")

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND NOT MINGW)
    set(optimization_flags_cc "${optimization_flags_cc} -fuse-ld=gold")
    # NOTE -flto must go together in both cc and ld flags; also, it's presently incompatible
    # with the -g option in at least some GNU compilers (saw in `man gcc` on Ubuntu)
    set(optimization_flags_cc "${optimization_flags_cc} -fuse-linker-plugin -flto-report -flto") #TODO fix LTO for clang
    set(optimization_flags_lt "${optimization_flags_lt} -flto") #TODO LTO for clang too
  endif()
endif()


#
# compiler specific settings and warnings here
#

set(shared_compile_flags "")
set(internal_compiler_warning_flags "")
set(external_compiler_warning_flags "")
set(cxx_flags_unoptimized "")
set(shared_linker_flags_unoptimized "")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  # MS Visual C
  set(shared_compile_flags "${shared_compile_flags} /TP /Zc:wchar_t /Gm- /fp:precise /errorReport:prompt /W1 /WX- /GR /Gd /GS /Oy- /EHs /analyze- /nologo")
  set(shared_linker_flags_unoptimized "${shared_linker_flags_unoptimized} /NOLOGO /SAFESEH:NO /NODEFAULTLIB:LIBCMT")
  if("${BITNESS}" STREQUAL "32")
    set(shared_linker_flags_unoptimized "${shared_linker_flags_unoptimized} /MACHINE:X86")
  else()
    set(shared_linker_flags_unoptimized "${shared_linker_flags_unoptimized} /MACHINE:X${BITNESS}")
  endif()

else()
  # LLVM Clang / Gnu GCC
  set(cxx_flags_unoptimized "${cxx_flags_unoptimized} ${stdlib_cxx} -std=c++11")

  set(shared_compile_flags "${shared_compile_flags} -m${BITNESS} ${stdlib_common} -fdiagnostics-show-option")
  set (internal_compiler_warning_flags "${internal_compiler_warning_flags} -Werror -Wextra -Wreturn-type -Wunused -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers")
  set (external_compiler_warning_flags "${external_compiler_warning_flags} -Wno-unused-variable -Wno-unused-parameter -Wno-incompatible-pointer-types -Wno-deprecated-declarations")

  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(shared_compile_flags "${shared_compile_flags} -fPIC")
    set (internal_compiler_warning_flags "${internal_compiler_warning_flags} -Wall")

    if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
      set(shared_compile_flags "${shared_compile_flags} -Wno-deprecated-register")
    endif()
  endif()

  set(shared_linker_flags_unoptimized "${shared_linker_flags_unoptimized} -m${BITNESS} ${stdlib_common} ${stdlib_cxx}")
endif()


if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(shared_linker_flags_unoptimized "${shared_linker_flags_unoptimized} -Wl,--no-undefined")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(shared_linker_flags_unoptimized "${shared_linker_flags_unoptimized} -Wl,-undefined,error")
endif()


#
# Set up Debug vs. Release options
#
set(build_type_specific_compile_flags)
set(build_type_specific_linker_flags)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set (build_type_specific_compile_flags "${build_type_specific_compile_flags} -g")

  set(build_type_specific_linker_flags "${build_type_specific_linker_flags} -O0")

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR MINGW)
    set (build_type_specific_compile_flags "${build_type_specific_compile_flags} -Og")
  endif()

  # Disable optimizations
  set(optimization_flags_cc)
  set(optimization_flags_lt)
endif()


#
# Assemble compiler and linker properties
#

# Settings for internal nupic.core code
set(INTERNAL_CXX_FLAGS_OPTIMIZED "${build_type_specific_compile_flags} ${shared_compile_flags} ${cxx_flags_unoptimized} ${internal_compiler_warning_flags} ${optimization_flags_cc}")

set(complete_linker_flags_unoptimized "${build_type_specific_linker_flags} ${shared_linker_flags_unoptimized}")
set(INTERNAL_LINKER_FLAGS_OPTIMIZED "${complete_linker_flags_unoptimized} ${optimization_flags_lt}")


# Settings for third-party code and code generated by 3rd-party tools (e.g., Swig bindings)
# (NOTE we omit the explicit compiler warning-related flags here to avoid
#  polluting build output with warnings from code that we don't control)
set(EXTERNAL_C_FLAGS_UNOPTIMIZED "${build_type_specific_compile_flags} ${shared_compile_flags} ${external_compiler_warning_flags}")
set(EXTERNAL_C_FLAGS_OPTIMIZED "${EXTERNAL_C_FLAGS_UNOPTIMIZED} ${optimization_flags_cc}")

set(EXTERNAL_CXX_FLAGS_UNOPTIMIZED "${build_type_specific_compile_flags} ${shared_compile_flags} ${external_compiler_warning_flags} ${cxx_flags_unoptimized}")
set(EXTERNAL_CXX_FLAGS_OPTIMIZED "${EXTERNAL_CXX_FLAGS_UNOPTIMIZED} ${optimization_flags_cc}")

set(EXTERNAL_LINKER_FLAGS_UNOPTIMIZED "${complete_linker_flags_unoptimized}")
set(EXTERNAL_LINKER_FLAGS_OPTIMIZED "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")


#
# Provide a string variant of the COMMON_COMPILER_DEFINITIONS list
#
set(COMMON_COMPILER_DEFINITIONS_STR)
foreach(compiler_definition ${COMMON_COMPILER_DEFINITIONS})
  set(COMMON_COMPILER_DEFINITIONS_STR "${COMMON_COMPILER_DEFINITIONS_STR} ${compiler_definition}")
endforeach()
