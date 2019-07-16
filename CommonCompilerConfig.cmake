# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2013-2018, Numenta, Inc.
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
# -----------------------------------------------------------------------------


# Configures common compiler/linker/loader settings for internal and external
# sources.
#
# NOTE SETTINGS THAT ARE SPECIFIC TO THIS OR THAT MODULE DO NOT BELONG HERE.

# INPUTS:
#	CMAKE_CXX_STANDARD  i.e. (C++) 11, 14, 17; defaults to C++11 or C++17
#	BITNESS   32,64, defaults to bitness of current machine.
#	PLATFORM:   defaults to ${CMAKE_SYSTEM_NAME}  
#	CMAKE_BUILD_TYPE   Debug, Release   defaults to Release

# OUTPUTS:
#
#	PLATFORM:   lowercase
#	BITNESS: Platform bitness: 32 or 64
#
#	COMMON_COMPILER_DEFINITIONS: list of -D define flags for the compilation of
#                               source files; e.g., for cmake `add_definitions()`
#	COMMON_COMPILER_DEFINITIONS_STR: string variant of COMMON_COMPILER_DEFINITIONS
#
# 	INTERNAL_CXX_FLAGS: list of C++ flags common to both release and debug.  They do contain 'generator' statements.
#                     so make sure the CMake function you use will process them.
#		      Usable as target_compile_options(target PUBLIC ${INTERNAL_CXX_FLAGS})
#
# 	INTERNAL_LINKER_FLAGS: string of linker flags for linking internal executables
#                      and shared libraries (DLLs) with optimizations that are
#                      compatible with INTERNAL_CXX_FLAGS
#
# 	COMMON_OS_LIBS: the list of common runtime libraries to use for this OS.
#
# 	CMAKE_AR: Name of archiving tool (ar) for static libraries. See cmake documentation
#
# 	CMAKE_RANLIB: Name of randomizing tool (ranlib) for static libraries. See cmake documentation
#
# 	CMAKE_LINKER: updated, if needed; use ld.gold if available. See cmake documentation
#
#	EXTERNAL_STATICLIB_CONFIGURE_DEFINITIONS_OPTIMIZED
#
#
# USAGE:
# Recommended, do this for each target foo
#   	target_compile_options(foo PUBLIC "${INTERNAL_CXX_FLAGS}")
#   	target_compile_definitions(foo PRIVATE ${COMMON_COMPILER_DEFINITIONS})
#   	set_target_properties(foo PROPERTIES LINK_FLAGS ${INTERNAL_LINKER_FLAGS})
# Add any module specific options such as /DLL, etc.
##############################################################

include(CheckCXXCompilerFlag)


# Identify platform name.
if(NOT PLATFORM)
  set(PLATFORM  ${CMAKE_SYSTEM_NAME})
endif()
string(TOLOWER ${PLATFORM} PLATFORM)


# Set the C++ standard version
#    Compiler support for <filesystem> in C++17:
#	https://en.cppreference.com/w/cpp/compiler_support
#       https://en.wikipedia.org/wiki/Xcode#Latest_versions
#
#	GCC 7.1 has <experimental/filesystem>, link with -libc++experimental or -lstdc++fs
#	GCC 8 has <filesystem>   link with -lstdc++fs
#	GCC 9   expected to support <filesystem>
#       AppleClang as of (XCode 10.1) does not support C++17 or filesystem
#           (although you can get llvm 7 from brew)
#	Clang 7 has complete <filesystem> support for C++17, link with -lc++fs (cmake "stdc++fs" library)
#       Clang 9 has complete <filesystem> support for C++17 by default. 
#	Visual Studio 2017 15.7 (v19.14)supports <filesystem> with C++17
#	MinGW has no support for filesystem.
#
# If we have support for <filesystem> and C++17, turn on the C++17 standard flag, 
# else set standard to C++11 and install the boost filesystem
# Also specify the external library for <filesystem> if needed.
# 

set(extra_lib_for_filesystem)   # sometimes -libc++experimental or -lstdc++fs
set(CMAKE_CXX_STANDARD 11) # -std=c++11 by default
set(boost_required ON)

if(NOT FORCE_CPP11)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9")
         set(CMAKE_CXX_STANDARD 17)
	 set(boost_required OFF)
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "8")
         set(CMAKE_CXX_STANDARD 17)
	 set(extra_lib_for_filesystem "stdc++fs")
	 set(boost_required "OFF")
    endif()	 
  elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "AppleClang")  # see CMake Policy CMP0025
    # does not support C++17 and filesystem (as of XCode 10.1)
  elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang") # clang + std::filesystem, see https://libcxx.llvm.org/docs/UsingLibcxx.html#using-filesystem
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7")
         set(CMAKE_CXX_STANDARD 17)
	 set(boost_required OFF)
      if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "7") # special library for older clang-7
        set(extra_lib_for_filesystem "stdc++fs")
      endif()
    endif()
  elseif(MSVC)
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.14")
            set(CMAKE_CXX_STANDARD 17)
	    set(boost_required OFF)
      endif()
  endif()
endif()
if (boost_required)
  set(NEEDS_BOOST ON)
else()
  # otherwise honors the override from parent.
  set(NEEDS_BOOST ${FORCE_BOOST})
endif()

set_property(GLOBAL PROPERTY CXX_STANDARD_REQUIRED ON)


# Identify platform "bitness".
if(NOT BITNESS)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	  set(BITNESS 64)
	else()
	  set(BITNESS 32)
	endif()
endif()



# Init exported properties
set(COMMON_COMPILER_DEFINITIONS)
set(INTERNAL_CXX_FLAGS_OPTIMIZED)
set(INTERNAL_LINKER_FLAGS_OPTIMIZED)
set(COMMON_OS_LIBS)


if(MSVC)
	# MS Visual C
	# on Windows using Visual Studio 2015, 2017, 2019   https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options-listed-by-category
	#  /permissive- forces standards behavior.  See https://docs.microsoft.com/en-us/cpp/build/reference/permissive-standards-conformance?view=vs-2017
	#  /Zc:__cplusplus   This is required to force MSVC to pay attention to the standard setting and sets __cplusplus.
	#                    NOTE: MSVC does not support C++11.  But does support C++14 and C++17.
	# Release Compiler flags:
	#	Common Stuff:  /permissive- /W3 /Gy /Gm- /O2 /Oi /EHsc /FC /nologo /Zc:__cplusplus
	#      Release Only:    /O2 /Oi /Gy  /MD
	#      Debug Only:       /Od /Zi /sdl /RTC1 /MDd
	set(INTERNAL_CXX_FLAGS /permissive- /W3 /Gm- /EHsc /FC /nologo /Zc:__cplusplus
							$<$<CONFIG:Release>:/O2 /Oi /Gy  /GL /MD> 
							$<$<CONFIG:Debug>:/Ob0 /Od /Zi /sdl /RTC1 /MDd>)
	#linker flags
	if("${BITNESS}" STREQUAL "32")
		set(machine "-MACHINE:X86")
	else()
		set(machine "-MACHINE:X${BITNESS}")
	endif()
	set(INTERNAL_LINKER_FLAGS ${machine} -NOLOGO -NODEFAULTLIB:LIBCMT -ignore:4099 $<$<CONFIG:Release>:-LTCG>)

	set(COMMON_COMPILER_DEFINITIONS 	
		_CONSOLE
		_MBCS
		NTA_OS_WINDOWS
		NTA_COMPILER_MSVC
		NTA_ARCH_${BITNESS}
		_CRT_SECURE_NO_WARNINGS
		_SCL_SECURE_NO_WARNINGS
		_CRT_NONSTDC_NO_DEPRECATE
		_SCL_SECURE_NO_DEPRECATE
		BOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE
		BOOST_ALL_NO_LIB
		BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
		BOOST_NO_WREGEX
		_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
		VC_EXTRALEAN
		WIN32_LEAN_AND_MEAN
		NOMINMAX
		NOGDI
		)

	set(COMMON_COMPILER_DEFINITIONS ${COMMON_COMPILER_DEFINITIONS} $<$<CONFIG:Debug>:NTA_ASSERTIONS_ON>)
		
	# common libs
	# Libraries linked by defaultwith all C++ applications
	# CMAKE_CXX_STANDARD_LIBRARIES:STRING=kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
        # Identify any additional system libs
	set(COMMON_OS_LIBS oldnames.lib psapi.lib ws2_32.lib)


else()
	# anything other than MSVC
	

	# Compiler `-D*` definitions
	#
	# Compiler definitions specific to htm.core code
	#
	string(TOUPPER ${PLATFORM} platform_uppercase)

	set(${COMMON_COMPILER_DEFINITIONS}
		-DNTA_OS_${platform_uppercase}
		-DNTA_ARCH_${BITNESS}
		-DHAVE_CONFIG_H
		-DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
		-DBOOST_NO_WREGEX
		)

	if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	  set(COMMON_COMPILER_DEFINITIONS ${COMMON_COMPILER_DEFINITIONS} -DNTA_ASSERTIONS_ON)
	endif()

	if(UNIX) # or UNIX like (i.e. APPLE and CYGWIN)
	  set(COMMON_COMPILER_DEFINITIONS ${COMMON_COMPILER_DEFINITIONS} -DHAVE_UNISTD_H)
	endif()

	if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	  set(COMMON_COMPILER_DEFINITIONS ${COMMON_COMPILER_DEFINITIONS} -DNTA_COMPILER_GNU)
	elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	  set(COMMON_COMPILER_DEFINITIONS ${COMMON_COMPILER_DEFINITIONS} -DNTA_COMPILER_CLANG)
	elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MinGW")
	  set(COMMON_COMPILER_DEFINITIONS ${COMMON_COMPILER_DEFINITIONS} -DNTA_COMPILER_GNU -D_hypot=hypot)
	endif()

	#
	# Set linker (ld)
	# These linkers are tried for faster linking performance
	# use ld.gold, or lld if available
	#
	execute_process(COMMAND ld.gold --version RESULT_VARIABLE EXIT_CODE_GOLD)
	if(EXIT_CODE_GOLD EQUAL 0)
	  message("Using ld.gold as LINKER.")
	  set(CMAKE_LINKER "ld.gold")
	  set(optimization_flags_cc ${optimization_flags_cc} -fuse-ld=gold)
	endif()
	execute_process(COMMAND ld.lld --version RESULT_VARIABLE EXIT_CODE_LLD)
	execute_process(COMMAND ld.lld-9 --version RESULT_VARIABLE EXIT_CODE_LLD9)
        if(EXIT_CODE_LLD EQUAL 0 OR EXIT_CODE_LLD9 EQUAL 0)
          message("Using ld.lld as LINKER.")
          set(CMAKE_LINKER "ld.lld")
          set(optimization_flags_cc ${optimization_flags_cc} -fuse-ld=lld)
        endif()


	#
	# Determine stdlib settings
	#
	set(stdlib_cxx)
	set(stdlib_common)

	if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	  set(stdlib_cxx ${stdlib_cxx} -stdlib=libc++)
	endif()

# TODO: investigate if we should use static or shared stdlib and gcc lib.
	if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
		set(stdlib_common ${stdlib_common} -static-libgcc)
		set(stdlib_cxx ${stdlib_cxx} -static-libstdc++)
	endif()



	#
	# compiler specific settings and warnings here
	#
	set(internal_compiler_warning_flags)
	set(cxx_flags_unoptimized)
	set(linker_flags_unoptimized)

	# Hide all symbols in DLLs except the ones with explicit visibility;
        # see https://gcc.gnu.org/wiki/Visibility
        set(cxx_flags_unoptimized ${cxx_flags_unoptimized} -fvisibility-inlines-hidden )


	# LLVM Clang / Gnu GCC
	set(cxx_flags_unoptimized ${cxx_flags_unoptimized} ${stdlib_cxx})

	set(cxx_flags_unoptimized ${cxx_flags_unoptimized} ${stdlib_common} -fdiagnostics-show-option)
	set (internal_compiler_warning_flags ${internal_compiler_warning_flags} -Werror -Wextra -Wreturn-type -Wunused -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers)

	CHECK_CXX_COMPILER_FLAG(-m${BITNESS} compiler_supports_machine_option)
	if (compiler_supports_machine_option)
		set(cxx_flags_unoptimized ${cxx_flags_unoptimized} -m${BITNESS})
		set(linker_flags_unoptimized ${linker_flags_unoptimized} -m${BITNESS})
	endif()
	if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
		set(cxx_flags_unoptimized ${cxx_flags_unoptimized} -marm)
		set(linker_flags_unoptimized ${linker_flags_unoptimized} -marm)
	endif()

	if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
		set(cxx_flags_unoptimized ${cxx_flags_unoptimized} -fPIC)
		set (internal_compiler_warning_flags ${internal_compiler_warning_flags} -Wall)

		if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
		  set(cxx_flags_unoptimized ${cxx_flags_unoptimized} -Wno-deprecated-register)
		endif()
	endif()

	set(shared_linker_flags_unoptimized ${shared_linker_flags_unoptimized} ${stdlib_common} ${stdlib_cxx})

	# Don't allow undefined symbols when linking executables
	if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	  set(linker_flags_unoptimized ${linker_flags_unoptimized} -Wl,--no-undefined)
	elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	  set(linker_flags_unoptimized ${linker_flags_unoptimized} -Wl,-undefined,error)
	endif()


	# Compatibility with gcc >= 4.9 which requires the use of gcc's own wrappers for
	# ar and ranlib in combination with LTO works also with LTO disabled
	IF(UNIX AND CMAKE_COMPILER_IS_GNUCXX AND (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug") AND
		  CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "4.9")
		set(CMAKE_AR "gcc-ar")
		set(CMAKE_RANLIB "gcc-ranlib")
		# EXTERNAL_STATICLIB_CMAKE_DEFINITIONS_OPTIMIZED duplicates settings for
		# CMAKE_AR and CMAKE_RANLIB. This is a workaround for a CMAKE bug
		# (https://gitlab.kitware.com/cmake/cmake/issues/15547) that prevents
		# the correct propagation of CMAKE_AR and CMAKE_RANLIB variables to all
		# externals
		list(APPEND EXTERNAL_STATICLIB_CMAKE_DEFINITIONS_OPTIMIZED
			 -DCMAKE_AR:PATH=gcc-ar
			 -DCMAKE_RANLIB:PATH=gcc-ranlib)
		# And ditto for externals that use the configure-based build system
		list(APPEND EXTERNAL_STATICLIB_CONFIGURE_DEFINITIONS_OPTIMIZED
			 AR=gcc-ar
			 RANLIB=gcc-ranlib)
	ENDIF()


        #
        # set OPTIMIZATION flags
	#
	#TODO: CMake automatically generates optimisation flags. Do we need this? - "I think yes ~breznak"
        set(optimization_flags_cc ${optimization_flags_cc} -pipe -O3)
        set(optimization_flags_lt ${optimization_flags_lt} -O3)
        if(NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
                set(optimization_flags_cc ${optimization_flags_cc} -mtune=generic)
        endif()
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND NOT MINGW)
                # NOTE -flto must go together in both cc and ld flags; also, it's presently incompatible
                # with the -g option in at least some GNU compilers (saw in `man gcc` on Ubuntu)
                set(optimization_flags_cc ${optimization_flags_cc} -fuse-linker-plugin -flto-report -flto -fno-fat-lto-objects) #TODO fix LTO for clang
                set(optimization_flags_lt ${optimization_flags_lt} -flto -fno-fat-lto-objects) #TODO LTO for clang too
        endif()


	#
	# Set up Debug vs. Release options
	#
	set(debug_specific_compile_flags)
	set(debug_specific_linker_flags)

	if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	  set (debug_specific_compile_flags ${debug_specific_compile_flags} -g)

	  set(debug_specific_linker_flags ${debug_specific_linker_flags} -O0)

	  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR MINGW)
		set (debug_specific_compile_flags ${debug_specific_compile_flags} -Og)

		# Enable diagnostic features of standard class templates, including ability
		# to examine containers in gdb.
		# See https://gcc.gnu.org/onlinedocs/libstdc++/manual/debug_mode_using.html
		list(APPEND COMMON_COMPILER_DEFINITIONS -D_GLIBCXX_DEBUG)
	  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
		# NOTE: debug mode is immature in Clang, and values of _LIBCPP_DEBUG above 0
		# require  the debug build of libc++ to be present at linktime on OS X.
		list(APPEND COMMON_COMPILER_DEFINITIONS -D_LIBCPP_DEBUG=0)
	  endif()

	  # Disable optimizations
	  set(optimization_flags_cc)
	  set(optimization_flags_lt)
	endif()


	#
	# Assemble compiler and linker properties
	#

	# Settings for internal htm.core code
	set(INTERNAL_CXX_FLAGS ${debug_specific_compile_flags} ${cxx_flags_unoptimized} ${internal_compiler_warning_flags} ${optimization_flags_cc})
	set(INTERNAL_LINKER_FLAGS ${debug_specific_linker_flags} ${linker_flags_unoptimized} ${optimization_flags_lt})
	
	#
	# Common system libraries for shared libraries and executables
	#
	set(COMMON_OS_LIBS ${extra_lib_for_filesystem})

	if("${PLATFORM}" STREQUAL "linux")
	  list(APPEND COMMON_OS_LIBS pthread dl)
	elseif("${PLATFORM}" STREQUAL "darwin")
	  list(APPEND COMMON_OS_LIBS c++abi)
	elseif(MSYS OR MINGW)
	  list(APPEND COMMON_OS_LIBS psapi ws2_32 wsock32 rpcrt4)
	endif()

endif()


#
# Provide a string variant of the COMMON_COMPILER_DEFINITIONS list
#
string (REPLACE ";" " " COMMON_COMPILER_DEFINITIONS_STR "${COMMON_COMPILER_DEFINITIONS}")

# Provide a string variant of the INTERNAL_CXX_FLAGS list
string (REPLACE ";" " " INTERNAL_CXX_FLAGS_STR "${INTERNAL_CXX_FLAGS}")

# Provide a string variant of the INTERNAL_LINKER_FLAGS list
string (REPLACE ";" " " INTERNAL_LINKER_FLAGS_STR "${INTERNAL_LINKER_FLAGS}")
set_property(GLOBAL PROPERTY LINK_LIBRARIES ${INTERNAL_LINKER_FLAGS})


