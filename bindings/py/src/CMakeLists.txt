# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have purchased from
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
#
# This builds the Python language interface.  This consists of the following libs
# for Python2.7
#   nupic.core.algorithms2.pyd
#   nupic.core.engine2.pyd
#   nupic.core.math2.pyd
#
# for Python3
#   nupic.core.algorithms.pyd
#   nupic.core.engine.pyd
#   nupic.core.math.pyd
#
# For PyBind11 info See: https://pybind11.readthedocs.io/en/stable/compiling.html
#######################################################################

cmake_minimum_required(VERSION 3.7)
project(nupic_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug
message(STATUS "Configuring Python interface")

#### TODO: do we need this?
#	if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND "${PLATFORM}" STREQUAL "linux")
		# NOTE When building manylinux python extensions, we want the static
		# libstdc++ due to differences in c++ ABI between the older toolchain in the
		# manylinux Docker image and libstdc++ in newer linux distros that is
		# compiled with the c++11 ABI. for example, with shared libstdc++, the
		# manylinux-built extension is unable to catch std::ios::failure exception
		# raised by the shared libstdc++.so while running on Ubuntu 16.04.
#		set(stdlib_cxx "${stdlib_cxx} -static-libstdc++")

		# NOTE We need to use shared libgcc to be able to throw and catch exceptions
		# across different shared libraries.
#		set(stdlib_common "${stdlib_common} -shared-libgcc")
#	endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${REPOSITORY_DIR}/external/common/include
)

### TODO: do we need this?
#	if (${NUPIC_BUILD_PYEXT_MODULES})
#		# Hide all symbols in DLLs except the ones with explicit visibility;
#		# see https://gcc.gnu.org/wiki/Visibility
#		set(cxx_flags_unoptimized "${cxx_flags_unoptimized} -fvisibility-inlines-hidden -fvisibility=hidden")
#	endif()

#
# Python support
#
set(src_py_algorithms_files
    bindings/algorithms/algorithm_module.cpp
    bindings/algorithms/py_Algorithms.cpp
    bindings/algorithms/py_Cells4.cpp
    bindings/algorithms/py_HTM.cpp
    bindings/algorithms/py_SDRClassifier.cpp
    bindings/algorithms/py_SpatialPooler.cpp
	)
	
set(src_py_engine_files
    plugin/PyBindRegion.cpp
    plugin/PyBindRegion.hpp
    plugin/RegisteredRegionImplPy.hpp
    bindings/engine/engine_module.cpp
    bindings/engine/py_Engine.cpp
    bindings/engine/py_Region.cpp
    bindings/engine/py_OS.cpp
    bindings/engine/py_Timer.cpp
    bindings/engine/py_utils.hpp
	)
	
set(src_py_math_files
    bindings/math/math_module.cpp    
    bindings/math/py_ArrayAlgo.cpp
    bindings/math/py_Domain.cpp
    bindings/math/py_Math.cpp
    bindings/math/py_Random.cpp
    bindings/math/py_reals.cpp
    bindings/math/py_SegmentSparseMatrix.cpp
    bindings/math/py_Set.cpp
#    bindings/math/py_SparseBinaryMatrix.cpp
#    bindings/math/py_SparseMatrix.cpp
    bindings/math/py_SparseTensor.cpp
    bindings/math/PyBindSparseTensor.hpp
    bindings/math/SparseTensorIndex.hpp
    )
    
set(src_py_test_files
	tests/main.cpp
	tests/PyRegionTest.cpp
	)
	
#set up file tabs in Visual Studio
source_group("algorithms" FILES ${src_py_algorithms_files})
source_group("engine" FILES ${src_py_engine_files})
source_group("math" FILES ${src_py_math_files})
source_group("test" FILES ${src_py_test_files})
############################################################
#
# Build the three shared binding libraries for the Python Interface
#
############################################################


# set CPP standard. default for PyBind11 is C++14.  If requested standard not available it will fall back to C++11.
if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++latest) # Enables MSVC C++17
else()
  set(PYBIND11_CPP_STANDARD -std=${INTERNAL_CPP_STANDARD})
endif()

include_directories(${REPOSITORY_DIR}/src)
include_directories(${pybind11_BINARY_DIR})


if(PYTHON2_BUILD)
	# build for python 2.7
	set(PYBIND11_PYTHON_VERSION 2.7)
	pybind11_add_module(nupic.core.algorithms2 
						${src_py_algorithms_files}
						${core_library}
						)
	pybind11_add_module(nupic.core.engine2 
						${src_py_engine_files}
						${core_library}
						)
	pybind11_add_module(nupic.core.math2 
						${src_py_math_files}
						${core_library}
						)
endif()

if(PYTHON3_BUILD)
	# build for python 3.x
	#set(PYBIND11_PYTHON_VERSION)  # defaults to 3.x
	pybind11_add_module(nupic.core.algorithms 
						${src_py_algorithms_files}
						${core_library}
						)
	pybind11_add_module(nupic.core.engine 
						${src_py_engine_files}
						${core_library}
						)
	pybind11_add_module(nupic.core.math 
						${src_py_math_files}
						${core_library}
						)
endif()
############################################################
#
# Build tests of the Python interface
#
############################################################

#TODO:


###################################################################
#
# Install targets into CMAKE_INSTALL_PREFIX
#
###################################################################
if(PYTHON2_BUILD)
  install(TARGETS
        nupic.core.algorithms2
        nupic.core.engine2
        nupic.core.math2
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()
if(PYTHON3_BUILD)
  install(TARGETS
        nupic.core.algorithms
        nupic.core.engine
        nupic.core.math
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()
