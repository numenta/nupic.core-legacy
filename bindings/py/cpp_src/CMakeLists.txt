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


include_directories(
  ${PROJECT_SOURCE_DIR}
  ${REPOSITORY_DIR}/external/common/include
)

# Hide all symbols in DLLs except the ones with explicit visibility;
# see https://gcc.gnu.org/wiki/Visibility
set(cxx_flags_unoptimized "${cxx_flags_unoptimized} -fvisibility-inlines-hidden -fvisibility=hidden")

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
    bindings/math/py_Set.cpp
    bindings/math/py_SparseBinaryMatrix.cpp
    bindings/math/py_SparseMatrix.cpp
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

# setup the distr directory
set(distr ${CMAKE_INSTALL_PREFIX}/distr)
file(MAKE_DIRECTORY ${distr})
file(COPY ../packaging/. DESTINATION ${distr} PATTERN *)

# set CPP standard. default for PyBind11 is C++14.  If requested standard not available it will fall back to C++11.
if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++latest) # Enables MSVC C++17
else()
  set(PYBIND11_CPP_STANDARD -std=${INTERNAL_CPP_STANDARD})
endif()

include_directories(${REPOSITORY_DIR}/src)
include_directories(${pybind11_BINARY_DIR})


if(PYTHON2_BUILD)
	# build for python 2.7.  (defaults to 3.x)
	set(PYBIND11_PYTHON_VERSION 2.7)
endif()

message(STATUS "Core lib to LINK is '${core_library}' ")


pybind11_add_module(algorithms ${src_py_algorithms_files} )
target_link_libraries(algorithms PRIVATE ${core_library})
target_compile_options(algorithms PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(algorithms PRIVATE ${COMMON_COMPILER_DEFINITIONS})

pybind11_add_module(engine_internal ${src_py_engine_files})
target_link_libraries(engine_internal PRIVATE ${core_library})
target_compile_options(engine_internal PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(engine_internal PRIVATE ${COMMON_COMPILER_DEFINITIONS})

pybind11_add_module(math ${src_py_math_files})
target_link_libraries(engine_internal PRIVATE ${core_library})
target_compile_options(engine_internal PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(engine_internal PRIVATE ${COMMON_COMPILER_DEFINITIONS})
############################################################
#
# Build tests of the Python interface
#
############################################################

#TODO: tests should be installed here


###################################################################
#
# Install targets into CMAKE_INSTALL_PREFIX/distr/src/nupic/bindings
#
###################################################################
install(TARGETS
        algorithms
        engine_internal
        math
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${distr}/src/nupic/bindings
        ARCHIVE DESTINATION lib)

