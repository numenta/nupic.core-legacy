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
#   algorithms.so
#   engine_internal.so
#   math.so
#
#
# For PyBind11 info See: https://pybind11.readthedocs.io/en/stable/compiling.html
#######################################################################

cmake_minimum_required(VERSION 3.7)
project(nupic_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug
message(STATUS "Configuring Python interface")


#
# Python support
#
set(src_py_algorithms_files
    bindings/algorithms/algorithm_module.cpp
    bindings/algorithms/py_SDR.cpp
    bindings/algorithms/py_SDR_Metrics.cpp
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
	plugin/unittest/main.cpp
	plugin/unittest/PyRegionTest.cpp
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


# determine which version of binding to build
if ("${BINDING_BUILD}" STREQUAL "Python2")
  set(PYBIND11_PYTHON_VERSION 2.7)
  # else Python3
endif()
if (MSVC)
  set(PYBIND11_CPP_STANDARD -std:${INTERNAL_CPP_STANDARD})
else(MSVC)
  set(PYBIND11_CPP_STANDARD -std=${INTERNAL_CPP_STANDARD})
endif(MSVC)
add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
set(pybind11_INCLUDE_DIRS ${pybind11_SOURCE_DIR}/include)


# It is ready to use pybind11_add_module() macro.
# For details: https://pybind11.readthedocs.io/en/stable/compiling.html#cmake


set(algorithms_shared_lib algorithms)
pybind11_add_module(${algorithms_shared_lib} ${src_py_algorithms_files} )
target_link_libraries(${algorithms_shared_lib} PRIVATE 
		${core_library}
		${COMMON_OS_LIBS}
		)
target_compile_options(${algorithms_shared_lib} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${algorithms_shared_lib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${algorithms_shared_lib} PRIVATE 
		${PYTHON_INCLUDE_DIRS}
		${pybind11_INCLUDE_DIRS}
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		${PROJECT_SOURCE_DIR}
		)


set(engine_shared_lib engine_internal)
pybind11_add_module(${engine_shared_lib} ${src_py_engine_files})
target_link_libraries(${engine_shared_lib} PRIVATE
		${core_library}
		${COMMON_OS_LIBS}
		)
target_compile_options(${engine_shared_lib} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${engine_shared_lib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${engine_shared_lib} PRIVATE 
		${PYTHON_INCLUDE_DIRS}
		${pybind11_INCLUDE_DIRS}
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		${PROJECT_SOURCE_DIR}
		)

set(math_shared_lib math)
pybind11_add_module(${math_shared_lib} ${src_py_math_files})
target_link_libraries(${math_shared_lib} PRIVATE 
		${core_library}
		${COMMON_OS_LIBS}
		)
target_compile_options(${math_shared_lib} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${math_shared_lib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${math_shared_lib} PRIVATE 
		${PYTHON_INCLUDE_DIRS}
		${pybind11_INCLUDE_DIRS}
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		${PROJECT_SOURCE_DIR}
		)


############################################################
#
# Build tests of the Python interface
#
############################################################

### TODO: Finish these tests
#  
#set(py_unit_test_executable  py_unit_test)
#add_executable(${py_unit_test_executable} ${src_py_test_files})
#target_link_libraries(${py_unit_test_executable} 
#    ${core_library}
#    gtest
#    ${COMMON_OS_LIBS}
#)
#target_compile_definitions(${py_unit_test_executable} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
#target_compile_options(${py_unit_test_executable} PUBLIC ${INTERNAL_CXX_FLAGS})
#set_target_properties(${py_unit_test_executable} PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_STR}")
#add_dependencies(${py_unit_test_executable} ${core_library} gtest)
#target_include_directories(${py_unit_test_executable} PRIVATE 
#		${PYTHON_INCLUDE_DIRS}
#		${pybind11_INCLUDE_DIRS}
#		${CORE_LIB_INCLUDES} 
#		${EXTERNAL_INCLUDES}
#		${PROJECT_SOURCE_DIR}
#		${gtest_INCLUDE_DIR}
#		)




###################################################################
#
# Install targets into CMAKE_INSTALL_PREFIX/distr/src/nupic/bindings
#
###################################################################
install(TARGETS
        algorithms
        engine_internal
        math
	${py_unit_test_executable}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${distr}/src/nupic/bindings
        ARCHIVE DESTINATION lib)
	
