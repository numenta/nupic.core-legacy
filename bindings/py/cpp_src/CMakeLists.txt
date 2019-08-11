# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2013-2015, Numenta, Inc.
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
#
# This builds the Python language interface.  This consists of the following libs
#   algorithms.so
#   engine_internal.so
#   math.so
#   encoders.so
#   sdr.so
#
#
# For PyBind11 info See: https://pybind11.readthedocs.io/en/stable/compiling.html
#######################################################################

cmake_minimum_required(VERSION 3.7)
project(htm_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug
message(STATUS "Configuring Python interface")
message(STATUS "   installing in ${CMAKE_INSTALL_PREFIX}")

#
# Python support
#
set(src_py_algorithms_files
    bindings/algorithms/algorithm_module.cpp
    bindings/algorithms/py_Connections.cpp
    bindings/algorithms/py_TemporalMemory.cpp
    bindings/algorithms/py_SDRClassifier.cpp
    bindings/algorithms/py_SpatialPooler.cpp
    )

set(src_py_sdr_files
    bindings/sdr/sdr_module.cpp
    bindings/sdr/py_SDR.cpp
    bindings/sdr/py_SDR_Metrics.cpp
    )

set(src_py_encoders_files
    bindings/encoders/encoders_module.cpp
    bindings/encoders/py_ScalarEncoder.cpp
    bindings/encoders/py_RDSE.cpp
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
    bindings/math/py_Random.cpp
    bindings/math/py_reals.cpp
    bindings/math/py_Topology.cpp
    )
    
set(src_py_test_files
	plugin/unittest/main.cpp
	plugin/unittest/PyRegionTest.cpp
	)
	
#set up file tabs in Visual Studio
source_group("algorithms" FILES ${src_py_algorithms_files})
source_group("sdr"        FILES ${src_py_sdr_files})
source_group("encoders"   FILES ${src_py_encoders_files})
source_group("engine"     FILES ${src_py_engine_files})
source_group("math"       FILES ${src_py_math_files})
source_group("test"       FILES ${src_py_test_files})
############################################################
#
# Build the shared binding libraries for the Python Interface
#
############################################################
#
# Note: if you need a debuggable version of the Python interface 
#       See: https://bugs.python.org/issue22411
#       and: https://github.com/pybind/pybind11/issues/1295
#          

# setup the distr directory
set(distr ${CMAKE_INSTALL_PREFIX}/distr)
file(MAKE_DIRECTORY ${distr})
file(COPY ${REPOSITORY_DIR}/py/. DESTINATION ${distr}/src PATTERN *)
file(COPY ../packaging/. DESTINATION ${distr} PATTERN *)

# determine which version of binding to build
if ("${BINDING_BUILD}" STREQUAL "Python2")
  set(PYBIND11_PYTHON_VERSION 2.7)
else()
  set(BINDING_BUILD Python3)
endif()
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


set(sdr_shared_lib sdr)
pybind11_add_module(${sdr_shared_lib} ${src_py_sdr_files} )
target_link_libraries(${sdr_shared_lib} PRIVATE
        ${core_library}
        ${COMMON_OS_LIBS}
        )
target_compile_options(${sdr_shared_lib} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${sdr_shared_lib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${sdr_shared_lib} PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        ${CORE_LIB_INCLUDES}
        ${EXTERNAL_INCLUDES}
        ${PROJECT_SOURCE_DIR}
        )


set(encoders_shared_lib encoders)
pybind11_add_module(${encoders_shared_lib} ${src_py_encoders_files} )
target_link_libraries(${encoders_shared_lib} PRIVATE 
        ${core_library}
        ${COMMON_OS_LIBS}
        )
target_compile_options(${encoders_shared_lib} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${encoders_shared_lib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${encoders_shared_lib} PRIVATE 
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

install(TARGETS
        sdr
        encoders
        algorithms
        engine_internal
        math
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${distr}/src/htm/bindings
        ARCHIVE DESTINATION lib)
	

