# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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

cmake_minimum_required(VERSION 3.7)
project(nupic_core CXX)

message(STATUS "Configuring nupic_core tests")
###############################################################
###                   TESTS                      ##############
###############################################################
#
# Build TESTS of the nupic core static library
#   ${core_library}  references the nupic_core  static library which includes depencancy libraries
#

set(CMAKE_VERBOSE_MAKEFILE ON) # toggle for cmake debug 



#  Build unit_tests
set(unit_tests_executable unit_tests)

set(algorithm_tests
	   unit/algorithms/AnomalyTest.cpp
	   unit/algorithms/Cells4Test.cpp
	   unit/algorithms/ConnectionsPerformanceTest.cpp
	   unit/algorithms/ConnectionsTest.cpp
	   unit/algorithms/HelloSPTPTest.cpp
	   unit/algorithms/SDRClassifierTest.cpp
	   unit/algorithms/SegmentTest.cpp
	   unit/algorithms/SpatialPoolerTest.cpp
	   unit/algorithms/TemporalMemoryTest.cpp
	   )
               
set(encoders_tests
           unit/encoders/ScalarEncoderTest.cpp
           )
	   
set(engine_tests
	   unit/engine/CppRegionTest.cpp
	   unit/engine/HelloRegionTest.cpp
	   unit/engine/InputTest.cpp
	   unit/engine/LinkTest.cpp
	   unit/engine/NetworkTest.cpp
           unit/engine/UniformLinkPolicyTest.cpp
	   unit/engine/YAMLUtilsTest.cpp
	   )
	   

set(math_tests
	   unit/math/DenseTensorUnitTest.cpp
	   unit/math/DenseTensorUnitTest.hpp
	   unit/math/DomainUnitTest.cpp
	   unit/math/DomainUnitTest.hpp
	   unit/math/IndexUnitTest.cpp
	   unit/math/IndexUnitTest.hpp
	   unit/math/MathsTest.cpp
	   unit/math/MathsTest.hpp
	   unit/math/SparseBinaryMatrixTest.cpp
	   unit/math/SparseMatrixTest.cpp
	   unit/math/SparseMatrixUnitTest.cpp
	   unit/math/SparseMatrixUnitTest.hpp
	   unit/math/SparseTensorUnitTest.cpp
	   unit/math/SparseTensorUnitTest.hpp
	   unit/math/TopologyTest.cpp
	   )
	   
set(ntypes_tests
	   unit/ntypes/ArrayTest.cpp
	   unit/ntypes/CollectionTest.cpp
	   unit/ntypes/DimensionsTest.cpp
	   unit/ntypes/MemParserTest.cpp
	   unit/ntypes/MemStreamTest.cpp
	   unit/ntypes/ScalarTest.cpp
	   unit/ntypes/ValueTest.cpp
	   )
	   
set(os_tests
	   unit/os/DirectoryTest.cpp
	   unit/os/EnvTest.cpp
	   unit/os/OSTest.cpp
	   unit/os/PathTest.cpp
	   unit/os/TimerTest.cpp
	   )
	   
set(types_tests
	   unit/types/BasicTypeTest.cpp
	   unit/types/ExceptionTest.cpp
	   unit/types/FractionTest.cpp
	   unit/types/SdrTest.cpp
	   unit/types/SdrProxyTest.cpp
	   )
	   
set(utils_tests
	   unit/utils/GroupByTest.cpp
	   unit/utils/MovingAverageTest.cpp
	   unit/utils/RandomTest.cpp
	   unit/utils/VectorHelpersTest.cpp
	   unit/utils/SdrMetricsTest.cpp
	   unit/utils/WatcherTest.cpp
	   )
	   

	   
#set up file tabs in Visual Studio
source_group("algorithm" FILES ${algorithm_tests})
source_group("encoders" FILES ${encoders_tests})
source_group("engine" FILES ${engine_tests})
source_group("math" FILES ${math_tests})
source_group("ntypes" FILES ${ntypes_tests})
source_group("os" FILES ${os_tests})
source_group("types" FILES ${types_tests})
source_group("utils" FILES ${utils_tests})
source_group("main" FILES unit/UnitTestMain.cpp)


set(src_executable_gtests
    unit/UnitTestMain.cpp
    ${algorithm_tests} 
    ${encoders_tests} 
    ${engine_tests} 
    ${math_tests} 
    ${ntypes_tests} 
    ${os_tests} 
    ${types_tests} 
    ${utils_tests} 
)

add_executable(${unit_tests_executable} ${src_executable_gtests})
target_link_libraries(${unit_tests_executable} 
    ${core_library}
    ${gtest_LIBRARIES}
    ${COMMON_OS_LIBS}
)
target_include_directories(${unit_tests_executable} PRIVATE 
	${gtest_INCLUDE_DIRS}
	${CORE_LIB_INCLUDES}
	${EXTERNAL_INCLUDES})
target_compile_definitions(${unit_tests_executable} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_compile_options(${unit_tests_executable} PUBLIC ${INTERNAL_CXX_FLAGS})
set_target_properties(${unit_tests_executable} PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_STR}")
add_dependencies(${unit_tests_executable} ${core_library})


# Create the RUN_TESTS target
#  This displays its output differently in Visual Studio
enable_testing()
add_test(NAME ${unit_tests_executable} COMMAND ${unit_tests_executable})

                  
		  
		  

#
# tests_all just calls other targets
#
# TODO This doesn't seem to have any effect; it's probably because the DEPENDS
# of add_custom_target must be files, not other high-level targets. If really
# need to run these tests during build, then either the individual
# add_custom_target of the individual test runners should be declared with the
# ALL option, or tests_all target whould be declared without DEPENDS, and
# add_dependencies should be used to set it's dependencies on the custom targets
# of the inidividual test runners.
add_custom_target(tests_all
                  DEPENDS ${unit_tests_executable}
                  COMMENT "Running all tests"
                  VERBATIM)
                  
install(TARGETS
        ${unit_tests_executable}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

