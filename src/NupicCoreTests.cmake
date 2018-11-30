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


###############################################################
######################                   TESTS                      ######################
###############################################################
#
# Build TESTS of the nupic core static library
#   ${core_library}  references the nupic_core  static library which includes depencancy libraries
#

set(CMAKE_VERBOSE_MAKEFILE ON) # toggle for cmake debug 


###############
#  Build unit_tests
set(unit_tests_executable unit_tests)

set(algorithm_tests
	   test/unit/algorithms/AnomalyTest.cpp
	   test/unit/algorithms/Cells4Test.cpp
	   test/unit/algorithms/CondProbTableTest.cpp
	   test/unit/algorithms/ConnectionsPerformanceTest.cpp
	   test/unit/algorithms/ConnectionsTest.cpp
	   test/unit/algorithms/HelloSPTPTest.cpp

	   test/unit/algorithms/NearestNeighborUnitTest.cpp
	   test/unit/algorithms/SDRClassifierTest.cpp
	   test/unit/algorithms/SegmentTest.cpp
	   test/unit/algorithms/Serialization.cpp
	   test/unit/algorithms/SpatialPoolerTest.cpp
	   test/unit/algorithms/TemporalMemoryTest.cpp
	   )
               
set(encoders_tests
       test/unit/encoders/ScalarEncoderTest.cpp
       )
	   
set(engine_tests
	   test/unit/engine/CppRegionTest.cpp
	   test/unit/engine/HelloRegionTest.cpp
	   test/unit/engine/InputTest.cpp
	   test/unit/engine/LinkTest.cpp
	   test/unit/engine/NetworkTest.cpp
       test/unit/engine/UniformLinkPolicyTest.cpp
	   test/unit/engine/YAMLUtilsTest.cpp
	   )
	   

set(math_tests
	   test/unit/math/DenseTensorUnitTest.cpp
	   test/unit/math/DenseTensorUnitTest.hpp
	   test/unit/math/DomainUnitTest.cpp
	   test/unit/math/DomainUnitTest.hpp
	   test/unit/math/IndexUnitTest.cpp
	   test/unit/math/IndexUnitTest.hpp
	   test/unit/math/MathsTest.cpp
	   test/unit/math/MathsTest.hpp
	   test/unit/math/SegmentMatrixAdapterTest.cpp
	   test/unit/math/SparseBinaryMatrixTest.cpp
	   test/unit/math/SparseMatrix01UnitTest.cpp
	   test/unit/math/SparseMatrix01UnitTest.hpp
	   test/unit/math/SparseMatrixTest.cpp
	   test/unit/math/SparseMatrixUnitTest.cpp
	   test/unit/math/SparseMatrixUnitTest.hpp
	   test/unit/math/SparseTensorUnitTest.cpp
	   test/unit/math/SparseTensorUnitTest.hpp
	   test/unit/math/TopologyTest.cpp
	   )
	   
set(ntypes_tests
	   test/unit/ntypes/ArrayTest.cpp
	   test/unit/ntypes/BufferTest.cpp
	   test/unit/ntypes/CollectionTest.cpp
	   test/unit/ntypes/DimensionsTest.cpp
	   test/unit/ntypes/MemParserTest.cpp
	   test/unit/ntypes/MemStreamTest.cpp
	   test/unit/ntypes/NodeSetTest.cpp
	   test/unit/ntypes/ScalarTest.cpp
	   test/unit/ntypes/ValueTest.cpp
	   )
	   
set(os_tests
	   test/unit/os/DirectoryTest.cpp
	   test/unit/os/EnvTest.cpp
	   test/unit/os/OSTest.cpp
	   test/unit/os/PathTest.cpp
	   test/unit/os/RegexTest.cpp
	   test/unit/os/TimerTest.cpp
	   )
	   
set(types_tests
	   test/unit/types/BasicTypeTest.cpp
	   test/unit/types/ExceptionTest.cpp
	   test/unit/types/FractionTest.cpp
	   )
	   
set(utils_tests
	   test/unit/utils/GroupByTest.cpp
	   test/unit/utils/MovingAverageTest.cpp
	   test/unit/utils/RandomTest.cpp
	   test/unit/utils/VectorHelpersTest.cpp
	   test/unit/utils/WatcherTest.cpp
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


set(src_executable_gtests
    test/unit/UnitTestMain.cpp
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
    gtest
    ${COMMON_OS_LIBS}
    -OPT:NOREF
)
target_include_directories(${unit_tests_executable} PUBLIC ${gtest_INCLUDE_DIR})
target_compile_definitions(${unit_tests_executable} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_compile_options(${unit_tests_executable} PUBLIC "${INTERNAL_CXX_FLAGS}")
set_target_properties(${unit_tests_executable} PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS}")
add_dependencies(${unit_tests_executable} ${core_library} gtest)

# Create the RUN_TESTS target
enable_testing()
add_test(NAME ${unit_tests_executable} COMMAND ${unit_tests_executable})

add_custom_target(unit_tests_run_with_output
                  COMMAND ${unit_tests_executable}
                  DEPENDS ${unit_tests_executable}
                  COMMENT "Executing test ${unit_tests_executable}"
                  VERBATIM)
                  
		  
		  

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

