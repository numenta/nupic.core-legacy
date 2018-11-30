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

cmake_minimum_required(VERSION 3.3)
project(nupic_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug

include(src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES

#
# NuPIC CMake options
#
option(NUPIC_IWYU "Enable include-what-you-use
  (http://include-what-you-use.org/). This requires the iwyu binary to be
  discoverable by CMake's find_program, with a minimum CMake version of 3.3.")
if(${NUPIC_IWYU})
  find_program(iwyu_path NAMES include-what-you-use iwyu)
  if(NOT iwyu_path)
    message(FATAL_ERROR "Could not find the program include-what-you-use")
  endif()
endif()




#
# Extract current git sha and record in nupic/Version.hpp
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE NUPIC_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/nupic/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")
  
include_directories(SYSTEM
                    ${PROJECT_BINARY_DIR}
					${PROJECT_SOURCE_DIR}
					${REPOSITORY_DIR}/external/common/include
					${yaml-cpp_INCLUDE_DIRS}
					${Boost_INCLUDE_DIRS}
					)


#
# Setup nupic_core static library, consisting of our own sources and yaml libraries;
#
# this is an intermediate archive that will be merged with external static
# libraries in a subsequent step to produce the nupic_core "combined" static
# library.
#
set(src_lib_static_nupiccore_solo nupiccore_solo)


set(algorithm_files
    nupic/algorithms/Anomaly.cpp
    nupic/algorithms/Anomaly.hpp
    nupic/algorithms/AnomalyLikelihood.cpp
    nupic/algorithms/AnomalyLikelihood.hpp
    nupic/algorithms/ArrayBuffer.hpp
    nupic/algorithms/BacktrackingTMCpp.cpp
    nupic/algorithms/BacktrackingTMCpp.hpp
    nupic/algorithms/BitHistory.cpp
    nupic/algorithms/BitHistory.hpp
    nupic/algorithms/Cell.cpp
    nupic/algorithms/Cell.hpp
    nupic/algorithms/Cells4.cpp
    nupic/algorithms/Cells4.hpp
    nupic/algorithms/ClassifierResult.cpp
    nupic/algorithms/ClassifierResult.hpp
    nupic/algorithms/CondProbTable.cpp
    nupic/algorithms/CondProbTable.hpp
    nupic/algorithms/Connections.cpp
    nupic/algorithms/Connections.hpp
    nupic/algorithms/InSynapse.cpp
    nupic/algorithms/InSynapse.hpp
    nupic/algorithms/OutSynapse.cpp
    nupic/algorithms/OutSynapse.hpp
    nupic/algorithms/Scanning.hpp
    nupic/algorithms/SDRClassifier.cpp
    nupic/algorithms/SDRClassifier.hpp
    nupic/algorithms/Segment.cpp
    nupic/algorithms/Segment.hpp
    nupic/algorithms/SegmentUpdate.cpp
    nupic/algorithms/SegmentUpdate.hpp
    nupic/algorithms/SpatialPooler.cpp
    nupic/algorithms/SpatialPooler.hpp
    nupic/algorithms/TemporalMemory.cpp
    nupic/algorithms/TemporalMemory.hpp
)


set(encoders_files 
    nupic/encoders/ScalarEncoder.cpp
    nupic/encoders/ScalarEncoder.hpp
    nupic/encoders/ScalarSensor.cpp
    nupic/encoders/ScalarSensor.hpp
)
    
set(engine_files
	nupic/engine/Collections.cpp
    nupic/engine/Input.cpp
    nupic/engine/Input.hpp
    nupic/engine/Link.cpp
    nupic/engine/Link.hpp
    nupic/engine/LinkPolicy.hpp
    nupic/engine/LinkPolicyFactory.cpp
    nupic/engine/LinkPolicyFactory.hpp
    nupic/engine/Network.cpp
    nupic/engine/Network.hpp
    nupic/engine/NuPIC.cpp
    nupic/engine/NuPIC.hpp
    nupic/engine/Output.cpp
    nupic/engine/Output.hpp
    nupic/engine/Region.cpp
    nupic/engine/Region.hpp
    nupic/engine/RegionImpl.cpp
    nupic/engine/RegionImpl.hpp
    nupic/engine/RegionImplFactory.cpp
    nupic/engine/RegionImplFactory.hpp
    nupic/engine/RegionIo.cpp
    nupic/engine/RegionParameters.cpp
    nupic/engine/RegisteredRegionImpl.hpp
    nupic/engine/RegisteredRegionImplCpp.hpp
    nupic/engine/Spec.cpp
    nupic/engine/Spec.hpp
    nupic/engine/TestFanIn2LinkPolicy.cpp
    nupic/engine/TestFanIn2LinkPolicy.hpp
    nupic/engine/YAMLUtils.cpp
    nupic/engine/YAMLUtils.hpp
)
  
set(math_files
    nupic/math/ArrayAlgo.hpp
    nupic/math/DenseMatrix.hpp
    nupic/math/Domain.hpp
    nupic/math/Functions.hpp
    nupic/math/Index.hpp
    nupic/math/Math.hpp
    nupic/math/NearestNeighbor.hpp
    nupic/math/SegmentMatrixAdapter.hpp
    nupic/math/Set.hpp
    nupic/math/SparseBinaryMatrix.hpp
    nupic/math/SparseMatrix.hpp
    nupic/math/SparseMatrix01.hpp
    nupic/math/SparseMatrixAlgorithms.cpp
    nupic/math/SparseMatrixAlgorithms.hpp
    nupic/math/SparseMatrixConnections.cpp
    nupic/math/SparseMatrixConnections.hpp
    nupic/math/SparseRLEMatrix.hpp
    nupic/math/SparseTensor.hpp
    nupic/math/StlIo.cpp
    nupic/math/StlIo.hpp
    nupic/math/Topology.cpp
    nupic/math/Topology.hpp
    nupic/math/Types.hpp
    nupic/math/Utils.hpp
)


set(ntypes_files
    nupic/ntypes/Array.hpp
    nupic/ntypes/ArrayBase.cpp
    nupic/ntypes/ArrayBase.hpp
    nupic/ntypes/ArrayRef.hpp
    nupic/ntypes/Buffer.cpp
    nupic/ntypes/Buffer.hpp
    nupic/ntypes/BundleIO.hpp
    nupic/ntypes/Collection.cpp
    nupic/ntypes/Collection.hpp
    nupic/ntypes/Dimensions.cpp
    nupic/ntypes/Dimensions.hpp
    nupic/ntypes/MemParser.cpp
    nupic/ntypes/MemParser.hpp
    nupic/ntypes/MemStream.hpp
    nupic/ntypes/NodeSet.hpp
    nupic/ntypes/ObjectModel.h
    nupic/ntypes/ObjectModel.hpp
    nupic/ntypes/Scalar.cpp
    nupic/ntypes/Scalar.hpp
    nupic/ntypes/Value.cpp
    nupic/ntypes/Value.hpp
)
  
set(os_files
    nupic/os/Directory.cpp
    nupic/os/Directory.hpp
    nupic/os/Env.cpp
    nupic/os/Env.hpp
    nupic/os/ImportFilesystem.hpp
    nupic/os/OS.cpp
    nupic/os/OS.hpp
    nupic/os/OSUnix.cpp
    nupic/os/OSWin.cpp
    nupic/os/Path.cpp
    nupic/os/Path.hpp
    nupic/os/Regex.cpp
    nupic/os/Regex.hpp
    nupic/os/Timer.cpp
    nupic/os/Timer.hpp    
)

set(regions_files
    nupic/regions/TestNode.cpp
    nupic/regions/TestNode.hpp
    nupic/regions/VectorFile.cpp
    nupic/regions/VectorFile.hpp
    nupic/regions/VectorFileEffector.cpp
    nupic/regions/VectorFileEffector.hpp
    nupic/regions/VectorFileSensor.cpp
    nupic/regions/VectorFileSensor.hpp   
)

set(types_files
    nupic/types/BasicType.cpp
    nupic/types/BasicType.hpp
    nupic/types/Exception.hpp
    nupic/types/Fraction.cpp
    nupic/types/Fraction.hpp
    nupic/types/Serializable.hpp
    nupic/types/Types.h
    nupic/types/Types.hpp
    nupic/types/Serializable.hpp
)

set(utils_files
    nupic/utils/GroupBy.hpp
    nupic/utils/Log.hpp
    nupic/utils/LoggingException.cpp
    nupic/utils/LoggingException.hpp
    nupic/utils/LogItem.cpp
    nupic/utils/LogItem.hpp
    nupic/utils/MovingAverage.cpp
    nupic/utils/MovingAverage.hpp
    nupic/utils/Random.cpp
    nupic/utils/Random.hpp
    nupic/utils/SlidingWindow.hpp
<<<<<<< Updated upstream
    nupic/utils/Watcher.cpp)

set(src_lib_static_nupiccore_srcs
    ${src_py_support_files}
    ${src_nupiccore_srcs})

set(src_lib_static_nupiccore_compile_flags
    "${src_compile_flags} -I${src_numpy_core}/include")
=======
    nupic/utils/StringUtils.cpp
    nupic/utils/StringUtils.hpp
    nupic/utils/TRandom.cpp
    nupic/utils/TRandom.hpp
    nupic/utils/VectorHelpers.hpp
    nupic/utils/Watcher.cpp
    nupic/utils/Watcher.hpp
)
>>>>>>> Stashed changes

#set up file tabs in Visual Studio
source_group("algorithms" FILES ${algorithm_files})
source_group("encoders" FILES ${encoders_files})
source_group("engine" FILES ${engine_files})
source_group("math" FILES ${math_files})
source_group("ntypes" FILES ${ntypes_files})
source_group("os" FILES ${os_files})
source_group("regions" FILES ${regions_files})
source_group("types" FILES ${types_files})
source_group("utils" FILES ${utils_files})

########################################################
# Lib
add_library(${src_lib_static_nupiccore_solo} STATIC 
    ${algorithm_files} 
    ${encoders_files} 
    ${engine_files} 
    ${math_files} 
    ${ntypes_files} 
    ${os_files} 
    ${regions_files} 
    ${types_files} 
    ${utils_files} 
)

# nupic_core_solo sources depend on headers installed by these external projects
add_dependencies(${src_lib_static_nupiccore_solo}
        yaml-cpp
		)
message(STATUS "INTERNAL_CXX_FLAGS=${INTERNAL_CXX_FLAGS}")
set_target_properties(${src_lib_static_nupiccore_solo} 
		PROPERTIES COMPILE_FLAGS "${INTERNAL_CXX_FLAGS}"
		)
target_compile_definitions(${src_lib_static_nupiccore_solo} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
if(${NUPIC_IWYU})
  set_target_properties(${src_lib_static_nupiccore_solo}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()


################################################################get_target_property(YAML_CPP_LIB yaml-cpp property?)
#
# Create the nupic_core "combined" static library by merging nupic_core_solo
# with our external static libraries
#
set(core_library   nupic_core)
set(${src_combined_nupiccore_source_archives}
    ${src_lib_static_nupiccore_solo}
    yaml-cpp
    ${Boost_LIBRARIES}
<<<<<<< Updated upstream
    ${src_common_os_libs})

message(STATUS "src_common_test_exe_libs = ${src_common_test_exe_libs}")


## Example custom executable 
#
## Setup test_py_region
#
#set(src_executable_pyregiontest py_region_test)
#add_executable(${src_executable_pyregiontest} test/integration/PyRegionTest.cpp)
#target_link_libraries(${src_executable_pyregiontest} ${src_common_test_exe_libs})
#set_target_properties(${src_executable_pyregiontest}
#                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
#set_target_properties(${src_executable_pyregiontest}
#                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
#add_custom_target(tests_py_region
#                  COMMAND ${src_executable_pyregiontest}
#                  DEPENDS ${src_executable_pyregiontest}
#                  COMMENT "Executing test ${src_executable_pyregiontest}"
#                  VERBATIM)

#
# Setup gtests
#
set(src_executable_gtests unit_tests)
add_executable(${src_executable_gtests}
               test/unit/algorithms/AnomalyTest.cpp
	       test/unit/algorithms/BacktrackingTMCppTest.cpp
               test/unit/algorithms/Cells4Test.cpp
               test/unit/algorithms/CondProbTableTest.cpp
               test/unit/algorithms/ConnectionsTest.cpp
	       test/unit/algorithms/ConnectionsPerformanceTest.cpp
	       test/unit/algorithms/HelloSPTPTest.cpp
               test/unit/algorithms/NearestNeighborUnitTest.cpp
               test/unit/algorithms/SDRClassifierTest.cpp
               test/unit/algorithms/SegmentTest.cpp
	       test/unit/algorithms/Serialization.cpp
               test/unit/algorithms/SpatialPoolerTest.cpp
               test/unit/algorithms/TemporalMemoryTest.cpp
               test/unit/encoders/ScalarEncoderTest.cpp
	       test/unit/engine/CppRegionTest.cpp
               test/unit/engine/InputTest.cpp
	       test/unit/engine/HelloRegionTest.cpp
               test/unit/engine/LinkTest.cpp
               test/unit/engine/NetworkTest.cpp
	       test/unit/engine/PyRegionTest.cpp
               test/unit/engine/UniformLinkPolicyTest.cpp
               test/unit/engine/YAMLUtilsTest.cpp
               test/unit/math/DenseTensorUnitTest.cpp
               test/unit/math/DomainUnitTest.cpp
               test/unit/math/IndexUnitTest.cpp
               test/unit/math/MathsTest.cpp
               test/unit/math/SegmentMatrixAdapterTest.cpp
               test/unit/math/SparseBinaryMatrixTest.cpp
               test/unit/math/SparseMatrix01UnitTest.cpp
               test/unit/math/SparseMatrixTest.cpp
               test/unit/math/SparseMatrixUnitTest.cpp
               test/unit/math/SparseTensorUnitTest.cpp
               test/unit/math/TopologyTest.cpp
               test/unit/ntypes/ArrayTest.cpp
               test/unit/ntypes/BufferTest.cpp
               test/unit/ntypes/CollectionTest.cpp
               test/unit/ntypes/DimensionsTest.cpp
               test/unit/ntypes/MemParserTest.cpp
               test/unit/ntypes/MemStreamTest.cpp
               test/unit/ntypes/NodeSetTest.cpp
               test/unit/ntypes/ScalarTest.cpp
               test/unit/ntypes/ValueTest.cpp
               test/unit/ntypes/SdrTest.cpp
               test/unit/os/DirectoryTest.cpp
               test/unit/os/EnvTest.cpp
               test/unit/os/OSTest.cpp
               test/unit/os/PathTest.cpp
               test/unit/os/RegexTest.cpp
               test/unit/os/TimerTest.cpp
               test/unit/py_support/PyHelpersTest.cpp
               test/unit/types/BasicTypeTest.cpp
               test/unit/types/ExceptionTest.cpp
               test/unit/types/FractionTest.cpp
               test/unit/UnitTestMain.cpp
               test/unit/utils/GroupByTest.cpp
               test/unit/utils/MovingAverageTest.cpp
               test/unit/utils/RandomTest.cpp
               test/unit/utils/SlidingWindowTest.cpp
               test/unit/utils/WatcherTest.cpp
               test/unit/utils/VectorHelpersTest.cpp
	       )
target_link_libraries(${src_executable_gtests}
                      ${src_lib_static_gtest}
                      ${src_common_test_exe_libs})
set_target_properties(${src_executable_gtests}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags}
                                 LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_unit
                  COMMAND ${src_executable_gtests}
                  DEPENDS ${src_executable_gtests}
                  COMMENT "Executing test ${src_executable_gtests}"
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
                  DEPENDS tests_unit
                  COMMENT "Running all tests"
                  VERBATIM)


#
# Use SWIG to generate Python extensions.
#
if (NUPIC_BUILD_PYEXT_MODULES)
  include(UseSWIG)

  # Set the output location for the language modules that are created.
  set(CMAKE_SWIG_OUTDIR ${PROJECT_BINARY_DIR})

  # Make sure the directory exists for the generated C++ files.
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/nupic/bindings)

  # TODO ZZZ set COMPILE_FLAGS on swig targets

  # SWIG options from:
  # https://github.com/swig/swig/blob/master/Source/Modules/python.cxx#L111
  set(src_swig_flags
      -c++
      -features
      autodoc=0,directors=0
      -noproxyimport
      -keyword
      -modern
      -modernargs
      -noproxydel
      -fvirtual
      -fastunpack
      -nofastproxy
      -fastquery
      -outputtuple
      -castmode
      -nosafecstrings
      -w402
      -w503
      -w511
      -w302
      -w362
      -w312
      -w389
      -DSWIG_PYTHON_LEGACY_BOOL
      -I${SWIG_DIR}/python
      -I${SWIG_DIR}
      ${src_compiler_definitions}
  )

  file(GLOB_RECURSE swig_header_deps *.i *.hpp *.h)

  message(STATUS "src_swig_flags = ${src_swig_flags}")

  # Tell swig which command-line options to use, allowing user to override
  set(CMAKE_SWIG_FLAGS ${src_swig_flags} ${CMAKE_SWIG_FLAGS})

  # Set up linker flags for python extension shared libraries
  set(_SRC_SWIG_EXTENSION_LINK_FLAGS "${PYEXT_LINKER_FLAGS_OPTIMIZED}")

  # NOTE Non-Windows Python extensions shouldn't be linking against libpython;
  # symbols should be available automatically when python loads the extension.
  #
  set(_SRC_SWIG_LINK_LIBRARIES
      ${src_lib_static_nupiccore_combined}
      ${src_common_os_libs})

  # Common dependencies for our python extensions for use with
  # SWIG_MODULE_name_EXTRA_DEPS
  # Make sure we don't execute the swig executable before it is built
  set(_SRC_SWIG_EXTRA_DEPS Swig)

  # NOTE Windows DLLs are shared executables with their own main; they require
  # all symbols to resolve at link time, so we have to add libpython for this
  # platform
  #
  if("${PLATFORM}" STREQUAL "windows")
    list(APPEND _SRC_SWIG_LINK_LIBRARIES
         ${PYTHON_LIBRARIES})
  endif()

  message(STATUS "_SRC_SWIG_EXTRA_DEPS     = ${_SRC_SWIG_EXTRA_DEPS}")
  message(STATUS "_SRC_SWIG_LINK_LIBRARIES = ${_SRC_SWIG_LINK_LIBRARIES}")
  message(STATUS "_SRC_SWIG_EXTENSION_LINK_FLAGS= ${_SRC_SWIG_EXTENSION_LINK_FLAGS}")
  message(STATUS "CMAKE_SWIG_FLAGS        = ${CMAKE_SWIG_FLAGS}")


  function(PREPEND_BOILERPLATE_TO_PYTHON_PROXY_MODULE
           MODULE_NAME)
    # Add a custom command to the Swig target to prepend boilerplate to the
    # swig-generated python proxy module
    # ${CMAKE_SWIG_OUTDIR}/${MODULE_NAME}.py.
    #
    # :param MODULE_NAME: the custom command will be added to the target
    #   corresponding to this Swig module name.
    set(preamble_filepath "${CMAKE_SOURCE_DIR}/src/nupic/bindings/swig_proxy_preamble.py")
    set(module_filepath "${CMAKE_SWIG_OUTDIR}/${MODULE_NAME}.py")

    add_custom_command(
      TARGET ${SWIG_MODULE_${MODULE_NAME}_REAL_NAME}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND}
            -DSRC_FILE_1=${preamble_filepath}
            -DSRC_FILE_2=${module_filepath}
            -DTARGET_FILE=${module_filepath}
            -P ${CMAKE_SOURCE_DIR}/src/ConcatTwoFiles.cmake
      COMMENT "Prepending ${preamble_filepath} to swig-generated module ${module_filepath}"
=======
>>>>>>> Stashed changes
    )
merge_static_libraries(${core_library} "${src_combined_nupiccore_source_archives}")


#########################################################
# Test
# The tests were placed in a separate file to avoid clutering up this file.
include(NupicCoreTests.cmake)


##################################################
#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${core_library}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Version.hpp is also used by the nupic.bindings release/deployment system
install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/nupic)

# install Data.csv for a test
install(FILES "${PROJECT_SOURCE_DIR}/test/unit/engine/Data.csv"
        DESTINATION "data/")

install(DIRECTORY nupic DESTINATION include
	  MESSAGE_NEVER
	  FILES_MATCHING PATTERN "*.h*"
	  PATTERN "*.hpp.in" EXCLUDE)

install(DIRECTORY ${REPOSITORY_DIR}/external/common/include/gtest
	  MESSAGE_NEVER
	  DESTINATION include/gtest
	  FILES_MATCHING PATTERN "*.h*")

install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
	  MESSAGE_NEVER
	  DESTINATION include)


          
#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
