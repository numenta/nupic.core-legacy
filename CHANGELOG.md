# Changelog

Also see [API_CHANGELOG](API_CHANGELOG.md).

## 2.0.0
* changed repository name from nupic.core to htm.core
* Added support for Visual Studio 2019

## 1.1.0

* Comunity release
*  4a9c8a2..0c4a30a
* added CI for Win, OSX, Linux
* testing binary releases

## 1.0.6

* This release was because of a deployment problem with 1.0.5 and issues with pypi.

## 1.0.5

* 58754cf7 NUP-2519: Upgrade pycapnp to 0.6.3
* 64fb803e DEVOPS-383: Move deploy to S3 job to circleci workflow
* 16dbf534 DEVOPS-383: Migrate CircleCI builds from 1.0 to 2.0
* ec14f2f5 pip install --use-wheel" was deprecated. See https://pip.pypa.io/en/stable/news/#deprecations-and-removals
* c2d31a95 ignore YCM configuration
* f7b86e0f "pip install --use-wheel" was deprecated. See https://pip.pypa.io/en/stable/news/#deprecations-and-removals
* 42df6caa NUP-2504: fix clang link optimization issue on private template function
* f5bc76b6 NUP-2504: Add unit test to softmax overflow
* 203493ed NUP-2506: Add missing state fields to serialization
* 5f1ddcbb NUP-2514: fix traversal limit
* 7e33ca44 NUP-2506: Add missing operator '==' to classes used in tests
* e5e48c8e fix accumulate
* b3513853 add softmax function
* 0032fbf5 Fix softmax overflow


## 1.0.4

* 860459cf NUP-2506: Add operator '==' to classes used in tests
* ac43320f NUP-2505: Remove win32 build from CI
* 6a580c06 Fix not include ClassifierResult.hpp error
* de567a4e NUP-2495: Update clang-format instructions
* 2f027a7b NUP-2475: Add sparse link validation
* 3071b8a5 DEVOPS-353: Update "setSparseOutput" documentation
* 7ae0d64f NUP-2495: Check style using clang-format
* 8a1d5eeb NUP-2475: Make sure Network API links use the same dtype at both ends
* 4e800a64 DEVOPS-353: Fix tests to run with nupic.core directly instead of nupic
* 9ae6155d DEVOPS-353: Implement sparse links
* de88baaa DEVOPS-353: Add 'isSparse' attribute to input/output
* 1e486fde DEVOPS-353: Allow element count to be smaller than buffer size
* c9377d52 Fix cmake version and project name
* f069c310 Add missing include required for NTA_ASSERT
* a6c186ae reformat code using clang-format
* 70d43da1 NUP-2492: Add constructor vs initialize test regarding Issue #1386
* dabd7f87 NUP-2491: Validate global inhibition parameters

## 1.0.3

* https://github.com/numenta/nupic.core/issues/1380 Added stronger type checking for SP compute input.

## 1.0.2

* NUP-2481: Update capnp to 0.6.1
* NUP-2469: Mark version as pre-release
* Don't run iwyu if doing gcc build
* RES-571 Explicitly declare cmake 3.3 minimum requirement

## 1.0.1

* NUP-2481: Update capnp to 0.6.1
* Don't run iwyu if doing gcc build
* RES-571 Explicitly declare cmake 3.3 minimum requirement

## 1.0.0

* Convert getProtoType to getSchema to be consistent with Python interface in nupic
* Add Circle badge to README
* Make --user optional since manylinux doens't use it
* Remove build tools setup for Linux CI jobs since new base image already provides these tools
* NUP-2341: Expose capnp serialization to SVM swig interface
* NUP-2341: Add missing 'getSchema'
* update boost to 1.64.0 stable release
* Add a "checkInputs" param to the TemporalMemory

## 0.7.2

* Add SWIG support for Cells4 capnp serialization

## 0.7.1

* SP optimization using minmax_element (#1326)
* Add capnp serialization to Cells4, Cell, Segment, SegmentUpdates, CState, etc. (#1346)
* Fix missing actValue in SDR classifier SWIG interface (#1348)

## 0.7.0

* Give clear exception when clients attempt to reuse ClassifierResult instance with SDRClassifier (PR #1342)
* Remove FastClaClassifier (PR #1341)
* Allow SDR classifier to handle multiple category (PR #1339)
* Add Cap'n Proto serialization to SVM (PR #1338)

## 0.6.3

* Change the Connections to stop doing automatic segment / synapse cleanup
* Revamp the TemporalMemory proto so that it captures all information
* For numpy, use a single PyArray_API symbol per binary rather than one per cpp file
* Use numpy 1.12.1
* Use pycapnp 0.5.12
* Use latest pytest 3.0.7, pytest-cov 2.5.0, pytest-xdist 1.16.0

## 0.6.2

* Updated Circle CI configuration and switch to that for OS X builds (instead of Travis CI)
* Documentation updates: PyRegion
* IWYU documentation update
* Fixed C++ boolean ref counting bug that caused segfaults when running Python projects
* Update pytest and pycapnp dependency versions
* Implemented byte buffer passing as alternative Python<->C++ strategy for capnp serialization logic. This has some memory/speed overhead but avoids ABI issues that could cause crashes
* Fixed prototest to get meaningful comparison numbers

## 0.6.1

* Eliminate installation of unnecessary header files when building nupic.bindings. Install Version.hpp unconditionally for the sake of the deployment usage in .travis.yaml
* Fixed servicing of delayed links such that the circular buffers of all links within a network are refreshed at the end of each time step; and 0-delay links copy data directly from src to dest, bypassing the circular buffer altogether.
* Add a nNonZerosPerRowOnCols SparseMatrix method
* Fix up some out of date SWIG interface code

## 0.6.0

* NUP-2366 Remove no-longer-relevent test code
* NUP-2366 Do not pass arguments in initialize, remove conflicting docstring
* Add a unit test for mapSegmentsToCells
* Change segments to UInt32 to speed up the bindings
* Fix up fetch_remote script to properly poll until the build is complete.
* Updates to tests moved over from nupic
* removed shebangs
* Changed all copyright headers on all files to AGPL.
* Tests folder refactored
* Reads version from VERSION for doc output
* Fixing travis-ci unit tests
* Doxygen only creates XML, used by Breathe to create docs.

## 0.5.3

* Added comment explaining the nuance of the static variables in Cells4::adaptSegment
* Improved explanation of containers in Cells4::adaptSegment implementation.
* Regenerate SWIG bindings C++ files if any headers change.
* Enable -D_LIBCPP_DEBUG=0 for debug-lite in Clang builds (e.g., OS X). NOTE: Clan't debug mode is not mature, and anything above level 0 appears to require debug build of libc++ during linking, which it not present by default on OS X.
* Fixed deletion of wrong synapses and index-out-of-bounds in Cells4::adaptSegment/Segment::freeNSynapses.
* corrected typo in CHANGELOG.md (boostingStrength -> boostStrength)
* Remove ExtendedTemporalMemory proto
* Fix bug in unit test, make TM catch this bug in DEBUG builds
* Perf: Densify prevActiveCells rather than binary searching
* Get POC of OS X build on circle.ci up and running.
* Bump version to prerelease.


## 0.5.1

* Improved network API introspection
* Fix bug with region dimensions inference in network API. fixes #1212
* Updates to info in RELEASE.md
* Fixup import favoring native unittest
* Use xcode6.4 image per documentation at https://docs.travis-ci.com/user/osx-ci-environment/#OS-X-Version
* NUP-2316 Fixed OS X build failure on xcode 8 caused by included some files in static library that shouldn't have been included.
* Misc. changes for sparse matrices and using them in Connections. Moving loops to C++.

## 0.5.0

* Link: Implemented delayed links. NOTE: this is not backwards compatible with CLAModel class, so Nupic will be updated as well.
* SparseMatrix: Stop copying input vectors
* Drop the declare_namespace call - Issue #1072
* SparseMatrix: Add option to specify rows in nNonZerosPerRow
* SparseMatrix: setRandomZerosOnOuter with list of counts
* SparseMatrix: rightVecSumAtNZ cleanup
* SparseMatrix: Put cols in dense array even when not necessary
* Removes experimental code that will now live in nupic.research.core.
* Removed unused, obsolete NetworkFactory class.

## 0.4.16

* SpatialPooler: Stop rounding the boost factors
* SpatialPooler: Rename "maxBoost" to "boostStrength"
* SpatialPooler: Remove minActiveDutyCycles and minPctActiveDutyCycles from spatial pooler

## 0.4.15

* SpatialPooler: Tweak the boost factor rounding to mimic numpy's rounding of float32s

## 0.4.14

* SpatialPooler: New boosting algorithm
* SpatialPooler: Stop putting tie-break info into the overlaps. `getBoostedOverlaps()` return values are now more correct.
* SpatialPooler: Use an EPSILON when comparing permanences to a threshold to avoid floating point differences.
* SpatialPooler: Round the boost factors to two decimal places to avoid floating point differences.

## 0.4.13

* Use VERSION file when generating docs.
* 64-bit gcc on Windows build error
* New: setZerosOnOuter, increaseRowNonZeroCountsOnOuterTo
* Remove a nupic.math test that should stay in nupic
* Use unittest, not unittest2, to match the nupic.core CI config
* Add C++ unit tests for the new SparseMatrix methods
* Finish moving nupic.bindings math tests to nupic.core
* s/AboveAndBelow/BelowAndAbove
* Moves some tests for SWIG bindings from nupic to nupic.core
* Better method names: s/AboveAndBelow/BelowAndAbove
* Four new SparseMatrix methods, enabling batch synapse learning
* Expose the nrows,ncols SparseBinaryMatrix ctor in the bindings

## 0.4.12

* Updated to numpy 1.11.2 everywhere.
* Initialize timer variable in 'Regioncpp' file

## 0.4.11

* Botched release, unavailable.

## 0.4.10

* Removes version as part of iterative artifact name
* Deleted linux ci scripts for travis.
* Removed calls to linux ci scripts, forced platform=darwin
* Remove gcc builds from matrix.
* Using image w/ same version of xcode as last passing master build
* Remove some executables from test list that we don't want to run every build.
* Complete test coverage for region registration.
* Reduce build times by removing some executables from build.
* Update py_region_test for new behavior and make sure the test is run in CI.
* Fixes #1108 by only throwing exception when registering region with class name that is already registered if the new region is from a different module than the original.
* Remove unused vagrant configuration
* Remove default values for vectors because GCC can't handle it
* gcc error: checking whether a UInt is positive
* "depolarizeCells", "reinforceCandidates", "growthCandidates"

## 0.4.9

* Obey wrapAround paramater for columns, not just inputs
* Make sure exceptions are properly exposed when parsing dataType in region spec.
* Network API: allow Bools to be used as creation params
* DEVOPS-157 Remove now-unused RESULT_KEY env var from the build-and-test-nupic-bindings.sh interface.
* Handle new synapsesForSegment behavior in Connections perf test

## 0.4.8

* Add missing argument to `Set-PsDebug -Trace`
* Issue1075 fix py_region_test failure on windows (#1082)
* Perf: Walk neighborhoods via iterators.

## 0.4.7

* Check that TM input is sorted indices in Release builds
* Update inhibition comments and docstrings.
* Use C arrays in TemporalMemory. Allows numpy array reuse.
* Added that `-DNUPIC_BUILD_PYEXT_MODULES=ON` is the default at this time.
* Added information about usage of the NUPIC_BUILD_PYEXT_MODULES cmake property.
* Describe flatIdx in Connections docstring
* Consistent param ordering between implementations.
* Store "numActivePotentialSynapses". No more "SegmentOverlap".

## 0.4.6

* Templates: Stop inferring const unnecessarily
* Build error sometimes in clang -- need copy constructor
* Check that minThreshold <= activationThreshold
* Split compute into activateCells and activateDendrites
* TM and ETM cleanup
* ETM: Grow synapses on active segments, not just matching segments
* Removal of MSVC TP compilation flag
* Also do the learnOnOneCell serializaton check in ::save
* Implement learnOnOneCell in the ETM

## 0.4.5

* Removed no longer used pre-built Swig executables for various flavors of Linux; nupic.core automatically builds Swig from embedded sources on non-Windows platforms.
* DEVOPS-141 Apply an export map to OS X, Linux, and MINGW builds of nupic.bindings shared objects.
* Refactor extension build steps into a function shared by algorithms, math, engine_internal, etc. in preparation for adding export maps.
* Work around issue in cmake 2.8.7: have external project AprUtil1StaticLib depend directly on external project Apr1StaticLib instead of its library wrapper ${APR1_STATIC_LIB_TARGET}; the latter was incorrectly interperting the dependency as another external project instead of library; but worked correctly on cmake 2.8.12.
* Completed wrapping of external static libs in `add_library` targets
* Represent external build of capnproto as single static library with target name ${CAPNP_STATIC_LIB_TARGET} and containing all capnproto library objects.
* No need for custom target in CREATE_CAPNPC_COMMAND function, since nupic_core_solo is the only consumer of the custom command's outputs.
* Try building nupic_core_solo intermediate static library without specifying system libraries. It's a static library and shouldn't need additional linking information.
* Removed nupic_core_solo from installed targets, since it's only an intermediate artifact and not intended to be an output product.
* issue-1034 Reorganized build to link nupic_core static library tests against the "combined" nupic_core static library, which is considered an output artifact, instead of nupic_core_solo static lib, which is only an intermediate step.
* Use library utils to correctly combine multiple static libraries into a single one.
* DEVOPS-135 Implement a hacky work-around by preloading pycapnp's exteions DLL in RTLD_GLOBAL mode, which enables resultion of capnproto references when nupic.bidnings extensions are loaded.
* Consider EPSILON while finding minPermanenceSynapse
* Refactor to make GCC stop finding false positives
* New implementation of GCC_UNINITIALIZED_VAR
* DEVOPS-135 Remove capnproto from python extensions shared libs in order to avoid conflict with capnproto methods compiled into pycapnp, possibly compiled with a different compiler/version and with different compiler/linker flags. Instead, we rely on dynamic linking of the necessary symbols from capnproto in pycapnp.
* Avoid clang error "unknown warning option '-Wno-maybe-uninitialized'"
* Store EPSILON as a static const Permanence
* Disable GCC's maybe-uninitialized warnings. Too many false positives.
* Fixes nupic.core github issue #1031: rename EXTERNAL_STATICLIB_CONFIGURE_DEFINITIONS_OPTIMIZED_STR to EXTERNAL_STATICLIB_CONFIGURE_DEFINITIONS_OPTIMIZED and turn it into a list.
* fixes floating point comparison bug
* Use group_by in TemporalMemory. No more ExcitedColumns.
* Work around another GCC maybe-uninitialized faulty error
* Build error when including <GroupBy.hpp> without <algorithm>
* a fresh look at ExcitedColumns: it's just a group_by
* Fixed indexing bug in serialization
* Fix the unsigned int underflow bug in the ETM
* Workaround for a mingwpy issue
* DEVOPS_116 Add stock Numenta file headers and improve some comments in manylinux wheel build scripts.
* Fixed version bug in SDR classifier deserialization
* Updated bindings to handle SDR classifier
* Fix bug in TemporalMemory save / load
* Make getLeastUsedCell non-allocating
* Put ExcitedColumns in a namespace to avoid collisions
* Add apical dendrites to ExtendedTemporalMemory
* DEVOPS-116 Implement prototype manylinux wheel build using docker image numenta/manylinux1_x86_64:centos6.8
* Stop implying pycapnp is in requirements.txt
* Updated SDRClassifier to match header file on getters/setters
* Fixed memory error (off-by-one in weight matrix creation)
* Fixed include bug in SDRClassifier
* Stable implementation of save/load for SDRClassifier
* Ignore installed version of pip.
* Make sure pip gets installed and that we get console output for everything.
* DEVOPS-84 Add support for building nupic.bindings for OS X in bamboo
* change heap allocation to stack allocation of Network
* Merged upstream changes and backed out the disabling of link time optimizations since it now appears to work on the ARM.
* Add a "formInternalConnections" parameter
* Add new "experimental" SWIG module
* Move ETM to "experimental" namespace
* Support "external active cells" in the ETM
* Use std::move to keep the previous active cells
* Untangle the pieces that calculate segment excitation
* Get rid of the Cell struct
* Initial version of ExtendedTemporalMemory: copy-pasted TM
* added blurps for CMAKE_AR and CMAKE_RANLIB
* CMakeLists.txt cleaned up, vars renamed, docs updated
* Updates pycapnp to 0.5.8.
* network-factory adds missing newline to test file
* network-factory updates test to ensure created network is unregistered.
* deprecate-spec-vars moves documentation from Spec to PyRegion
* network-factory adds a test to show that creating a network without any regions or links works
* network-factory expose functions to bindings and extend functionality to accept a yaml string
* network-factory cleans up includes and makes createNetworkFromYaml public
* updated conditional cmake_args, fixes #981
* network-factory passes yaml parser instead of path
* network-factory ensure Network destructor is called on error
* deprecate-spec-vars set defaults for regionLevel and requireSplitterMap and make them optional
* Removed the duplicate conditional addition of the NTA_ASM compiler flag from src/CMakeLists.txt.  The original remains in CommonCompilerConfig.
* Removed check for Wreturn-type compatibility
* CMake warning if total memory is less than 1GB
* Added back the -O2 flag and the -Wno-return-type flag
* GCC Bitness handling
* Remove NTA_ASM flag
* Removed the inversion of the NTA_ASM logic made by @codedhard
* 1. Added CMake flag to make arm specific changes to the build 2. Used new CMake flag to disable inline assembly in build 3. Added NTA_ASM conditional to parts of ArrayAlgo.hpp that didn't use it    to enable inline assembly
* Add visualization event handlers to Connections class

## 0.4.4

* Timer: test drift
* add CSV parser library

## 0.4.3

* Adds include-what-you-use option in CMake configuration.
* Use the supported mingwpy toolchain from pypi.anaconda.org/carlkl/simple
* Define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS for nupic.core source build in attempt to make build work with official mingwpy toolchain
* Protect all public member variables (except connections)
* Move algorithm details out of the compute method
* Split bamboo linux build into debug and release.
* Move common environment setup for Bamboo into separate script.
* Update SWIG to use local tarfile rather than download from sourceforge.
* Store the new lastUsedIteration, don't just increment old one
* Unit test: destroyed synapse + segment reuse
* Bugfix: Keep the destroyed synapses count in sync.
* Walk the columns and segments via ExcitedColumns iterator
* Fail loudly when Python regions reassign an output
* Disable asserts in release builds
* De-duplicate a loop.
* Removes 'long double', NTA_QUAD_PRECISION, and Real128 references from our code.
* Add Proto for SDR Classifier
* TemporalMemory: replace phases with "columnSegmentWalk". Much faster.
* MovingAverage.compute() returns updated value

## 0.4.2

* Define and use environment variables containing compiler and linker flags specifically for 3rd-party source code.
* Revert to installing mingwpy toolchains from https://bitbucket.org/carlkl/mingw-w64-for-python/downloads.
* Better rand() approach, won't ever be greater than 1
* Stop computing overlaps twice per timestep. 4% faster.
* Stop storing segment overlap counts in a map. Now 56% faster.
* Enable errors for windows32 build in the matrix.
* Use proper float comparisons to avoid issues on 32 vs 64 bit machines.
* Perf: When using sets, use std::set::find, not std::find
* Swig needs `-include cmath` when building with mingw toolchain.
* Initialize swig_generated_file_compile_flags directly from COMMON_CXX_FLAGS_OPTIMIZED to avoid -Werror.
* Comment out header and binary install targets in CapnProto.cmake, since this is a duplicate of the exact same steps in srce/CMakeLists.txt, which installs other externals as well.
* Attempt to fix execution of helloregion test from Travis build by modifying .travis.yml to change directory to build/release/bin just as instructed by nupic.core/README.md.
* Added a comment about nupic.core's swig wraps relying on the macro CAPNP_LITE to have a value.
* Fix windows build of nupic.core's Swig wraps that expects CAPNP_LITE not only to be defined (capnp recommends just defining it), but to actually have a value (non-zero).
* Invoke additional nupic.core tests in Travis to make sure that things get tested and tests don't get broken again unnoticed:   connections_performance_test   helloregion   hello_sp_tp   prototest
* Refactored CommonCompilerConfig.cmake to exposed optimized and unoptimized flag sets. Fixed linux GCC build of CapnProto components by changing its configuration to use unoptimized versions of common flags. Cleaned up use of local variables in src/CMakeFiles.txt. nupic.core/CmakeLists.txt is now the only place that defines REPOSITORY_DIR for use by other modules. Fixed dependency in add_custom_command function inside CREATE_CAPNPC_COMMAND to use passed-in arg SPEC_FILES instead of a property from src/CMakeLists.txt.
* Add BITNESS to Swig.cmake

## 0.4.1

* Cast arg to UInt32 to avoid call resolution ambiguity on the value.write() method on Win32 platform.
* Finish adding support for the Bool type
* Expose encoder's 'n' so other regions' inputWidth is calculable from the outside
* Remove reference to deleted Linear.cpp.
* Run nupic.core encoders via boilerplate Sensor wrapper
* Removes Linear.hpp, Linear.cpp, and reference in algorithms.i since it doesn't appear to be used anywhere.
* Support Debug builds in Clang, put them in README
* Add a NTA_BasicType_Bool so that we can parse bools in YAML
* FloatEncoder base, no more common Encoder, new signature
* Prepends BINDINGS_VERSION plus dot separater to wheel filename
* Fix integer division bug, add tests that would catch this bug
* Don't divide by a constant. Multiply. Dodges divide-by-0.
* Explicitly mark all derived virtual methods as virtual.
* Fix gcc build issue
* C++ Encoder base + ScalarEncoder + ported unit tests

## 0.4.0

* Reduce EPSILON threshold for TM to minitage compatibility issues.
* Updates AV yaml to push a commit-sha'd wheel to AWS
* Reduce permanence threshold by epsilon before comparing to avoid rounding edge cases
* Make threshold for destroying synapses larger to catch roundoff errors
* Make comparison identical to python
* Add accessors for TM columnForCell and cellsForColumn
* Temporal Memory: recordIteration should be false in second call to computeActivity
* Fix: Incompatibility in Connections.computeActivity with python
* Change TM init parameters to accept segment and synapse limit.

## 0.3.1

* Secondary sort on segment idx
* Sort segments before iterating for python compatibility
* Sort unpredictedActiveColumns before iterating for python compatibility

## 0.3.0

* Updated SWIG bindings for accessors
* Added TemporalMemory accessors
* Update bindings for C++ Connections to expose 'read' class method
* Destroy lowest permanence synapse when synapse limit is reached
* Fix for bug in Segment::freeNSynapses
* Added initialization code from Tester::init into PathTest::PathTest that is required for PathTest::copyInTemp to run successfully.
* Remove references to Tester from UnitTestMain.cpp
* Deleted Tester class and TesterTest files
* Update SWIG binding of Network to add read class method
* Refactor PyRegion subclasses to take specific proto object in read
* Update SWIG binding of TemporalPooler to add read class method
* Enable basic building with ninja (cmake -GNinja)
* Added include of Log.hpp in NodeSet.hpp
* Update SWIG bindings of SpatialPooler and CLAClassifier to add read class methods to C++ classes

## 0.2.7

* Full filename for twine.
* Absolute path for twine's binary file upload.

## 0.2.6

* Fixed incorrect binary path for twine.

## 0.2.5

* Fixing twine upload of windows binary.

## 0.2.4

* Working out issues with the release process.

## 0.2.3

* Windows Support!
* Add bindings for CLAClassifier serialization
* Adding Windows twine pip install and PyPi upload
* Storing release version in VERSION in proj root.
* Makes build-from-source the default behavior for capnp and swig, requiring the flags FIND_CAPNP or FIND_SWIG to be specified in order to use a preinstalled version.
* Add Dockerfile for building nupic.core from source
* Update of Windows bindings setup
* Add "python setup.py test" option (fixes #697)
* Adding a CMake ExternalProject to download capnp win32 compiler tools
* Simplify setup.py, remove unused command line args, add setup.py clean command. Update Travis to build Python extensions as part of cmake so Python install doesn't rebuild them.
* Allow finding pre-built capnp.
* Revert back to numpy==1.9.2

## 0.2.2

* First changelog entry.
