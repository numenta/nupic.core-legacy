# Changelog

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
