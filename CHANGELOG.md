# Changelog

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
