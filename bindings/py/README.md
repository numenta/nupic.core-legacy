
## Layout of directories:
```
  REPO_DIR    external  -- cmake scripts to download/build dependancies
    bindings
        py   -- Location of Python bindings
          packaging -- Contains things needed to build package
          cpp_src 
            bindings  -- C++ pybind11 mapping source code 
            plugin    -- C++ code to manage python plugin
        tests       -- Unit test for python interface    py        htm           -- python source code goes here        tests    src        htm           -- C++ source code goes here        examples        tests    build             -- where all things modified by the build are placed (disposible)        ThirdParty    -- where all thirdParty packages are installed.        scripts       -- temporary build files        Release       -- Where things are installed
```

## This is where we build the distribution package:
```
  REPO_DIR
     build
        Release
            bin         --- Contains executables for unit tests and examples
            include     --- Include C++ header files
            lib         --- Static & Dynamic compiled libraries for htm core found here
          - distr                            ( copy from REPO_DIR/bindings/py/packaging/* by CMake)
         /      build                   -- setup.py; setup() puts stuff in here        /       dist                    -- setup.py; setup() puts stuff in here        |       dummy.c                           |       requirements.txt             ( copy from REPO_DIR/requirements.txt by setup.py; setup())        |       src                     --setup.py; setup() will look in here for packages, should find htm, tests        |           htm                      ( copy from REPO_DIR/py/htm/*  by CMake)        |               advanced          |               algorithms         |               bindings             ( copy from REPO_DIR/bindings/py/packaging/bindings/* by CMake)        |                   __init__.py      ( copy from REPO_DIR/bindings/py/packaging, by CMake)        |                   *.pyd            (the extension libraries from CMake build)        |                   check.py   Package used             regions     to build                tools      Wheel              encoders        |               examples             ( copy from REPO_DIR/bindings/py/packaging/examples/* by CMake)         \              optimization         ( copy form REPO_DIR/py/htm/optimization by CMake)            \             README.md            ( copy form REPO_DIR/py/htm by CMake)             -            utils_test.py               ```
