
## Layout of directories:
```
  Repository
     bindings
        py   -- Location of Python bindings
          packaging -- Contains things needed to build package
          cpp_src 
            bindings  -- C++ pybind11 mapping source code 
            plugin    -- C++ code to manage python plugin
          tests       -- Unit test for python interface
```

## This is where we build the distribution package:
```
  Repository
     build
        scripts       --- CMake build artifacts are in here.
        Release
          bin         --- Contains executables for unit tests and examples
          include     --- Include C++ header files
          lib         --- Static & Dynamic compiled libraries for htm core found here
          distr       --- Python distribution (packaging directory copied here)
            setup.py
            requirements.txt
            dist      --- pip egg is generated here
            htm       --- Pure python code is copied to here
            src
              htm     --- Python library assembled here
                bindings --- C++ extension libraries installed here
                  regions
                  tools
```
