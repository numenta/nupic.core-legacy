
## Layout of directories:
```
  Repository
     bindings
        py   -- location of Python bindings
          packaging -- contains things needed to build package
          cpp_src 
            bindings  -- C++ pybind11 mapping source code 
            plugin    -- C++ code to manage python plugin
          tests       -- .py unit test for python interface
```

## This is where we build the distribution package:
```
  Repository
     build
        Release
     bin         --- contains unit test executables
     distr       --- Python distribution (packaging directory copied here)
                           setup.py, requirements.txt will be here
        dist       --- pip egg is generated here
        src
           htm
        bindings  -- python extension libraries installed here
           regions
           tools
           include      --- include files
           lib          --- htm.core.lib  static lib for core found here
        scripts         --- CMake build artifacts are in here.
```
