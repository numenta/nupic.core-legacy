## Instructions for creating the bindings

Build and Install
* Install Python 2.7 and/or 3.x
* Download and install Boost. Place `BoostRoot` environment variable in path.
* Downoad/fork the repository http://github.com/htm-community/nupic.cpp
* `cd Repository`
* `python setup.py install --user --prefix=`
* Run C++ unit tests `cd Repository/Release/bin`  then execute `./unit_tests`
* Run Python unit tests `cd Repository` then `python setup.py test`

This will build the C++ htm.core library extension package and install it in the Python
you have running.  

If you run into problems due to caching of arguments in CMake, delete the folder `Repository/build` and try again.


Layout of directories:
`
  Repository
     bindings
        py   -- location of Python bindings
          packaging -- contiains things needed to build package
          cpp_src 
            bindings  -- C++ pybind11 mapping source code 
            plugin    -- C++ code to manage python plugin
          tests    -- .py unit test for python interface
`

This is where we build the distribution package
`
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
`

