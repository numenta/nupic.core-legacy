# C++ NuPIC HelloWorld example

This file serves two purposes: 

## Demo C++ aplication 

Using only C++ `nupic.core` implementations, currently the chain uses 
` SpatialPooler > TemporalPooler` 

More classes should be added as they are ported (Encoders, Classifier, Anomaly, ...)

## Profiling code 

For ongoing optimization goals, we need a simple but complete code to run benchmarks and profile. 
You can easily change the constants (`EPOCHS, DIM,...`) and try this code on your branch. 

### Using `valgrind` profiler (for memory, #calls usage)

Steps to profile methods' execution time with `valgrind`'s extension `callgrind`: 

* You will need to install the `valgrind` profiling tool
* Compile your C++ source with debug info (`-g`) and (optionally) optimizations on. Eg.: 
```
CC=gcc-4.8 CXX=g++-4.8 cmake -DCMAKE_BUILD_TYPE=Debug ../../src -DCMAKE_INSTALL_PREFIX=../release -DCMAKE_EXPORT_COMPILE_COMMANDS=O && time make -j4
```
* run desired program with (valgrind/callgrind)[http://valgrind.org/docs/manual/cl-manual.html]: 
```
valgrind --tool=callgrind ./hello_sp_tp 
```
it will generate file `callgrind.out.<pid>` which can be viewed in a graphical tool (eg. `KCacheGrind` for Ubuntu and others) or proccessed on 
command line: `callgrind_annotate callgrind.out.<pid>`
