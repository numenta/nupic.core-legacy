NuPIC Core external libraries
=============================

NuPIC Core depends on a number of pre-built external libraries which are
normally distributed with the source.  However, since Solaris is not an
officially supported platform, you will need to build the libraries yourself.
Use the following commands as a guide.

**BEFORE YOU BEGIN:** Obtain the source for apr, apr-util, yaml-cpp, yaml, and
zlib and extract in $NUPIC_CORE/external/sparc64

```
cd $NUPIC_CORE/external/sparc64/apr-1.5.1
gmake clean
CC="$CC -m64" ./configure --prefix=$NUPIC_CORE/external/sparc64
VERBOSE=1 gmake
gmake install
```

```
cd $NUPIC_CORE/external/sparc64/apr-util-1.5.3
CC="$CC -m64" ./configure --prefix=$NUPIC_CORE/external/sparc64 --with-apr=$NUPIC_CORE/external/sparc64
VERBOSE=1 gmake
gmake install
```

```
cd $NUPIC_CORE/external/sparc64/yaml-cpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$NUPIC_CORE/external/sparc64 -DBUILD_SHARED_LIBS=OFF ..
VERBOSE=1 gmake
gmake install
```

```
cd $NUPIC_CORE/external/sparc64/yaml-0.1.5
gmake clean
CC="$CC -m64" ./configure --prefix=$NUPIC_CORE/external/sparc64
VERBOSE=1 gmake
gmake install
```

```
cd $NUPIC_CORE/external/sparc64/zlib-1.2.8
gmake clean
CC="$CC -m64" ./configure --prefix=$NUPIC_CORE/external/sparc64
VERBOSE=1 gmake
gmake install
```

And, of course to build NuPIC Core itself:

```
mkdir -p $NUPIC_CORE/build/scripts
rm -rf $NUPIC_CORE/build/scripts/*
cd $NUPIC_CORE/build/scripts
cmake -DUSER_CXX_COMPILER=$CXX $NUPIC_CORE/src
VERBOSE=1 gmake
```
