# External Dependencies

## Cap'n Proto v0.5.2

**Note**: Compiling on newer versions of linux with glibc >=2.14 can be
problematic. A change to memcpy causes linking to the newer glibc version,
resulting in binaries that cannot be used on older platforms. See details
and a potential workaround here:
http://www.win.tue.nl/~aeb/linux/misc/gcc-semibug.html

Some compiler flags are required and can be specified as below with the
`CXXFLAGS` environment variable.

```
curl -O https://capnproto.org/capnproto-c++-0.5.2.tar.gz
tar zxf capnproto-c++-0.5.2.tar.gz
cd capnproto-c++-0.5.2
CXXFLAGS="-fPIC -std=c++11 -m64" ./configure
make -j3 check
make install
```

Notes:

- Append `--prefix=path/to/install` to `configure` script to install somewhere other than `/usr/local`. You will then need to pass this path to CMake with the `cmake` flag `-DCMAKE_PREFIX_PATH`.
- Append `--disable-shared` to `configure` when building portable egg or wheel binaries.
- `-j3` will run three jobs in parallel. This can be omitted to avoid parallelism or changed to specify a different number of parallel jobs.
