# External Dependencies

## Cap'n Proto v0.5.0

We install everything the user will need, including the capnp command-line
tool. As such, the `--disable-shared` option to `./configure` must be used
to ensure that the binaries are statically compiled. Additionally, the
`-fPIC` compiler flag is required so we include all NuPIC flags. The
commands thus look similar to this:

```
wget https://capnproto.org/capnproto-c++-0.5.0.tar.gz
tar zxf capnproto-c++-0.5.0.tar.gz
cd capnproto-c++-0.5.0
CXXFLAGS="-fPIC -std=c++11 -m64 -fvisibility=hidden -Wall -Wreturn-type -Wunused -Wno-unused-parameter" ./configure --prefix=$HOME/nta/nupic.core/external/common --disable-shared
make -j6 check
make install

# Move bin/lib files into platform-specific locations (linux64 in this case).
mv external/common/bin/* external/linux64/bin/
mv external/common/lib/*.a external/linux64/lib/
# Remove extra lib files that we don't need.
rm -r external/common/lib
```
