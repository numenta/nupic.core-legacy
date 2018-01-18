zlib-1.2.8.tar.gz from http://www.zlib.net/

Patch zlib sources to remove building of zlib DLL/.so and dependent executables to enable hiding of symbobls
in nupic.bindings extension shared libs except those with explicit visibility (per https://gcc.gnu.org/wiki/Visibility).
zlib sources don't set explicit visibility attributes on its API functions, thus dependent executables fail to link when
we pass C_FLAGS that hide visibility by default. We only need the static lib, anyway.

