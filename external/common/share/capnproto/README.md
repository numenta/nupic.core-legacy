capnproto-c++-0.6.1.tar.gz: from https://capnproto.org/capnproto-c++-0.6.1.tar.gz
capnproto-c++-win32-0.6.1.zip: from https://capnproto.org/capnproto-c++-win32-0.6.1.zip

Capnp added a dependency to `kj::ThrowOverflow` function in `capnp/common.h` 
implementing the function in `units.c++`. However `units.c++` is only compiled
when `CAPNP_LITE=0` causing a link error when `CAPNP_LITE=1`. See
https://github.com/capnproto/capnproto/pull/437
We patch capnp 0.6.1 adding `units.c++` to `CAPNP_LITE=1` build.  
