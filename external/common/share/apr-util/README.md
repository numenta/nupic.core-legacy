unix/apr-util-1.5.4.tar.gz from https://apr.apache.org/

We use the apr-util unix sources for both unix (linux, Mac OS X) and windows builds.
apr-util provides CMakeLists.txt for Windows builds; it uses configure-based builds for unix (and friends).

nupic.core's external "package" wraps the build of this library in a cmake module.

We patch the apr-util sources as follows:

1. Comment out the shared library libaprutil-1 cmake target (used on Windows builds) to enable the
   static version of the library to link against the static version of apr-1 in the Windows build.
   apr-util has a single variable for passing the apr-1 lib into its CMakeLists.txt, and apr-util build
   fails while building the shared library (DLL) if we pass the static version of apr-1 lib.
2. Fix up APU_VERSION_STRING_CSV in apu_version.csv by removing ## patterns from the macro's value to
   make it compatible with the version of mingwpy toolchain that we're using for Windows Builds.
