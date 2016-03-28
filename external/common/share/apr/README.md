unix/apr-1.5.2.tar.gz from https://apr.apache.org/

We use the apr unix sources for both unix (linux, Mac OS X) and windows builds.
apr provides CMakeLists.txt for Windows builds; it uses configure-based builds for unix (and friends).

nupic.core's external "package" wraps the build of this library in a cmake module.

We patch the apr sources as follows:

1. Comment out the shared library libapr-1 cmake target (used on Windows builds) to prevent it from
   building unnecessarily.
2. Fix up APR_VERSION_STRING_CSV in apr_version.csv by removing ## patterns from the macro's value to
   make it compatible with the version of mingwpy toolchain that we're using for Windows Builds
