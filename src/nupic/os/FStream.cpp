/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Definitions for the FStream classes
 *
 * These classes are versions of ifstream and ofstream that accept platform
 * independent (i.e. windows or unix) utf-8 path specifiers for their
 * constructor and open() methods.
 *
 * The native ifstream and ofstream classes on unix already accept UTF-8, but on
 * windows, we must convert the utf-8 path to unicode and then pass it to the
 * 'w' version of ifstream or ofstream
 */

#include "nupic/os/FStream.hpp"
#include "nupic/os/Directory.hpp"
#include "nupic/os/Env.hpp"
#include "nupic/os/Path.hpp"
#include "nupic/utils/Log.hpp"
#include <cstdlib>
#include <fstream>
#if defined(NTA_OS_WINDOWS) && !defined(NTA_COMPILER_GNU)
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif
#include <zlib.h>

using namespace nupic;

//////////////////////////////////////////////////////////////////////////
/// Print out diagnostic information when a file open fails
/////////////////////////////////////////////////////////////////////////
void IFStream::diagnostics(const char *filename) {

  bool forceLog = false;

  // We occasionally get error 116(ESTALE) "Stale NFS file handle" (TOO-402)
  // when creating
  //  a file using OFStream::open() on a shared drive on unix systems. We found
  //  that if we perform a directory listing after encountering the error, that
  //  a retry immediately after is successful. So.... we log this information if
  //  we get errno==ESTALE OR if NTA_FILE_LOGGING is set.
#ifdef ESTALE
  if (errno == ESTALE)
    forceLog = true;
#endif

  if (forceLog || ::getenv("NTA_FILE_LOGGING")) {
    NTA_DEBUG << "FStream::open() failed opening file " << filename
              << "; errno = " << errno << "; errmsg = " << strerror(errno)
              << "; cwd = " << Directory::getCWD();

    Directory::Iterator di(Directory::getCWD());
    Directory::Entry e;
    while (di.next(e)) {
      NTA_DEBUG << "FStream::open() ls: " << e.path;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
/// open the given file by name
/////////////////////////////////////////////////////////////////////////
void IFStream::open(const char *filename, ios_base::openmode mode) {
#if defined(NTA_OS_WINDOWS) && !defined(NTA_COMPILER_GNU)
  std::wstring pathW = Path::utf8ToUnicode(filename);
  std::ifstream::open(pathW.c_str(), mode);
#else
  std::ifstream::open(filename, mode);
#endif

  // Check for error
  if (!is_open()) {
    IFStream::diagnostics(filename);
// On unix, running nfs, we occasionally get errors opening a file on an nfs
// drive and it seems that simply doing a retry makes it successful
#if !defined(NTA_OS_WINDOWS)
    std::ifstream::clear();
    std::ifstream::open(filename, mode);
#endif
  }
}

//////////////////////////////////////////////////////////////////////////
/// open the given file by name
/////////////////////////////////////////////////////////////////////////
void OFStream::open(const char *filename, ios_base::openmode mode) {
#if defined(NTA_OS_WINDOWS) && !defined(NTA_COMPILER_GNU)
  std::wstring pathW = Path::utf8ToUnicode(filename);
  std::ofstream::open(pathW.c_str(), mode);
#else
  std::ofstream::open(filename, mode);
#endif

  // Check for error
  if (!is_open()) {
    IFStream::diagnostics(filename);
// On unix, running nfs, we occasionally get errors opening a file on an nfs
// drive and it seems that simply doing a retry makes it successful
#if !(defined(NTA_OS_WINDOWS) && !defined(NTA_COMPILER_GNU))
    std::ofstream::clear();
    std::ofstream::open(filename, mode);
#endif
  }
}

void *ZLib::fopen(const std::string &filename, const std::string &mode,
                  std::string *errorMessage) {
  if (mode.empty())
    throw std::invalid_argument("Mode may not be empty.");

#if defined(NTA_OS_WINDOWS) && !defined(NTA_COMPILER_GNU)
  std::wstring wfilename(Path::utf8ToUnicode(filename));
  int cflags = _O_BINARY;
  int pflags = 0;
  if (mode[0] == 'r') {
    cflags |= _O_RDONLY;
  } else if (mode[0] == 'w') {
    cflags |= _O_TRUNC | _O_CREAT | _O_WRONLY;
    pflags |= _S_IREAD | _S_IWRITE;
  } else if (mode[0] == 'a') {
    cflags |= _O_APPEND | _O_CREAT | _O_WRONLY;
    pflags |= _S_IREAD | _S_IWRITE;
  } else {
    throw std::invalid_argument("Mode must start with 'r', 'w' or 'a'.");
  }

  int fd = _wopen(wfilename.c_str(), cflags, pflags);
  gzFile fs = gzdopen(fd, mode.c_str());

  if (fs == 0) {
    // TODO: Build an error message for Windows.
  }

#else
  gzFile fs = nullptr;
  { // zlib may not be thread-safe in its current compiled state.
    int attempts = 0;
    const int maxAttempts = 1;
    int lastError = 0;
    while (1) {
      fs = gzopen(filename.c_str(), mode.c_str());
      if (fs)
        break;

      int error = errno;
      if (error != lastError) {
        std::string message("Unknown error.");
        // lastError = error;
        switch (error) {
        case Z_STREAM_ERROR:
          message = "Zlib stream error.";
          break;
        case Z_DATA_ERROR:
          message = "Zlib data error.";
          break;
        case Z_MEM_ERROR:
          message = "Zlib memory error.";
          break;
        case Z_BUF_ERROR:
          message = "Zlib buffer error.";
          break;
        case Z_VERSION_ERROR:
          message = "Zlib version error.";
          break;
        default:
          message = ::strerror(error);
          break;
        }
        if (errorMessage) {
          *errorMessage = message;
        } else if (maxAttempts >
                   1) { // If we will try again, warn about failure.
          std::cerr << "Warning: Failed to open file '" << filename
                    << "': " << message << std::endl;
        }
      }
      if ((++attempts) >= maxAttempts)
        break;
      ::usleep(10000);
    }
  }
#endif
  return fs;
}
