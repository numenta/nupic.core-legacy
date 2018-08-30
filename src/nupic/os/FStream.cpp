/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
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
