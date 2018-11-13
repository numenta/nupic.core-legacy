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

/** @file */

#ifndef NTA_DIRECTORY_HPP
#define NTA_DIRECTORY_HPP

//----------------------------------------------------------------------

#include <string>
#include <nupic/os/Path.hpp>

// Compiler support for <filesystem> in C++17:
// https://en.cppreference.com/w/cpp/compiler_support
//    GCC 7.1 has <experimental/filesystem>, link with -libc++experimental or -lstdc++fs
//    GCC 8 has <filesystem>   link with -lstdc++fs
//    GCC 9   expected to support <filesystem>
//    Clang 4 (XCode10) has no support for <filesystem>, partial C++17
//    Clang 7 has complete <filesystem> support for C++17
//    Visual Studio 2017 15.7 (v19.14)supports <filesystem> with C++17
//    MinGW has no support for filesystem.
//
// If >= C++17 then
//   use std::filesystem, if it exists
//   else, use std::experimental::filesystem if it exists.
//   else, use boost::filesystem
// else use boost::filesystem
// Note: For the boost version link with boost system, filesystem libraries.
#if __cplusplus >= 201703L || (defined(_MSC_VER) && _MSC_VER >= 1914)
  // C++17 or greater
  #if __has_include ( <filesystem> )
    #include <filesystem>
    namespace fs = std::filesystem;
    namespace er = std;
  #else
    #if __has_include ( <experimental/filesystem> )
      #include <experimental/filesystem>
      namespace fs = std::experimental::filesystem;
      namespace er = std;
    #else
      #include <boost/filesystem.hpp>
      namespace fs = boost::filesystem;
      namespace er = boost::system;
      #define USE_BOOST_FILESYSTEM 1
    #endif
  #endif
#else
  // C++11
  #include <boost/filesystem.hpp>
  namespace fs = boost::filesystem;
  namespace er = boost::system;
  #define USE_BOOST_FILESYSTEM 1
#endif

//----------------------------------------------------------------------

namespace nupic {
class Path;

namespace Directory {
// check if a directory exists
bool exists(const std::string &path);

// true if the directory is empty
bool empty(const std::string &path);

// return the amount of available space on this path's device.
Size free_space(const std::string & path);

// get current working directory
std::string getCWD();

// set current working directories
void setCWD(const std::string &path);  // be careful about using this.

// Copy directory tree rooted in 'source' to 'destination'
void copyTree(const std::string &source, const std::string &destination);

// Remove directory tree rooted in 'path'
bool removeTree(const std::string &path, bool noThrow = false);

// Create directory 'path' including all parent directories if missing
// returns the first directory that was actually created.
//
// For example if path is /A/B/C/D
//    if /A/B/C/D exists it returns ""
//    if /A/B exists it returns /A/B/C
//    if /A doesn't exist it returns /A/B/C/D
//
// Failures will throw an exception
void create(const std::string &path, bool otherAccess = false, bool recursive = false);

std::string list(const std::string &path, std::string indent = "");

struct Entry {
  enum Type { FILE, DIRECTORY, LINK, OTHER };

  Type type;
  std::string path;         // full absolute path
  std::string filename;     // just the filename and extension or directory name
};

class Iterator {
public:
  Iterator(const nupic::Path & path);
  Iterator(const std::string & path);
  ~Iterator() {}

  // Resets directory to start. Subsequent call to next()
  // will retrieve the first entry
  void reset();

  // get next directory entry
  Entry *next(Entry &e);

private:
  Iterator() {}
  Iterator(const Iterator &) {}

private:
  fs::path p_;
  fs::directory_iterator current_;
  fs::directory_iterator end_;

};
} // namespace nupic/Directory
} // namespace nupic

#endif // NTA_DIRECTORY_HPP


