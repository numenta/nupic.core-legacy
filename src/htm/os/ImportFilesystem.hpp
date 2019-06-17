/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

#ifndef NTA_IMPORT_FILESYSTEM_HPP
#define NTA_IMPORT_FILESYSTEM_HPP

//----------------------------------------------------------------------

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
#if __cplusplus >= 201703L
  // C++17 or greater
  #if __has_include ( <filesystem> )
    #include <filesystem>
    namespace fs = std::filesystem;
  #else
    #if __has_include ( <experimental/filesystem> )
      #include <experimental/filesystem>
      namespace fs = std::experimental::filesystem;
    #else
      #define USE_BOOST_FILESYSTEM 1
      #include <boost/filesystem.hpp>
    #endif
  #endif
#else
  // C++11
  #include <boost/filesystem.hpp>
  #define USE_BOOST_FILESYSTEM 1
#endif

#ifdef USE_BOOST_FILESYSTEM
  namespace fs = boost::filesystem;
  namespace er = boost::system;
  #define FS_PermNone    fs::perms::no_perms
  #define FS_OwnerExec   fs::perms::owner_exe
  #define FS_GroupExec   fs::perms::group_exe
  #define FS_OthersExec  fs::perms::others_exe
  #define FS_Overwrite   fs::copy_option::overwrite_if_exists
#else
  namespace er = std;
  #define FS_PermNone    fs::perms::none
  #define FS_OwnerExec   fs::perms::owner_exec
  #define FS_GroupExec   fs::perms::group_exec
  #define FS_OthersExec  fs::perms::others_exec
  #define FS_Overwrite   fs::copy_options::overwrite_existing
#endif

#endif // NTA_IMPORT_FILESYSTEM_HPP