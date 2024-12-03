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

#ifndef NTA_F_STREAM_HPP
#define NTA_F_STREAM_HPP

#include <fstream>

namespace nupic {

///////////////////////////////////////////////////////////////////////////////////////
/// IFStream
///
/// @b Responsibility
///
/// Open a file for reading
///
/// @b Description
///
/// This class overrides the open() and constructor methods of the standard
/// ifstream to handle utf-8 paths.
///
/////////////////////////////////////////////////////////////////////////////////////
class IFStream : public std::ifstream {

public:
  //////////////////////////////////////////////////////////////////////////
  /// Construct an OFStream
  /////////////////////////////////////////////////////////////////////////
  IFStream() : std::ifstream() {}

  ///////////////////////////////////////////////////////////////////////////////////
  /// WARNING: the std library does not declare a virtual destructor for
  /// std::basic_ofstream
  ///  or std::basic_ifstream, which we sub-class. Therefore, the destructor for
  ///  this class will NOT be called and therefore it should not allocate any
  ///  data members that need to be deleted at destruction time.
  ///////////////////////////////////////////////////////////////////////////////////
  virtual ~IFStream() {}

  //////////////////////////////////////////////////////////////////////////
  /// Construct an IFStream
  ///
  /// @param filename the name of the file to open
  /// @param mode the open mode
  /////////////////////////////////////////////////////////////////////////
  IFStream(const char *filename, ios_base::openmode mode = ios_base::in)
      : std::ifstream() {
    open(filename, mode);
  }

  //////////////////////////////////////////////////////////////////////////
  /// open the given file by name
  ///
  /// @param filename the name of the file to open
  /// @param mode the open mode
  /////////////////////////////////////////////////////////////////////////
  void open(const char *filename, ios_base::openmode mode = ios_base::in);

  //////////////////////////////////////////////////////////////////////////
  /// print out diagnostic information on a failed open
  /////////////////////////////////////////////////////////////////////////
  static void diagnostics(const char *filename);

}; // end class IFStream

///////////////////////////////////////////////////////////////////////////////////////
/// OFStream
///
/// @b Responsibility
///
/// Open a file for writing
///
/// @b Description
///
/// This class overrides the open() and constructor methods of the standard
/// ofstream to handle utf-8 paths.
///
/////////////////////////////////////////////////////////////////////////////////////
class OFStream : public std::ofstream {

public:
  //////////////////////////////////////////////////////////////////////////
  /// Construct an OFStream
  /////////////////////////////////////////////////////////////////////////
  OFStream() : std::ofstream() {}

  ///////////////////////////////////////////////////////////////////////////////////
  /// WARNING: the std library does not declare a virtual destructor for
  /// std::basic_ofstream
  ///  or std::basic_ifstream, which we sub-class. Therefore, the destructor for
  ///  this class will NOT be called and therefore it should not allocate any
  ///  data members that need to be deleted at destruction time.
  ///////////////////////////////////////////////////////////////////////////////////
  virtual ~OFStream() {}

  //////////////////////////////////////////////////////////////////////////
  /// Construct an OFStream
  ///
  /// @param filename the name of the file to open
  /// @param mode the open mode
  /////////////////////////////////////////////////////////////////////////
  OFStream(const char *filename, ios_base::openmode mode = ios_base::out)
      : std::ofstream() {
    open(filename, mode);
  }

  //////////////////////////////////////////////////////////////////////////
  /// open the given file by name
  ///
  /// @param filename the name of the file to open
  /// @param mode the open mode
  /////////////////////////////////////////////////////////////////////////
  void open(const char *filename, ios_base::openmode mode = ios_base::out);

}; // end class OFStream

class ZLib {
public:
  static void *fopen(const std::string &filename, const std::string &mode,
                     std::string *errorMessage = nullptr);
};

} // end namespace nupic

#endif // NTA_F_STREAM_HPP
