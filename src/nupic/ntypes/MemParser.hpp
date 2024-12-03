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

/** @file */

#ifndef NTA_MEM_PARSER2_HPP
#define NTA_MEM_PARSER2_HPP

#include <sstream>
#include <vector>

#include "nupic/types/Types.hpp"

namespace nupic {

////////////////////////////////////////////////////////////////////////////
/// Class for parsing numbers and strings out of a memory buffer.
///
/// This provides a significant performance advantage over using the standard
/// C++ stream input operators operating on a stringstream in memory.
///
/// @b Responsibility
///  - provide high level parsing functions for extracing numbers and strings
///     from a memory buffer
///
/// @b Resources/Ownerships:
///  - Owns a memory buffer that it allocates in it's constructor.
///
/// @b Notes:
///  To use this class, you pass in an input stream and a total # of bytes to
///  the constructor. The constructor will then read that number of bytes from
///  the stream into an internal buffer maintained by the MemParser object.
///  Subsequent calls to MemParser::get() will extract numbers/strings from the
///  internal buffer.
///
//////////////////////////////////////////////////////////////////////////////
class MemParser {
public:
  /////////////////////////////////////////////////////////////////////////////////////
  /// Constructor
  ///
  /// @param in     The input stream to get characters from.
  /// @param bytes  The number of bytes to extract from the stream for parsing.
  ///               0 means extract all bytes
  ///////////////////////////////////////////////////////////////////////////////////
  MemParser(std::istream &in, UInt32 bytes = 0);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Destructor
  ///
  /// Free the MemParser object
  ///////////////////////////////////////////////////////////////////////////////////
  virtual ~MemParser();

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read an unsigned integer out of the stream
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void get(unsigned long &val);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read an unsigned long long out of the stream
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void get(unsigned long long &val);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read an signed integer out of the stream
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void get(long &val);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read a double precision floating point number out of the stream
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void get(double &val);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read a double precision floating point number out of the stream
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void get(float &val);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read a string out of the stream
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void get(std::string &val);

  /////////////////////////////////////////////////////////////////////////////////////
  /// >> operator's
  ///////////////////////////////////////////////////////////////////////////////////
  friend MemParser &operator>>(MemParser &in, unsigned long &val) {
    in.get(val);
    return in;
  }

  friend MemParser &operator>>(MemParser &in, unsigned long long &val) {
    in.get(val);
    return in;
  }

  friend MemParser &operator>>(MemParser &in, long &val) {
    in.get(val);
    return in;
  }

  friend MemParser &operator>>(MemParser &in, unsigned int &val) {
    unsigned long lval;
    in.get(lval);
    val = lval;
    return in;
  }

  friend MemParser &operator>>(MemParser &in, int &val) {
    long lval;
    in.get(lval);
    val = lval;
    return in;
  }

  friend MemParser &operator>>(MemParser &in, double &val) {
    in.get(val);
    return in;
  }

  friend MemParser &operator>>(MemParser &in, float &val) {
    in.get(val);
    return in;
  }

  friend MemParser &operator>>(MemParser &in, std::string &val) {
    in.get(val);
    return in;
  }

private:
  std::string str_;
  const char *bufP_;
  UInt32 bytes_;

  const char *startP_;
  const char *endP_;
};

} // namespace nupic

#endif // NTA_MEM_PARSER2_HPP
