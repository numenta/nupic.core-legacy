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
 *
 */

#include "nupic/ntypes/MemParser.hpp"
#include "nupic/ntypes/MemStream.hpp"
#include "nupic/utils/Log.hpp"
#include <cstdlib>
#include <cstring>

using namespace std;
namespace nupic {

////////////////////////////////////////////////////////////////////////////
// MemParser constructor
//////////////////////////////////////////////////////////////////////////////
MemParser::MemParser(std::istream &in, UInt32 bytes) {
  if (bytes == 0) {
    // -----------------------------------------------------------------------------
    // Read all available bytes from the stream
    // -----------------------------------------------------------------------------
    std::string data;
    const int chunkSize = 0x10000;
    auto chunkP = new char[chunkSize];
    while (!in.eof()) {
      in.read(chunkP, chunkSize);
      NTA_CHECK(in.good() || in.eof())
          << "MemParser::MemParser() - error reading data from stream";
      data.append(chunkP, in.gcount());
    }

    bytes_ = (UInt32)data.size();
    bufP_ = new char[bytes_ + 1];
    NTA_CHECK(bufP_ != nullptr) << "MemParser::MemParser() - out of memory";
    ::memmove((void *)bufP_, data.data(), bytes_);
    ((char *)bufP_)[bytes_] = 0;

    delete[] chunkP;
  } else {
    // -----------------------------------------------------------------------------
    // Read given # of bytes from the stream
    // -----------------------------------------------------------------------------
    bytes_ = bytes;
    bufP_ = new char[bytes_ + 1];
    NTA_CHECK(bufP_ != nullptr) << "MemParser::MemParser() - out of memory";

    in.read((char *)bufP_, bytes);
    ((char *)bufP_)[bytes] = 0;
    NTA_CHECK(in.good())
        << "MemParser::MemParser() - error reading data from stream";
  }
  // Setup start and end pointers
  startP_ = bufP_;
  endP_ = startP_ + bytes_;
}

////////////////////////////////////////////////////////////////////////////
// Destructor
//////////////////////////////////////////////////////////////////////////////
MemParser::~MemParser() { delete[] bufP_; }

////////////////////////////////////////////////////////////////////////////
// Read an unsigned integer number out
//////////////////////////////////////////////////////////////////////////////
void MemParser::get(unsigned long &val) {
  const char *prefix = "MemParser::get(unsigned long&) - ";
  char *endP;

  NTA_CHECK(startP_ < endP_) << prefix << "EOF";

  val = ::strtoul(startP_, &endP, 0);

  NTA_CHECK(endP != startP_ && endP <= endP_)
      << prefix << "parse error, not a valid integer";

  startP_ = endP;
}

////////////////////////////////////////////////////////////////////////////
// Read an unsigned long long number out
//////////////////////////////////////////////////////////////////////////////
void MemParser::get(unsigned long long &val) {
  const char *prefix = "MemParser::get(unsigned long long&) - ";
  char *endP;

  NTA_CHECK(startP_ < endP_) << prefix << "EOF";

  val = ::strtoul(startP_, &endP, 0);

  NTA_CHECK(endP != startP_ && endP <= endP_)
      << prefix << "parse error, not a valid integer";

  startP_ = endP;
}

////////////////////////////////////////////////////////////////////////////
// Read an signed integer number out
//////////////////////////////////////////////////////////////////////////////
void MemParser::get(long &val) {
  const char *prefix = "MemParser::get(long&) - ";
  char *endP;

  NTA_CHECK(startP_ < endP_) << prefix << "EOF";

  val = ::strtol(startP_, &endP, 0);

  NTA_CHECK(endP != startP_ && endP <= endP_)
      << prefix << "parse error, not a valid integer";

  startP_ = endP;
}

////////////////////////////////////////////////////////////////////////////
// Read a double-precision float out
//////////////////////////////////////////////////////////////////////////////
void MemParser::get(double &val) {
  const char *prefix = "MemParser::get(double&) - ";
  char *endP;

  NTA_CHECK(startP_ < endP_) << prefix << "EOF";

  val = ::strtod(startP_, &endP);

  NTA_CHECK(endP != startP_ && endP <= endP_)
      << prefix << "parse error, not a valid floating point value";

  startP_ = endP;
}

////////////////////////////////////////////////////////////////////////////
// Read a single-precision float out
//////////////////////////////////////////////////////////////////////////////
void MemParser::get(float &val) {
  double f;
  get(f);
  val = (float)f;
}

////////////////////////////////////////////////////////////////////////////
// Read string out
//////////////////////////////////////////////////////////////////////////////
void MemParser::get(std::string &val) {
  const char *prefix = "MemParser::get(string&) - ";

  // First, skip leading white space
  const char *cP = startP_;
  while (cP < endP_) {
    char c = *cP;
    if (c != 0 && c != ' ' && c != '\t' && c != '\n' && c != '\r')
      break;
    cP++;
  }
  NTA_CHECK(cP < endP_) << prefix << "EOF";

  size_t len = strcspn(cP, " \t\n\r");
  NTA_CHECK(len > 0) << prefix << "parse error, not a valid string";

  val.assign(cP, len);
  startP_ = cP + len;
}

} // namespace nupic
