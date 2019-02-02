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
 * Implementation of utility functions for string conversion
 */
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING 1


#include <nupic/utils/Log.hpp>
#include <nupic/utils/StringUtils.hpp>
#include <cstring>
#include <cstdlib> // for wcstombs and mbstowcs

using namespace nupic;

std::string StringUtils::trim(const std::string &s) {
  size_t i,j;
  for(i = 0; i < s.length(); i++)
	if (!std::isspace(s[i])) break;
  for(j = s.length(); j > i; j--)
    if (!std::isspace(s[j-1])) break;
  return s.substr(i, j-i);
}


bool StringUtils::toBool(const std::string &s, bool throwOnError, bool *fail) {
  if (fail)
    *fail = false;
  bool b = false;
  std::string us(s);
  std::transform(us.begin(), us.end(), us.begin(), tolower);
  if (us == "true" || us == "yes" || us == "1") {
    b = true;
  } else if (us == "false" || us == "no" || us == "0") {
    b = false;
  } else if (!throwOnError) {
    if (fail)
      *fail = true;
  } else {
    NTA_THROW << "StringUtils::toBool: tried to parse non-boolean string \"" << s << "\"";
  }
  return b;
}


Real32 StringUtils::toReal32(const std::string& s, bool throwOnError, bool * fail) {
  if (fail)
    *fail = false;
  Real32 r;
  std::istringstream ss(s);
  ss >> r;
  if (ss.fail() || !ss.eof()) {
    if (throwOnError) {
      NTA_THROW << "StringUtils::toReal32 -- invalid string \"" << s << "\"";
    } else {
      if (fail)
        *fail = true;
    }
  }

  return r;
}

UInt32 StringUtils::toUInt32(const std::string& s, bool throwOnError, bool * fail)
{
  if (fail)
    *fail = false;
  UInt32 i;
  std::istringstream ss(s);
  ss >> i;
  if (ss.fail() || !ss.eof()) {
    if (throwOnError) {
      NTA_THROW << "StringUtils::toInt -- invalid string \"" << s << "\"";
    } else {
      if (fail)
        *fail = true;
    }
  }

  return i;
}

Int32 StringUtils::toInt32(const std::string& s, bool throwOnError, bool * fail) {
  if (fail)
    *fail = false;
  Int32 i;
  std::istringstream ss(s);
  ss >> i;
  if (ss.fail() || !ss.eof()) {
    if (throwOnError) {
      NTA_THROW << "StringUtils::toInt -- invalid string \"" << s << "\"";
    } else {
      if (fail)
        *fail = true;
    }
  }

  return i;
}

UInt64 StringUtils::toUInt64(const std::string& s, bool throwOnError, bool * fail) {
  if (fail)
    *fail = false;
  UInt64 i;
  std::istringstream ss(s);
  ss >> i;
  if (ss.fail() || !ss.eof()) {
    if (throwOnError) {
      NTA_THROW << "StringUtils::toInt -- invalid string \"" << s << "\"";
    } else {
      if (fail)
        *fail = true;
    }
  }

  return i;
}


size_t StringUtils::toSizeT(const std::string& s, bool throwOnError, bool * fail) {
  if (fail)
    *fail = false;
  size_t i;
  std::istringstream ss(s);
  ss >> i;
  if (ss.fail() || !ss.eof()) {
    if (throwOnError) {
      NTA_THROW << "StringUtils::toSizeT -- invalid string \"" << s << "\"";
    } else {
      if (fail)
        *fail = true;
    }
  }
  return i;
}

bool StringUtils::startsWith(const std::string& s, const std::string& prefix) {
  return s.find(prefix) == 0;
}

bool StringUtils::endsWith(const std::string& s, const std::string& ending) {
  if (ending.size() > s.size())
    return false;
  size_t found = s.rfind(ending);
  if (found == std::string::npos)
    return false;
  if (found != s.size() - ending.size())
    return false;
  return true;
}



std::string StringUtils::fromInt(long long i) {
  std::stringstream ss;
  ss << i;
  return ss.str();
}

/////////////////////////////////////////////////////////////////////////
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

std::string StringUtils::base64Encode(const void* bytes_to_encode, Size in_len) {
  const unsigned char* ptr = (const unsigned char*)bytes_to_encode;
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *ptr++;
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;

}


std::string StringUtils::base64Encode(const std::string& s)
{
    return StringUtils::base64Encode(&s[0], s.size());
}


std::string StringUtils::base64Decode(const std::string& encoded_string) {
  size_t in_len = encoded_string.size();
  size_t i = 0;
  size_t j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = static_cast<unsigned char>(base64_chars.find(char_array_4[i]));

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }

  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = static_cast<unsigned char>(base64_chars.find(char_array_4[j]));

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}
/////////////////////////////////////////////////////////////////////////


#define HEXIFY(val) ((val) > 9 ? ('a' + (val)-10) : ('0' + (val)))

std::string StringUtils::hexEncode(const void *buf, Size inLen) {
  std::string s(inLen * 2, '\0');
  const unsigned char *charbuf = (const unsigned char *)buf;
  for (Size i = 0; i < inLen; i++) {
    unsigned char x = charbuf[i];
    // high order bits
    unsigned char val = x >> 4;
    s[i * 2] = HEXIFY(val);
    val = x & 0xF;
    s[i * 2 + 1] = HEXIFY(val);
  }
  return s;
}



//--------------------------------------------------------------------------------
void StringUtils::toIntList(const std::string &s, std::vector<Int> &list,  bool allowAll, bool asRanges) {
  if (!toIntListNoThrow(s, list, allowAll, asRanges)) {
    throw(std::runtime_error("toIntList() - Invalid string: " + s));
  }
}

//--------------------------------------------------------------------------------
bool StringUtils::toIntListNoThrow(const std::string &s, std::vector<Int> &list,bool allowAll, bool asRanges) {

  UInt startNum, endNum;
  const char *startP = s.c_str();
  char *endP;

  // Set global errno to 0. strtoul sets this if a conversion error occurs.
  errno = 0;

  // Loop through the string
  list.clear();

  // Skip white space at start
  while (*startP && isspace(*startP))
    startP++;

  // Do we allow all?
  if (allowAll) {
    if (!strncmp(startP, "all", 3) && startP[3] == 0)
      return true;
    if (startP[0] == 0)
      return true;
  } else {
    if (startP[0] == 0)
      return false;
  }

  while (*startP) {
    // ------------------------------------------------------------------------------
    // Get first digit
    startNum = strtoul(startP, &endP, 10 /*base*/);
    if (errno != 0)
      return false;
    startP = endP;

    // Skip white space
    while (*startP && isspace(*startP))
      startP++;

    // ------------------------------------------------------------------------------
    // Do we have a '-'? If so, get the second number
    if (*startP == '-') {
      startP++;
      endNum = strtoul(startP, &endP, 10 /*base*/);
      if (errno != 0)
        return false;
      startP = endP;

      // Store all number into the vector
      if (endNum < startNum)
        return false;
      if (asRanges) {
        list.push_back((Int)startNum);
        list.push_back((Int)(endNum - startNum + 1));
      } else {
        for (UInt i = startNum; i <= endNum; i++)
          list.push_back((Int)i);
      }

      // Skip white space
      while (*startP && isspace(*startP))
        startP++;
    } else {
      list.push_back((Int)startNum);
      if (asRanges)
        list.push_back((Int)1);
    }

    // Done if end of string
    if (*startP == 0)
      break;

    // ------------------------------------------------------------------------------
    // Must have a comma between entries
    if (*startP++ != ',')
      return false;

    // Skip white space after the comma
    while (*startP && isspace(*startP))
      startP++;

    // Must be more digits after the comma
    if (*startP == 0)
      return false;
  }

  return true;
}


//--------------------------------------------------------------------------------
std::shared_ptr<Byte> StringUtils::toByteArray(const std::string &s,  Size bitCount) {
  // Get list of integers
  std::vector<Int> list;
  toIntList(s, list, true /*allowAll*/);
  if (list.empty())
    return std::shared_ptr<Byte>(nullptr);

  // Put this into the mask
  Size numBytes = (bitCount+7) / 8;
  std::shared_ptr<Byte> mask(new Byte[numBytes], std::default_delete<Byte[]>());
  Byte* maskP = mask.get();
  ::memset(maskP, 0, numBytes);
  for (auto &elem : list) {
    UInt entry = elem;
    if (entry >= bitCount)
      NTA_THROW << "toByteArray() - "
                << "The list " << s
                << " contains an entry greater than the max allowed of "
                << bitCount;
    maskP[entry / 8] |= 1 << (entry % 8);
  }

  // Return it
  return mask;
}
//--------------------------------------------------------------------------------

// codecvt()is deprecated in c++17 but no replacement is offered.
// Apparently it was not actually implemented in C++11 on some compilers.
// So we will use mbstowcs() and wcstombs()
// Not the best but will work for filenames.
// WARNING: not threadsafe
std::wstring StringUtils::utf8ToUnicode(const std::string &str)
{
	std::wstring ws(str.size(), L' '); // overestimate number of code points
	ws.resize(std::mbstowcs(&ws[0], str.c_str(), str.size())); // shink to fit
	return ws;
}

std::string StringUtils::unicodeToUtf8(const std::wstring &wstr)
{
	size_t size = 0;
	char * lc = ::setlocale(LC_ALL, "en_US.utf8"); // determines code page generated
	std::string str(wstr.size()*3, ' ');  // overestimate number of bytes and create space
	size = std::wcstombs(&str[0], &wstr[0], wstr.size());
	::setlocale(LC_ALL, lc); // restore locale
	str.resize(size);
	return str;
}

