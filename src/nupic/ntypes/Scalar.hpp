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
 * Definitions for the Scalar class
 *
 * A Scalar object is an instance of an NTA_BasicType -- essentially a union
 * It is used internally in the conversion of YAML strings to C++ objects.
 */

#ifndef NTA_SCALAR_HPP
#define NTA_SCALAR_HPP

#include <nupic/types/Types.h>
#include <nupic/utils/Log.hpp> // temporary, while implementation is in hpp
#include <string>

namespace nupic {
class Scalar {
public:
  Scalar(NTA_BasicType theTypeParam);

  NTA_BasicType getType();

  template <typename T> T getValue() const;

  union {
    NTA_Handle handle;
    NTA_Byte byte;
    NTA_Int16 int16;
    NTA_UInt16 uint16;
    NTA_Int32 int32;
    NTA_UInt32 uint32;
    NTA_Int64 int64;
    NTA_UInt64 uint64;
    NTA_Real32 real32;
    NTA_Real64 real64;
    bool boolean;
  } value;

private:
  NTA_BasicType theType_;
};

} // namespace nupic

#endif // NTA_SCALAR_HPP
