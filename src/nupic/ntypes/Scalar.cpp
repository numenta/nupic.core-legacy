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
 * Implementation of the Scalar class
 *
 * A Scalar object is an instance of an NTA_BasicType -- essentially a union
 * It is used internally in the conversion of YAML strings to C++ objects.
 */

#include <nupic/ntypes/Scalar.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

Scalar::Scalar(NTA_BasicType theTypeParam) {
  theType_ = theTypeParam;
  value.uint64 = 0;
}

NTA_BasicType Scalar::getType() { return theType_; }

// gcc 4.2 complains about the template specializations
// in a different namespace if we don't include this
namespace nupic {

template <> Handle Scalar::getValue<Handle>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Handle);
  return value.handle;
}
template <> Byte Scalar::getValue<Byte>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Byte);
  return value.byte;
}
template <> UInt16 Scalar::getValue<UInt16>() const {
  NTA_CHECK(theType_ == NTA_BasicType_UInt16);
  return value.uint16;
}
template <> Int16 Scalar::getValue<Int16>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Int16);
  return value.int16;
}
template <> UInt32 Scalar::getValue<UInt32>() const {
  NTA_CHECK(theType_ == NTA_BasicType_UInt32);
  return value.uint32;
}
template <> Int32 Scalar::getValue<Int32>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Int32);
  return value.int32;
}
template <> UInt64 Scalar::getValue<UInt64>() const {
  NTA_CHECK(theType_ == NTA_BasicType_UInt64);
  return value.uint64;
}
template <> Int64 Scalar::getValue<Int64>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Int64);
  return value.int64;
}
template <> Real32 Scalar::getValue<Real32>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Real32);
  return value.real32;
}
template <> Real64 Scalar::getValue<Real64>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Real64);
  return value.real64;
}
template <> bool Scalar::getValue<bool>() const {
  NTA_CHECK(theType_ == NTA_BasicType_Bool);
  return value.boolean;
}
} // namespace nupic
