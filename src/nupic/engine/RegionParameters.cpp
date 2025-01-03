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
 * Implementation of Region methods related to parameters
 */

#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/Types.h>
#include <nupic/utils/Log.hpp>

namespace nupic {

// setParameter

void Region::setParameterInt32(const std::string &name, Int32 value) {
  impl_->setParameterInt32(name, (Int64)-1, value);
}

void Region::setParameterUInt32(const std::string &name, UInt32 value) {
  impl_->setParameterUInt32(name, (Int64)-1, value);
}

void Region::setParameterInt64(const std::string &name, Int64 value) {
  impl_->setParameterInt64(name, (Int64)-1, value);
}

void Region::setParameterUInt64(const std::string &name, UInt64 value) {
  impl_->setParameterUInt64(name, (Int64)-1, value);
}

void Region::setParameterReal32(const std::string &name, Real32 value) {
  impl_->setParameterReal32(name, (Int64)-1, value);
}

void Region::setParameterReal64(const std::string &name, Real64 value) {
  impl_->setParameterReal64(name, (Int64)-1, value);
}

void Region::setParameterHandle(const std::string &name, Handle value) {
  impl_->setParameterHandle(name, (Int64)-1, value);
}

void Region::setParameterBool(const std::string &name, bool value) {
  impl_->setParameterBool(name, (Int64)-1, value);
}

// getParameter

Int32 Region::getParameterInt32(const std::string &name) const {
  return impl_->getParameterInt32(name, (Int64)-1);
}

Int64 Region::getParameterInt64(const std::string &name) const {
  return impl_->getParameterInt64(name, (Int64)-1);
}

UInt32 Region::getParameterUInt32(const std::string &name) const {
  return impl_->getParameterUInt32(name, (Int64)-1);
}

UInt64 Region::getParameterUInt64(const std::string &name) const {
  return impl_->getParameterUInt64(name, (Int64)-1);
}

Real32 Region::getParameterReal32(const std::string &name) const {
  return impl_->getParameterReal32(name, (Int64)-1);
}

Real64 Region::getParameterReal64(const std::string &name) const {
  return impl_->getParameterReal64(name, (Int64)-1);
}

Handle Region::getParameterHandle(const std::string &name) const {
  return impl_->getParameterHandle(name, (Int64)-1);
}

bool Region::getParameterBool(const std::string &name) const {
  return impl_->getParameterBool(name, (Int64)-1);
}

// array parameters

void Region::getParameterArray(const std::string &name, Array &array) const {
  size_t count = impl_->getParameterArrayCount(name, (Int64)(-1));
  // Make sure we have a buffer to put the data in
  if (array.getBuffer() != nullptr) {
    // Buffer has already been allocated. Make sure it is big enough
    if (array.getCount() > count)
      NTA_THROW << "getParameterArray -- supplied buffer for parameter " << name
                << " can hold " << array.getCount()
                << " elements but parameter count is " << count;
  } else {
    array.allocateBuffer(count);
  }

  impl_->getParameterArray(name, (Int64)-1, array);
}

void Region::setParameterArray(const std::string &name, const Array &array) {
  // We do not check the array size here because it would be
  // expensive -- involving a check against the nodespec,
  // and only usable in the rare case that the nodespec specified
  // a fixed size. Instead, the implementation can check the size.
  impl_->setParameterArray(name, (Int64)-1, array);
}

void Region::setParameterString(const std::string &name, const std::string &s) {
  impl_->setParameterString(name, (Int64)-1, s);
}

std::string Region::getParameterString(const std::string &name) {
  return impl_->getParameterString(name, (Int64)-1);
}

bool Region::isParameterShared(const std::string &name) const {
  return impl_->isParameterShared(name);
}

} // namespace nupic
