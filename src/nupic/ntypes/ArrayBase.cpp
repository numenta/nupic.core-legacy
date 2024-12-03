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
 * Implementation of the ArrayBase class
 */

#include <iostream> // for ostream
#include <stdlib.h> // for size_t

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

/**
 * Caller provides a buffer to use.
 * NuPIC always copies data into this buffer
 * Caller frees buffer when no longer needed.
 */
ArrayBase::ArrayBase(NTA_BasicType type, void *buffer, size_t count)
    : buffer_((char *)buffer), count_(count), type_(type), own_(false) {
  if (!BasicType::isValid(type)) {
    NTA_THROW << "Invalid NTA_BasicType " << type
              << " used in array constructor";
  }
  bufferSize_ = count * BasicType::getSize(type);
}

/**
 * Caller does not provide a buffer --
 * Nupic will either provide a buffer via setBuffer or
 * ask the ArrayBase to allocate a buffer via allocateBuffer.
 */
ArrayBase::ArrayBase(NTA_BasicType type)
    : buffer_(nullptr), count_(0), type_(type), own_(false), bufferSize_(0) {
  if (!BasicType::isValid(type)) {
    NTA_THROW << "Invalid NTA_BasicType " << type
              << " used in array constructor";
  }
}

/**
 * The destructor calls releaseBuffer() to make sure the ArrayBase
 * doesn't leak.
 */
ArrayBase::~ArrayBase() { releaseBuffer(); }

/**
 * Ask ArrayBase to allocate its buffer
 */
void ArrayBase::allocateBuffer(size_t count) {
  if (buffer_ != nullptr) {
    NTA_THROW
        << "allocateBuffer -- buffer already set. Use releaseBuffer first";
  }
  count_ = count;
  // Note that you can allocate a buffer of size zero.
  // The C++ spec (5.3.4/7) requires such a new request to return
  // a non-NULL value which is safe to delete.  This allows us to
  // disambiguate uninitialized ArrayBases and ArrayBases initialized with
  // size zero.
  bufferSize_ = count_ * BasicType::getSize(type_);
  buffer_ = new char[bufferSize_];
  own_ = true;
}

void ArrayBase::setBuffer(void *buffer, size_t count) {
  if (buffer_ != nullptr) {
    NTA_THROW << "setBuffer -- buffer already set. Use releaseBuffer first";
  }
  buffer_ = (char *)buffer;
  count_ = count;
  own_ = false;
  bufferSize_ = count_ * BasicType::getSize(type_);
}

void ArrayBase::releaseBuffer() {
  if (buffer_ == nullptr)
    return;
  if (own_)
    delete[] buffer_;
  buffer_ = nullptr;
  count_ = 0;
  bufferSize_ = 0;
}

void *ArrayBase::getBuffer() const { return buffer_; }

size_t ArrayBase::getBufferSize() const { return bufferSize_; }

// number of elements of given type in the buffer
size_t ArrayBase::getCount() const { return count_; };

// max number of elements this buffer can hold
size_t ArrayBase::getMaxElementsCount() const {
  return bufferSize_ / BasicType::getSize(type_);
};

void ArrayBase::setCount(size_t count) {
  NTA_CHECK(count * BasicType::getSize(type_) <= bufferSize_)
      << "Invalid count value of " << count << " given, "
      << "count must be " << bufferSize_ / BasicType::getSize(type_)
      << " or less";
  count_ = count;
}

NTA_BasicType ArrayBase::getType() const { return type_; };

namespace nupic {
std::ostream &operator<<(std::ostream &outStream, const ArrayBase &a) {
  auto const inbuf = a.getBuffer();
  auto const numElements = a.getCount();
  auto const elementType = a.getType();

  switch (elementType) {
  case NTA_BasicType_Byte:
    ArrayBase::_templatedStreamBuffer<NTA_Byte>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Int16:
    ArrayBase::_templatedStreamBuffer<NTA_Int16>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_UInt16:
    ArrayBase::_templatedStreamBuffer<NTA_UInt16>(outStream, inbuf,
                                                  numElements);
    break;
  case NTA_BasicType_Int32:
    ArrayBase::_templatedStreamBuffer<NTA_Int32>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_UInt32:
    ArrayBase::_templatedStreamBuffer<NTA_UInt32>(outStream, inbuf,
                                                  numElements);
    break;
  case NTA_BasicType_Int64:
    ArrayBase::_templatedStreamBuffer<NTA_Int64>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_UInt64:
    ArrayBase::_templatedStreamBuffer<NTA_UInt64>(outStream, inbuf,
                                                  numElements);
    break;
  case NTA_BasicType_Real32:
    ArrayBase::_templatedStreamBuffer<NTA_Real32>(outStream, inbuf,
                                                  numElements);
    break;
  case NTA_BasicType_Real64:
    ArrayBase::_templatedStreamBuffer<NTA_Real64>(outStream, inbuf,
                                                  numElements);
    break;
  case NTA_BasicType_Handle:
    ArrayBase::_templatedStreamBuffer<NTA_Handle>(outStream, inbuf,
                                                  numElements);
    break;
  case NTA_BasicType_Bool:
    ArrayBase::_templatedStreamBuffer<bool>(outStream, inbuf, numElements);
    break;
  default:
    NTA_THROW << "Unexpected Element Type: " << elementType;
    break;
  }

  return outStream;
}

} // namespace nupic
