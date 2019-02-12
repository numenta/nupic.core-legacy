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
 * Implementation of the ArrayBase class
 */

#include <iostream> // for ostream
//#include <iomanip>  // for std::setprecision
#include <cstring>  // for memcpy, memcmp
#include <stdlib.h> // for size_t

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

/**
 * This makes a deep copy of the buffer so this class will own the buffer.
 */
ArrayBase::ArrayBase(NTA_BasicType type, void *buffer, size_t count) {
  if (!BasicType::isValid(type)) {
    NTA_THROW << "Invalid NTA_BasicType " << type
              << " used in array constructor";
  }
  type_ = type;
  allocateBuffer(count);
  if (has_buffer()) {
    std::memcpy((char *)getBuffer(), (char *)buffer,
                count * BasicType::getSize(type));
  }
}

/**
 * constructor for Array object containing an SDR.
 * The SDR is copied. Array is the owner of the copy.
 */
ArrayBase::ArrayBase(const SDR &sdr) {
  type_ = NTA_BasicType_SDR;
  auto dim = sdr.dimensions;
  allocateBuffer(dim);
  if (has_buffer()) {
    std::memcpy((char *)getBuffer(), (char *)sdr.getDense().data(), count_);
  }
  // sdr.setDenseInplace();
}

/**
 * Caller does not provide a buffer --
 * Nupic will either provide a buffer via setBuffer or
 * ask the ArrayBase to allocate a buffer via allocateBuffer.
 */
ArrayBase::ArrayBase(NTA_BasicType type) {
  if (!BasicType::isValid(type)) {
    NTA_THROW << "Invalid NTA_BasicType " << type
              << " used in array constructor";
  }
  type_ = type;
  own_ = true;
  releaseBuffer();
}

/**
 * The destructor will result in the shared_ptr being deleted.
 * If this is the last reference to the pointer, and this class owns the buffer,
 * the pointer will be deleted...making sure it will not leak.
 */
ArrayBase::~ArrayBase() {}

/**
 * Ask ArrayBase to allocate its buffer.  This class owns the buffer.
 * If there was already a buffer allocated, it will be released.
 * The buffer will be deleted when the last copy of this class has been deleted.
 */
void ArrayBase::allocateBuffer(size_t count) {
  // Note that you can allocate a buffer of size zero.
  // The C++ spec (5.3.4/7) requires such a new request to return
  // a non-NULL value which is safe to delete.  This allows us to
  // disambiguate uninitialized ArrayBases and ArrayBases initialized with
  // size zero.
  count_ = count;
  capacity_ = count_ * BasicType::getSize(type_);
  if (type_ == NTA_BasicType_SDR) {
    std::vector<UInt> dimension;
    dimension.push_back((UInt)count);
    allocateBuffer(dimension);
  } else {
    std::shared_ptr<char> sp(new char[capacity_],
                             std::default_delete<char[]>());
    buffer_ = sp;
    own_ = true;
  }
}

void ArrayBase::allocateBuffer( const std::vector<UInt> dimensions) { // only for SDR
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Dimensions can only be set on the SDR payload";
  SDR *sdr = new SDR(dimensions);
  std::shared_ptr<char> sp((char *)(sdr));
  buffer_ = sp;
  own_ = true;
  count_ = sdr->size;
  capacity_ = count_ * sizeof(Byte);
}

/**
 * Will fill the buffer with 0's.
 */
void ArrayBase::zeroBuffer() {
  if (has_buffer()) {
    if (type_ == NTA_BasicType_SDR) {
        getSDR()->zero();
    } else
      std::memset(buffer_.get(), 0, capacity_);
  }
}

/**
 * Internal function
 * Use the given pointer as the buffer.
 * The caller is responsible to delete the buffer.
 * This class will NOT own the buffer so when this class and all copies
 * of this class are deleted the buffer will NOT be deleted.
 * NOTE: A crash condition WILL exists if this class is used
 *       after the object pointed to has gone out of scope. No protections.
 * This allows external buffers to be carried in the Array structure.
 */
void ArrayBase::setBuffer(void *buffer, size_t count) {
  NTA_CHECK(type_ != NTA_BasicType_SDR);
  count_ = count;
  capacity_ = count * BasicType::getSize(type_);
  buffer_ = std::shared_ptr<char>((char *)buffer, nonDeleter());
  own_ = false;
}
void ArrayBase::setBuffer(SDR &sdr) {
  type_ = NTA_BasicType_SDR;
  buffer_ = std::shared_ptr<char>((char *)&sdr, nonDeleter());
  count_ = sdr.size;
  capacity_ = count_ * BasicType::getSize(type_);
  own_ = false;
}



void ArrayBase::releaseBuffer() {
  buffer_.reset();
  count_ = 0;
  capacity_ = 0;
}

void *ArrayBase::getBuffer() {
  if (has_buffer()) {
    if (type_ == NTA_BasicType_SDR) {
      return getSDR()->getDense().data();
    }
    return buffer_.get();
  }
  return nullptr;
}

const void *ArrayBase::getBuffer() const {
  if (has_buffer()) {
    if (buffer_ != nullptr && type_ == NTA_BasicType_SDR) {
      return getSDR()->getDense().data();
    }
    return buffer_.get();
  }
  return nullptr;
}

SDR *ArrayBase::getSDR() {
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Does not contain an SDR object";
  SDR *sdr = (SDR *)buffer_.get();
  sdr->setDense(sdr->getDense()); // cleanup cache
  return sdr;
}
const SDR *ArrayBase::getSDR() const {
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Does not contain an SDR object";
  if (has_buffer()) {
    const SDR *sdr = (SDR *)buffer_.get();
    return sdr;
  }
  return nullptr;
}

/**
 * Actual size in bytes of the space allocated for the buffer.
 */
size_t ArrayBase::getBufferSize() const {
  if (has_buffer() && type_ == NTA_BasicType_SDR) {
    return getSDR()->size;
  }
  return capacity_;
}

/**
 * number of elements of the given type in the buffer.
 */
size_t ArrayBase::getCount() const {
  if (has_buffer() && type_ == NTA_BasicType_SDR) {
    return ((SDR *)(buffer_.get()))->size;
  }
  return count_;
};

/**
 * max number of elements this buffer can hold.
 * We use this to determine if there is extra space in the buffer
 * to hold the new data so we can avoid having to re-allocate.
 */
size_t ArrayBase::getMaxElementsCount() const {
  if (has_buffer()) {
    if (type_ == NTA_BasicType_SDR)
      return getCount();
    return capacity_ / BasicType::getSize(type_);
  } else
    return 0;
};

/**
 * This can be used to truncate an array to a smaller size.
 * Not usable with an SDR.
 */
void ArrayBase::setCount(size_t count) {
  NTA_CHECK(type_ != NTA_BasicType_SDR) << "Operation not valid for SDR";
  NTA_ASSERT(count <= getMaxElementsCount())
      << "Cannot set the array count (" << count
      << ") greater than the capacity (" << getMaxElementsCount() << ").";
  count_ = count;
}

/**
 * Return the NTA_BasicType of the current contents.
 */
NTA_BasicType ArrayBase::getType() const { return type_; };

/**
 * Return true if a buffer has been allocated.
 */
bool ArrayBase::has_buffer() const { return (buffer_.get() != nullptr); }

/**
 * Convert the buffer contents of the current ArrayBase into
 * the type of the incoming ArrayBase 'a'. Applying an offset if specified.
 * This may be called multiple times to set values of different offsets.
 * If there is not enough room in the destination buffer a new one is created.
 * After allocating the buffer, zero it to clear zero values (if converting
 * from Sparse to Dense).
 *
 * For Fan-In condition, be sure there is enough space in the buffer before
 * the first conversion to avoid loosing data during re-allocation. Then do
 * them in order so that the largest index is last.
 *
 * Be careful when using this with destination of SDR...it will remove
 * dimensions if buffer is  not big enough.
 *
 * args:
 *    a         - Destination buffer
 *    offset    - Index used as starting index. (defaults to 0)
 *    maxsize   - Total size of destination buffer (if 0, use source capacity)
 *                This is used to allocate destination buffer size (in counts).
 */
void ArrayBase::convertInto(ArrayBase &a, size_t offset, size_t maxsize) const {
  if (maxsize == 0)
    maxsize = getMaxElementsCount() + offset;
  if (maxsize > a.getMaxElementsCount()) {
    a.allocateBuffer(maxsize);
    a.zeroBuffer();
  }
  if (offset == 0) {
    // This should be the first set of a Fan-In.
    a.setCount(maxsize);
  }
  NTA_CHECK(getCount() + offset <= maxsize);
  char *toPtr =  (char *)a.getBuffer(); // char* so it has size
  if (offset)
    toPtr += (offset * BasicType::getSize(a.getType()));
  const void *fromPtr = getBuffer();
  BasicType::convertArray(toPtr, a.type_, fromPtr, type_, getCount());
  a.setCount(offset + getCount());
}

bool ArrayBase::isInstance(const ArrayBase &a) const {
  if (a.buffer_ == nullptr || buffer_ == nullptr)
    return false;
  return (buffer_.get() == a.buffer_.get());
}


///////////////////////////////////////////////////////////////////////////////
//    Compare operators
///////////////////////////////////////////////////////////////////////////////
// Compare contents of two ArrayBase objects
// Note: An Array and an ArrayRef could be the same if type, count, and buffer
// contents are the same.
bool operator==(const ArrayBase &lhs, const ArrayBase &rhs) {
  if (lhs.getType() != rhs.getType() || lhs.getCount() != rhs.getCount())
    return false;
  if (lhs.getCount() == 0)
    return true;
  return (std::memcmp(lhs.getBuffer(), rhs.getBuffer(),
                      lhs.getCount() * BasicType::getSize(lhs.getType())) == 0);
}

////////////////////////////////////////////////////////////////////////////////
//         Stream Serialization (as binary)
////////////////////////////////////////////////////////////////////////////////
void ArrayBase::save(std::ostream &outStream) const {
  outStream << "[ " << count_ << " " << BasicType::getName(type_) << " ";
  if (has_buffer() && type_ == NTA_BasicType_SDR) {
    const SDR *sdr = getSDR();
    sdr->save(outStream);
  } else {

    if (count_ > 0) {
      Size size = count_ * BasicType::getSize(type_);
      outStream.write((const char *)buffer_.get(), size);
    }
  }
  outStream << "]" << std::endl;
}
void ArrayBase::load(std::istream &inStream) {
  std::string tag;
  size_t count;

  NTA_CHECK(inStream.get() == '[')
      << "Binary load of Array, expected starting '['.";
  inStream >> count;
  inStream >> tag;
  type_ = BasicType::parse(tag);
  if (count > 0 && type_ == NTA_BasicType_SDR) {
    SDR *sdr = new SDR();
    sdr->load(inStream);
    std::shared_ptr<char> sp((char *)(sdr));
    buffer_ = sp;
  } else {
    allocateBuffer(count);
    inStream.ignore(1);
    inStream.read(buffer_.get(), capacity_);
  }
  NTA_CHECK(inStream.get() == ']')
      << "Binary load of Array, expected ending ']'.";
  inStream.ignore(1); // skip over the endl
}

////////////////////////////////////////////////////////////////////////////////
//         Stream Serialization  (as Ascii text character strings)
//              [ type count ( item item item ...) ... ]
////////////////////////////////////////////////////////////////////////////////

template <typename T>
static void _templatedStreamBuffer(std::ostream &outStream, const void *inbuf,
                                   size_t numElements) {
  outStream << "( ";

  // Stream the elements
  auto it = (const T *)inbuf;
  auto const end = it + numElements;
  if (it < end) {
    for (; it < end; ++it) {
      outStream << 0 + (*it) << " ";
    }
    // note: Adding 0 to value so Byte displays as numeric.
  }
  outStream << ") ";
}

std::ostream &operator<<(std::ostream &outStream, const ArrayBase &a) {
  auto const inbuf = a.getBuffer();
  auto const numElements = a.getCount();
  auto const elementType = a.getType();

  outStream << "[ " << BasicType::getName(elementType) << " " << numElements
            << " ";

  switch (elementType) {
  case NTA_BasicType_Byte:
    _templatedStreamBuffer<Byte>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Int16:
    _templatedStreamBuffer<Int16>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_UInt16:
    _templatedStreamBuffer<UInt16>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Int32:
    _templatedStreamBuffer<Int32>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_UInt32:
    _templatedStreamBuffer<UInt32>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Int64:
    _templatedStreamBuffer<Int64>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_UInt64:
    _templatedStreamBuffer<UInt64>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Real32:
    _templatedStreamBuffer<Real32>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Real64:
    _templatedStreamBuffer<Real64>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_Bool:
    _templatedStreamBuffer<bool>(outStream, inbuf, numElements);
    break;
  case NTA_BasicType_SDR:
    _templatedStreamBuffer<Byte>(outStream, inbuf, numElements);
    break;
  default:
    NTA_THROW << "Unexpected Element Type: " << elementType;
    break;
  }
  outStream << " ] ";

  return outStream;
}

template <typename T>
static void _templatedStreamBuffer(std::istream &inStream, void *buf,
                                   size_t numElements) {
  std::string v;
  inStream >> v;
  NTA_CHECK(v == "(")
      << "deserialize Array buffer...expected an opening '(' but not found.";

  // Stream the elements
  auto it = (T *)buf;
  auto const end = it + numElements;
  if (it < end) {
    for (; it < end; ++it) {
      inStream >> *it;
    }
  }
  inStream >> v;
  NTA_CHECK(v == ")")
      << "deserialize Array buffer...expected a closing ')' but not found.";
}

std::istream &operator>>(std::istream &inStream, ArrayBase &a) {
  std::string v;
  size_t numElements;

  inStream >> v;
  NTA_CHECK(v == "[")
      << "deserialize Array object...expected an opening '[' but not found.";

  inStream >> v;
  a.type_ = BasicType::parse(v);
  inStream >> numElements;
  if (numElements > 0 && a.type_ == NTA_BasicType_SDR) {
    SDR *sdr = new SDR();
    sdr->load(inStream);
    std::shared_ptr<char> sp((char *)(sdr));
    a.buffer_ = sp;
  } else {
    a.allocateBuffer(numElements);
  }

  if (a.has_buffer()) {
    auto inbuf = a.getBuffer();
    switch (a.type_) {
    case NTA_BasicType_Byte:
      _templatedStreamBuffer<Byte>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_Int16:
      _templatedStreamBuffer<Int16>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_UInt16:
      _templatedStreamBuffer<UInt16>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_Int32:
      _templatedStreamBuffer<Int32>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_UInt32:
      _templatedStreamBuffer<UInt32>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_Int64:
      _templatedStreamBuffer<Int64>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_UInt64:
      _templatedStreamBuffer<UInt64>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_Real32:
      _templatedStreamBuffer<Real32>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_Real64:
      _templatedStreamBuffer<Real64>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_Bool:
      _templatedStreamBuffer<bool>(inStream, inbuf, numElements);
      break;
    case NTA_BasicType_SDR:
      _templatedStreamBuffer<Byte>(inStream, inbuf, numElements);
      break;
    default:
      NTA_THROW << "Unexpected Element Type: " << a.type_;
      break;
    }
  }
  inStream >> v;
  NTA_CHECK(v == "]")
      << "deserialize Array buffer...expected a closing ']' but not found.";
  inStream.ignore(1);

  return inStream;
}

} // namespace nupic
