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
  if (count_ > 0) {
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
  } else {
    releaseBuffer();
  }
}

void ArrayBase::allocateBuffer(
    const std::vector<UInt> dimensions) { // only for SDR
  NTA_CHECK(type_ == NTA_BasicType_SDR)
      << "Dimensions can only be set on the SDR payload";
  size_t s = 1;
  for (UInt dim : dimensions)
    s *= dim;
  if (dimensions.size() > 0 && s > 0) {
    SDR *sdr = new SDR(dimensions);
    std::shared_ptr<char> sp((char *)(sdr));
    buffer_ = sp;
    own_ = true;
    count_ = sdr->size;
    capacity_ = count_ * sizeof(Byte);
  } else {
    releaseBuffer();
  }
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
  count_ = count;
  capacity_ = count * BasicType::getSize(type_);
  if (count > 0) {
    buffer_ = std::shared_ptr<char>((char *)buffer, nonDeleter());
    own_ = false;
  } else {
    releaseBuffer();
  }
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
  } else
    return nullptr;
}
const void *ArrayBase::getBuffer() const {
  if (has_buffer()) {
    if (buffer_ != nullptr && type_ == NTA_BasicType_SDR) {
      return getSDR()->getDense().data();
    }
    return buffer_.get();
  } else
    return nullptr;
}

SDR *ArrayBase::getSDR() {
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Does not contain an SDR object";
  NTA_CHECK(has_buffer()) << "Empty, does not contain an SDR object";
  SDR *sdr = (SDR *)buffer_.get();
  sdr->setDense(sdr->getDense()); // cleanup cache
  return sdr;
}
const SDR *ArrayBase::getSDR() const {
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Does not contain an SDR object";
  NTA_CHECK(has_buffer()) << "Empty, does not contain an SDR object";
  const SDR *sdr = (SDR *)buffer_.get();
  return sdr;
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
bool ArrayBase::has_buffer() const { return (capacity_ > 0); }

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
    if (a.getType() == NTA_BasicType_Sparse)
      a.setCount(0);
    else
      a.setCount(maxsize);
  }
  NTA_CHECK(getCount() + offset <= maxsize);
  if (type_ == NTA_BasicType_Sparse)
    fromSparse(a, offset);
  else if (a.getType() == NTA_BasicType_Sparse)
    toSparse(a, offset);
  else {
    char *toPtr =
        (char *)a.getBuffer(); // type as char* so there is an element size
    if (offset)
      toPtr += (offset * BasicType::getSize(a.getType()));
    const void *fromPtr = getBuffer();
    BasicType::convertArray(toPtr, a.type_, fromPtr, type_, getCount());
    a.setCount(offset + getCount());
  }
}

bool ArrayBase::isInstance(const ArrayBase &a) const {
  if (a.buffer_ == nullptr || buffer_ == nullptr)
    return false;
  return (buffer_.get() == a.buffer_.get());
}

template <typename T>
static void NonZeroT(ArrayBase &a, const ArrayBase *b, size_t offset) {
  // Helper function for toSparse().
  // populate the new array in a with indexes of non-zero values in b.
  // NOTE: NTA_BasicType_SDR and NTA_BasicType_Sparse not handled here.
  T *from = (T *)b->getBuffer();
  size_t j = a.getCount(); // current end of buffer.
  UInt32 *Destptr = (UInt32 *)a.getBuffer();
  for (size_t i = 0u; i < b->getCount(); i++) {
    if (from[i])
      Destptr[j++] = (UInt32)i + (UInt32)offset;
  }
  a.setCount(j); // set new end of buffer.
}

// populate the given array a with the a sparse version of the current array.
// Destination buffer already allocated and checked for size.  Could be called
// multiple times to populate different offsets in the buffer. Each new index
// is appended to the buffer.
void ArrayBase::toSparse(ArrayBase &a, size_t offset) const {
  UInt32 *p1;
  const UInt32 *p2;

  a.type_ = NTA_BasicType_Sparse; // to Array of type Sparse
  if (!has_buffer()) {
    a.setCount(0); // buffer is empty.
    return;
  }
  switch (type_) { // coming from Array of this type
  case NTA_BasicType_Byte:
    NonZeroT<Byte>(a, this, offset);
    break;
  case NTA_BasicType_Int16:
    NonZeroT<Int16>(a, this, offset);
    break;
  case NTA_BasicType_UInt16:
    NonZeroT<UInt16>(a, this, offset);
    break;
  case NTA_BasicType_Int32:
    NonZeroT<Int32>(a, this, offset);
    break;
  case NTA_BasicType_UInt32:
    NonZeroT<UInt32>(a, this, offset);
    break;
  case NTA_BasicType_Int64:
    NonZeroT<Int64>(a, this, offset);
    break;
  case NTA_BasicType_UInt64:
    NonZeroT<UInt64>(a, this, offset);
    break;
  case NTA_BasicType_Real32:
    NonZeroT<Real32>(a, this, offset);
    break;
  case NTA_BasicType_Real64:
    NonZeroT<Real64>(a, this, offset);
    break;
  case NTA_BasicType_Bool:
    NonZeroT<bool>(a, this, offset);
    break;
  case NTA_BasicType_SDR: {
    // from the sparse portion of the SDR
    const SDR *sdr = getSDR();
    const SDR_flatSparse_t &v = sdr->getFlatSparse();
    p1 = (UInt32 *)a.getBuffer();
    p2 = v.data();
    size_t j = a.getCount(); // current end of buffer.
    for (size_t i = 0u; i < v.size(); i++) {
      p1[j++] = (*p2++) + (UInt32)offset;
    }
    a.setCount(j); // New end of buffer
  } break;
  case NTA_BasicType_Sparse: {
    // already sparse, just append values with offset
    size_t j = a.getCount(); // current end of buffer.
    p1 = (UInt32 *)a.getBuffer();
    p2 = (UInt32 *)getBuffer();
    for (size_t i = 0u; i < count_; i++) {
      p1[j] = p2[i] + (UInt32)offset;
    }
    a.setCount(j); // New end of buffer
  } break;
  default:
    NTA_THROW << "Unexpected source array type.";
  }
}

template <typename T>
static void DenseT(ArrayBase &a, UInt32 *Fromptr, size_t count, size_t offset) {
  // Helper function for fromSparse( )
  // populate the new array with indexes of non-zero values.
  // NOTE: NTA_BasicType_SDR and NTA_BasicType_Sparse not handled here.
  size_t maxsize = a.getMaxElementsCount();
  T *newBuffer = (T *)a.getBuffer() + offset;
  for (size_t i = 0u; i < count; i++) {
    if (Fromptr[i] + offset < maxsize)
      newBuffer[Fromptr[i]] = (T)1;
  }
}

//  populate the given Array a with a flat dense version in a's type.
//  The offset is provided for handling the Fan-in case. It is where this
//  buffer's contribution should be inserted.
//  Destination buffer allready allocated and checked for size.  Could be called
//  multiple times to populate different offsets in the buffer.
void ArrayBase::fromSparse(ArrayBase &a, size_t offset) const {
  NTA_CHECK(type_ == NTA_BasicType_Sparse)
      << "This buffer does not contain a sparse type.";
  if (!has_buffer()) {
    a.zeroBuffer();
    return;
  }
  UInt32 *value = (UInt32 *)getBuffer();
  switch (a.getType()) { // to an Array of this type.
  case NTA_BasicType_Byte:
    DenseT<Byte>(a, value, count_, offset);
    break;
  case NTA_BasicType_Int16:
    DenseT<Int16>(a, value, count_, offset);
    break;
  case NTA_BasicType_UInt16:
    DenseT<UInt16>(a, value, count_, offset);
    break;
  case NTA_BasicType_Int32:
    DenseT<Int32>(a, value, count_, offset);
    break;
  case NTA_BasicType_UInt32:
    DenseT<UInt32>(a, value, count_, offset);
    break;
  case NTA_BasicType_Int64:
    DenseT<Int64>(a, value, count_, offset);
    break;
  case NTA_BasicType_UInt64:
    DenseT<UInt64>(a, value, count_, offset);
    break;
  case NTA_BasicType_Real32:
    DenseT<Real32>(a, value, count_, offset);
    break;
  case NTA_BasicType_Real64:
    DenseT<Real64>(a, value, count_, offset);
    break;
  case NTA_BasicType_Bool:
    DenseT<bool>(a, value, count_, offset);
    break;
  case NTA_BasicType_SDR: {
    UInt *fromBuf = (UInt *)getBuffer();
    SDR *sdr = a.getSDR();
    SDR_flatSparse_t &flatSparse = sdr->getFlatSparse();
    for (size_t i = 0; i < count_; i++)
      flatSparse.push_back(fromBuf[i] + (UInt)offset);
    sdr->setFlatSparse(sdr->getFlatSparse()); // reset cache
    // Note: before making this conversion, clear the destination SDR
    //       to avoid duplicates during a Fan-in.
  } break;
  case NTA_BasicType_Sparse: {
    UInt32 *newBuffer = (UInt32 *)a.getBuffer();
    UInt32 *originalBuffer = (UInt32 *)getBuffer();
    size_t j = a.getCount(); // end of buffer
    for (size_t i = 0u; i < count_; i++) {
      newBuffer[j++] = originalBuffer[i] + (UInt32)offset;
    }
    a.setCount(j);
    // Note: before making this conversion, clear the destination buffer
    //       to avoid duplicates during a Fan-in.
  } break;
  default:
    NTA_THROW << "Unexpected source array type.";
  }
}

///////////////////////////////////////////////////////////////////////////////
//    Compare operators
///////////////////////////////////////////////////////////////////////////////
// Compare contents of two ArrayBase objects
// Note: An Array and an ArrayRef could be the same if type, count, and buffer
// contents are the same.
bool operator==(ArrayBase &lhs, ArrayBase &rhs) {
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
  case NTA_BasicType_Sparse:
    _templatedStreamBuffer<UInt32>(outStream, inbuf, numElements);
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
