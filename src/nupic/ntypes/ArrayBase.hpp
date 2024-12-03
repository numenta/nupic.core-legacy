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
 * Definitions for the ArrayBase class
 *
 * An ArrayBase object contains a memory buffer that is used for
 * implementing zero-copy and one-copy operations in NuPIC.
 * An ArrayBase contains:
 * - a pointer to a buffer
 * - a length
 * - a type
 * - a flag indicating whether or not the object owns the buffer.
 */

#ifndef NTA_ARRAY_BASE_HPP
#define NTA_ARRAY_BASE_HPP

#include <iostream> // for ostream
#include <stdlib.h> // for size_t
#include <string>

#include <nupic/types/Types.h>

namespace nupic {
/**
 * An ArrayBase is used for passing arrays of data back and forth between
 * a client application and NuPIC, minimizing copying. It facilitates
 * both zero-copy and one-copy operations. The array can be of variable length
 * independent of buffer size. Array length cannot exceed buffer size.
 */
class ArrayBase {
public:
  /**
   * Caller provides a buffer to use.
   * NuPIC always copies data into this buffer
   * Caller frees buffer when no longer needed.
   */
  ArrayBase(NTA_BasicType type, void *buffer, size_t count);

  /**
   * Caller does not provide a buffer --
   * Nupic will either provide a buffer via setBuffer or
   * ask the ArrayBase to allocate a buffer via allocateBuffer.
   */
  explicit ArrayBase(NTA_BasicType type);

  /**
   * The destructor ensures the array doesn't leak its buffer (if
   * it owns it).
   */
  virtual ~ArrayBase();

  /**
   * Ask ArrayBase to allocate its buffer
   */
  void allocateBuffer(size_t count);

  void setBuffer(void *buffer, size_t count);

  void releaseBuffer();

  void *getBuffer() const;

  // number of elements of given type in the buffer
  size_t getCount() const;

  // max number of elements this buffer can hold
  size_t getMaxElementsCount() const;

  /**
   * Returns the allocated buffer size in bytes independent of array length
   */
  size_t getBufferSize() const;

  /**
   * Set array length independent of buffer size. Array length cannot exceed
   * buffer size
   */
  void setCount(size_t count);

  NTA_BasicType getType() const;

protected:
  // buffer_ is typed so that we can use new/delete
  // cast to/from void* as necessary
  char *buffer_;
  size_t count_;
  NTA_BasicType type_;
  bool own_;
  size_t bufferSize_;

private:
  /**
   * Element-type-specific templated function for streaming elements to
   * ostream. Elements are comma+space-separated and enclosed in braces.
   *
   * @param outStream   output stream
   * @param inbuf       input buffer
   * @param numElements number of elements to use from the beginning of buffer
   */
  template <typename SourceElementT>
  static void _templatedStreamBuffer(std::ostream &outStream, const void *inbuf,
                                     size_t numElements) {
    outStream << "(";

    // Stream the elements
    auto it = (const SourceElementT *)inbuf;
    auto const end = it + numElements;
    if (it < end) {
      for (; it < end - 1; ++it) {
        outStream << *it << ", ";
      }

      outStream << *it; // final element without the comma
    }

    outStream << ")";
  }

  friend std::ostream &operator<<(std::ostream &, const ArrayBase &);
};

// Serialization for diagnostic purposes
std::ostream &operator<<(std::ostream &, const ArrayBase &);

} // namespace nupic

#endif
