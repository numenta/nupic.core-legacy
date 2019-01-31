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
 * Definitions for the ArrayBase class
  *
  * An ArrayBase object contains a memory buffer that is used for
  * implementing zero-copy and one-copy operations in NuPIC.
  * An ArrayBase contains:
  * - a pointer to a buffer (held in a shared_ptr)
  * - a length
  * - a capacity   (useful if buffer is larger than data in buffer)
  * - a type
  * - a flag indicating whether or not the object owns the buffer.
  * Note: if buffer is not owned, shared_ptr will not delete it.
  */

#ifndef NTA_ARRAY_BASE_HPP
#define NTA_ARRAY_BASE_HPP

#include <iostream> // for ostream, istream
#include <stdlib.h> // for size_t
#include <string>
#include <memory>	// for shared_ptr

#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>





namespace nupic
{
  /**
   * An ArrayBase is used for passing arrays of data back and forth between
   * a client application and NuPIC, minimizing copying. It facilitates
   * both zero-copy and one-copy operations.
   */
  class ArrayBase : public Serializable
  {
  public:
    /**
     * Caller provides a buffer to use.
     * NuPIC always copies data into this buffer
     * Caller frees buffer when no longer needed.
     */
    ArrayBase(NTA_BasicType type, void* buffer, size_t count);

    /**
     * Caller does not provide a buffer --
     * Nupic will either provide a buffer via setBuffer or
     * ask the ArrayBase to allocate a buffer via allocateBuffer.
     */
    explicit ArrayBase(NTA_BasicType type);

    /**
     * Copy constructor.
     */
    ArrayBase(const ArrayBase& other) {
      type_ = other.type_;
      buffer_ = other.buffer_;
      count_ = other.count_;
      capacity_ = other.capacity_;
      own_ = other.own_;
    }

    /**
     * The destructor ensures the array doesn't leak its buffer (if
     * it owns it).
     */
    virtual ~ArrayBase();


    /**
         * Ask ArrayBase to allocate its buffer
         */
    virtual void
    allocateBuffer(size_t count);

    /**
         * Ask ArrayBase to zero fill its buffer
        */
    virtual void
    zeroBuffer();


    virtual void
    setBuffer(void *buffer, size_t count);

    virtual void
    releaseBuffer();

    void*
    getBuffer() const;

    // number of elements of given type in the buffer
    size_t
    getCount() const;

    // max number of elements this buffer can hold (capacity)
	  size_t getMaxElementsCount() const;

	  // Returns the allocated buffer size in bytes independent of array length
    size_t getBufferSize() const;


    void setCount(size_t count);

    NTA_BasicType
    getType() const;

    bool
    isInstance(const ArrayBase &a);


    /**
    * serialization and deserialization for an Array
    */
    // binary representation
    void save(std::ostream &outStream) const override;
    void load(std::istream &inStream) override;

    // ascii text representation
    //    [ type count ( item item item ...) ... ]
    friend std::ostream &operator<<(std::ostream &outStream,  const ArrayBase &a);
    friend std::istream &operator>>(std::istream &inStream, ArrayBase &a);

  protected:
    // buffer_ is typed so that we can use new/delete
    // cast to/from void* as necessary
    std::shared_ptr<char> buffer_;
    size_t count_;      // number of elements in the buffer
    size_t capacity_;   // size of the allocated buffer in bytes
    NTA_BasicType type_;// type of data in this buffer
    bool own_;
    void convertInto(ArrayBase &a, size_t offset=0) const;

    // Used by the Array class to return an NZ array from local array.
    // Template defines the type of the local array.
    void NonZero(ArrayBase& a) const;

    template <typename T>
    void NonZeroT(ArrayBase &a) const;


  private:

  };

  // If this class does NOT own the buffer we instantiate the shared_ptr
  // with a version that uses this class as the deleter.  This results
  // in the buffer not being deleted when the last instance of this class
  // is deleted. The Caller is responsible for deleting the buffer.
  struct nonDeleter {
    void operator()(char *p) const {
    }
  };
  ///////////////////////////////////////////////////////////
  // for stream serialization on an Array
  //    [ type count ( item item item ) ]
  // for inStream the Array object must already exist and initialized with a type.
  // The buffer will be allocated and populated with this class as owner.
  std::ostream &operator<<(std::ostream &outStream, const ArrayBase &a);
  std::istream &operator>>(std::istream &inStream, ArrayBase &a);

  // Compare contents of two ArrayBase objects
  // Note: An Array and an ArrayRef could be the same if type, count, and buffer
  // contents are the same.
  bool operator==(const ArrayBase &lhs, const ArrayBase &rhs);
  inline bool operator!=(const ArrayBase &lhs, const ArrayBase &rhs) {return !(lhs == rhs);}

} // namespace


#endif

