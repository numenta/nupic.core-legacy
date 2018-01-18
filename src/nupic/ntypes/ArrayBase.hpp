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

namespace nupic
{
  /**
   * An ArrayBase is used for passing arrays of data back and forth between
   * a client application and NuPIC, minimizing copying. It facilitates
   * both zero-copy and one-copy operations.
   */
  class ArrayBase
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
     * The destructor ensures the array doesn't leak its buffer (if
     * it owns it).
     */
    virtual ~ArrayBase();


    /**
     * Ask ArrayBase to allocate its buffer
     */
    void
    allocateBuffer(size_t count);

    void
    setBuffer(void *buffer, size_t count);

    void
    releaseBuffer();

    void*
    getBuffer() const;

    // number of elements of given type in the buffer
    size_t
    getCount() const;

    NTA_BasicType
    getType() const;

  protected:
    // buffer_ is typed so that we can use new/delete
    // cast to/from void* as necessary
    char* buffer_;
    size_t count_;
    NTA_BasicType type_;
    bool own_;

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
    static void _templatedStreamBuffer(std::ostream& outStream,
                                       const void* inbuf,
                                       size_t numElements)
    {
      outStream << "(";

      // Stream the elements
      auto it = (const SourceElementT*)inbuf;
      auto const end = it + numElements;
      if (it < end)
      {
        for (; it < end - 1; ++it)
        {
          outStream << *it << ", ";
        }

        outStream << *it;  // final element without the comma
      }

      outStream << ")";
    }

    friend std::ostream& operator<<(std::ostream&, const ArrayBase&);
  };

  // Serialization for diagnostic purposes
  std::ostream& operator<<(std::ostream&, const ArrayBase&);

}

#endif

