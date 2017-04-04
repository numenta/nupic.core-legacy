/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
 * Array utilities
 */

#ifndef NTA_ARRAY_UTILS_HPP
#define NTA_ARRAY_UTILS_HPP

#include <iostream> // for ostream
//#include <string>
#include <stdlib.h> // for size_t

#include <nupic/ntypes/Array.hpp>
#include <nupic/types/Types.hpp>


namespace nupic
{

  class ArrayUtils
  {
  public:
    /**
     * Stream a buffer of NTA_BasicType for debugging.
     *
     * NOTE: presently, only numeric element types are supported.
     *
     * @param outStream   output stream
     * @param inbuf       input buffer
     * @param numElements number of elements to use from the beginning of buffer
     * @param elementType type of the elements
     *
     * @returns The given outStream reference
     */
    static std::ostream& streamBufferOfBasicType(std::ostream& outStream,
                                                 const void* inbuf,
                                                 size_t numElements,
                                                 NTA_BasicType elementType)
    {
      switch (elementType)
      {
      case NTA_BasicType_Byte:
        _templatedStreamBuffer<NTA_Byte>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Int16:
        _templatedStreamBuffer<NTA_Int16>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_UInt16:
        _templatedStreamBuffer<NTA_UInt16>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Int32:
        _templatedStreamBuffer<NTA_Int32>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_UInt32:
        _templatedStreamBuffer<NTA_UInt32>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Int64:
        _templatedStreamBuffer<NTA_Int64>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_UInt64:
        _templatedStreamBuffer<NTA_UInt64>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Real32:
        _templatedStreamBuffer<NTA_Real32>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Real64:
        _templatedStreamBuffer<NTA_Real64>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Handle:
        _templatedStreamBuffer<NTA_Handle>(outStream, inbuf, numElements);
        break;
      case NTA_BasicType_Bool:
        _templatedStreamBuffer<bool>(outStream, inbuf, numElements);
        break;

      default:
        NTA_THROW << "Unexpected Element Type: " << elementType;
        break;
      }

      return outStream;
    }

  private:
    /**
     * Element-type-specific templated function for streaming elements to
     * ostream.
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

      auto it = (const SourceElementT*)inbuf;
      auto const end = it + numElements;
      for (; it < end - 1; ++it)
      {
        outStream << *it << ", ";
      }

      outStream << *it << ")";  // final element without the comma
    }

  }; // class ArrayUtils

} // namespace nupic


#endif // NTA_ARRAY_UTILS_HPP
