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
 * Defines the abstract Encoder base class
 */

#ifndef NTA_ENCODERS_BASE
#define NTA_ENCODERS_BASE

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/Types.hpp>

namespace nupic
{
  /** An encoder converts a value to a sparse distributed representation.
   *
   * @b Description
   * This is the base class for encoders.
   *
   * Methods that must be implemented by subclasses:
   * - getWidth() - returns the output width, in bits
   * - encodeIntoArray() - encodes input and puts the encoded value into the
   *   output, which is an array of length returned by getWidth()
   */
  class Encoder
  {
  public:
    virtual ~Encoder()
    {}

    virtual void encodeIntoArray(const ArrayBase & input, UInt output[],
                                 bool learn) = 0;
    virtual int getWidth() const = 0;
  };
} // end namespace nupic

#endif // NTA_ENCODERS_BASE
