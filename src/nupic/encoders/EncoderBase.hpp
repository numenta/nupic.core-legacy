/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
 * Defines the abstract Encoder base classes.
 * An encoder converts a value to an array of bits.
 */

#ifndef NTA_ENCODERS_BASE
#define NTA_ENCODERS_BASE

#include <nupic/types/Types.hpp>

namespace nupic
{
  /**
   * @b Description
   * Base class for encoders that encode a single floating point number.
   */
  class FloatEncoder
  {
  public:
    virtual ~FloatEncoder()
      {}

    /**
     * Encodes input, puts the encoded value into output, and returns the a
     * bucket number for the encoding.
     *
     * The bucket number is essentially the input encoded into an integer rather
     * than an array. A bucket number is easier to "decode" or to use inside a
     * classifier.
     *
     * @param input The value to encode
     * @param output Should have length of at least getOutputWidth()
     */
    virtual int encodeIntoArray(Real64 input, Real32 output[]) = 0;

    /**
     * Returns the output width, in bits.
     */
    virtual int getOutputWidth() const = 0;
  };
} // end namespace nupic

#endif // NTA_ENCODERS_BASE
