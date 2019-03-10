/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
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
 * ---------------------------------------------------------------------
 */

/** @file
 * Defines the base class for all encoders.
 */

#ifndef NTA_ENCODERS_BASE
#define NTA_ENCODERS_BASE

#include <nupic/types/Sdr.hpp>

namespace nupic {

/**
 * Description:
 * Base class for all encoders.
 *
 * Subclasses must implement method encode, and Serializable interface.
 * Subclasses can optionally implement method reset.
 */
template<typename DataType>
class BaseEncoder : public Serializable
{
public:
    /**
     * Members dimensions & size describe the shape of the encoded output SDR.
     * This is the total number of bits which the result has.
     */
    const vector<UInt> &dimensions = dimensions_;
    const UInt         &size       = size_;

    virtual void reset() {}

    virtual void encode(DataType input, SDR &output) = 0;

    virtual ~BaseEncoder() {}

protected:
    BaseEncoder() {}

    BaseEncoder(const vector<UInt> dimensions)
        { initialize( dimensions ); }

    virtual void initialize(const vector<UInt> dimensions) {
        dimensions_ = dimensions;
        size_       = SDR(dimensions).size;
    }

private:
    vector<UInt> dimensions_;
    UInt         size_;
};
} // end namespace nupic
#endif // NTA_ENCODERS_BASE
