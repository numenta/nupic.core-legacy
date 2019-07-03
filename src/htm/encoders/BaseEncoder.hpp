/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
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
 * --------------------------------------------------------------------- */

/** @file
 * Defines the base class for all encoders.
 */

#ifndef NTA_ENCODERS_BASE
#define NTA_ENCODERS_BASE

#include <htm/types/Sdr.hpp>

namespace htm {

/**
 * Base class for all encoders.
 * An encoder converts a value to a sparse distributed representation.
 *
 * Subclasses must implement method encode and Serializable interface.
 * Subclasses can optionally implement method reset.
 *
 * There are several critical properties which all encoders must have:
 *
 * 1) Semantic similarity:  Similar inputs should have high overlap.  Overlap
 * decreases smoothly as inputs become less similar.  Dissimilar inputs have
 * very low overlap so that the output representations are not easily confused.
 *
 * 2) Stability:  The representation for an input does not change during the
 * lifetime of the encoder.
 *
 * 3) Sparsity: The output SDR should have a similar sparsity for all inputs and
 * have enough active bits to handle noise and subsampling.
 *
 * Reference: https://arxiv.org/pdf/1602.05925.pdf
 */
template<typename DataType>
class BaseEncoder : public Serializable
{
public:
    /**
     * Members dimensions & size describe the shape of the encoded output SDR.
     * This is the total number of bits in the result.
     */
    const std::vector<UInt> &dimensions = dimensions_;
    const UInt              &size       = size_;

    virtual void reset() {}

    virtual void encode(DataType input, SDR &output) = 0;

    virtual ~BaseEncoder() {}

protected:
    BaseEncoder() {}

    BaseEncoder(const std::vector<UInt> dimensions)
        { initialize( dimensions ); }

    void initialize(const std::vector<UInt> dimensions) {
        dimensions_ = dimensions;
        size_       = SDR(dimensions).size;
    }

private:
    std::vector<UInt> dimensions_;
    UInt              size_;
};
} // end namespace htm
#endif // NTA_ENCODERS_BASE
