/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.
 *               2019, David McDougall
 *
 * Unless you have an agreement with Numenta, Inc., for a separate license for
 * this software code, the following terms and conditions apply:
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
 * Define the ScalarEncoder
 */

#ifndef NTA_ENCODERS_SCALAR
#define NTA_ENCODERS_SCALAR

#include <nupic/types/Types.hpp>
#include <nupic/encoders/BaseEncoder.hpp>

namespace nupic {

  /**
   * TODO, description
   * TODO, example usage
   */
  struct ScalarEncoderParameters
  {
    /**
     * Members "minimum" and "maximum" define the range of the input signal.
     * These endpoints are inclusive.
     */
    double  minimum;
    double  maximum;

    /**
     * Member "clipInput" determines whether to allow input values outside the
     * range [minimum, maximum].
     * If true, the input will be clipped into the range [minimum, maximum].
     * If false, inputs outside of the range will raise an error.
     */
    bool clipInput;

    /**
     * TODO
     */
    bool periodic;

    /**
     * Member "active" is the number of true bits in the encoded output SDR.
     * The output encodings will have a contiguous block of this many 1's.
     */
    UInt active;
    /**
     * Member "sparsity" is an alternative way to specify the member "active".
     * Sparsity requires that the size to also be specified.
     * Specify only one: active or sparsity.
     */
    Real sparsity;

    /**
     * These three (3) members define the total number of bits in the output:
     *      size,
     *      radius,
     *      resolution.
     *
     * These are mutually exclusive and only one of them should be non-zero when
     * constructing the encoder.
     */

    /**
     * Member "size" is the total number of bits in the encoded output SDR.
     */
    UInt size;

    /**
     * Member "radius" Two inputs separated by more than the radius have
     * non-overlapping representations. Two inputs separated by less than the
     * radius will in general overlap in at least some of their bits. You can
     * think of this as the radius of the input.
     */
    double  radius;

    /**
     * Member "resolution" Two inputs separated by greater than, or equal to the
     * resolution are guaranteed to have different representations.
     */
    double  resolution;
  };

  /**
   * Encodes a real number as a contiguous block of 1s.
   *
   * Description:
   * The ScalarEncoder encodes a numeric (floating point) value into an array
   * of bits. The output is 0's except for a contiguous block of 1's. The
   * location of this contiguous block varies continuously with the input value.
   */
  class ScalarEncoder : public BaseEncoder<double>
  {
  public:
    ScalarEncoder( ScalarEncoderParameters &parameters );
    void initialize( ScalarEncoderParameters &parameters );

    const ScalarEncoderParameters &parameters = args_;

    void encode(double input, SDR &output) override;

    ~ScalarEncoder() override {};

    // TODO: save & load override
    // TODO: REmember to zero the conflicting stuff out of the parameters so
    // that it can be laoded via constructor.

  private:
    ScalarEncoderParameters args_;
  };   // end class ScalarEncoder
}      // end namespace nupic
#endif // end NTA_ENCODERS_SCALAR
