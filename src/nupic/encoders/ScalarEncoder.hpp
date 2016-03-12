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
 * Define the ScalarEncoder and PeriodicScalarEncoder
 */

#ifndef NTA_ENCODERS_SCALAR
#define NTA_ENCODERS_SCALAR

#include <nupic/types/Types.hpp>

namespace nupic
{
  /**
   * @b Description
   * Base class for ScalarEncoders
   */
  class ScalarEncoderBase
  {
  public:
    virtual ~ScalarEncoderBase()
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

  /** Encodes a floating point number as a contiguous block of 1s.
   *
   * @b Description
   * A ScalarEncoder encodes a numeric (floating point) value into an array
   * of bits. The output is 0's except for a contiguous block of 1's. The
   * location of this contiguous block varies continuously with the input value.
   *
   * Conceptually, the set of possible outputs is a set of "buckets". If there
   * are m buckets, the ScalarEncoder distributes m points along the domain
   * [minValue, maxValue], including the endpoints. To figure out the bucket
   * index of an input, it rounds the input to the nearest of these points.
   *
   * This approach is different from the PeriodicScalarEncoder because two
   * buckets, the first and last, are half as wide as the rest, since fewer
   * numbers in the input domain will round to these endpoints. This behavior
   * makes sense because, for example, with the input space [1, 10] and 10
   * buckets, 1.49 is in the first bucket and 1.51 is in the second.
   */
  class ScalarEncoder : public ScalarEncoderBase
  {
  public:
    /**
     * Constructs a ScalarEncoder
     *
     * @param w The number of bits that are set to encode a single value -- the
     *   "width" of the output signal
     * @param minValue The minimum value of the input signal, inclusive.
     * @param maxValue The maximum value of the input signal, inclusive.
     * @param clipInput Whether to allow input values outside the [minValue, maxValue]
     *   range.  If true, the input will be clipped to minValue or maxValue.
     *
     * There are three mutually exclusive parameters that determine the overall
     * size of of the output. Only one of these should be nonzero:
     *
     * @param n The number of bits in the output. Must be greater than or equal to w.
     * @param radius Two inputs separated by more than the radius have
     *   non-overlapping representations. Two inputs separated by less than the
     *   radius will in general overlap in at least some of their bits. You can
     *   think of this as the radius of the input.
     * @param resolution Two inputs separated by greater than, or equal to the
     *   resolution are guaranteed to have different representations.
     */
    ScalarEncoder(int w, double minValue, double maxValue, int n, double radius,
                  double resolution, bool clipInput);
    ~ScalarEncoder() override;

    virtual int encodeIntoArray(Real64 input, Real32 output[]) override;
    virtual int getOutputWidth() const override { return n_; }

  private:
    int w_;
    int n_;
    double minValue_;
    double maxValue_;
    double bucketWidth_;
    bool clipInput_;
  }; // end class ScalarEncoder

  /** Encodes a floating point number as a block of 1s that might wrap around.
   *
   * @b Description
   * A PeriodicScalarEncoder encodes a numeric (floating point) value into an
   * array of bits. The output is 0's except for a contiguous block of 1's that
   * may wrap around the edge. The location of this contiguous block varies
   * continuously with the input value.
   *
   * Conceptually, the set of possible outputs is a set of "buckets". If there
   * are m buckets, the PeriodicScalarEncoder plots m equal-width bands along
   * the domain [minValue, maxValue]. The bucket index of an input is simply its
   * band index.
   *
   * Because of the equal-width buckets, the rounding differs from the
   * ScalarEncoder. In cases where the ScalarEncoder would put 1.49 in the first
   * bucket and 1.51 in the second, the PeriodicScalarEncoder will put 1.99 in
   * the first bucket and 2.0 in the second.
   */
  class PeriodicScalarEncoder : public ScalarEncoderBase
  {
  public:
    /**
     * Constructs a PeriodicScalarEncoder
     *
     * @param w The number of bits that are set to encode a single value -- the
     *   "width" of the output signal
     * @param minValue The minimum value of the input signal, inclusive.
     * @param maxValue The maximum value of the input signal, exclusive. All
     *   inputs will be strictly less than this value.
     *
     * There are three mutually exclusive parameters that determine the overall
     * size of the output. Only one of these should be nonzero:
     *
     * @param n The number of bits in the output. Must be greater than or equal
     *   to w.
     * @param radius Two inputs separated by more than the radius have
     *   non-overlapping representations. Two inputs separated by less than the
     *   radius will in general overlap in at least some of their bits. You can
     *   think of this as the radius of the input.
     * @param resolution Two inputs separated by greater than, or equal to the
     *   resolution are guaranteed to have different representations.
     */
    PeriodicScalarEncoder(int w, double minValue, double maxValue, int n,
                          double radius, double resolution);
    virtual ~PeriodicScalarEncoder() override;

    virtual int encodeIntoArray(Real64 input, Real32 output[]) override;
    virtual int getOutputWidth() const override { return n_; }

  private:
    int w_;
    int n_;
    double minValue_;
    double maxValue_;
    double bucketWidth_;
  }; // end class PeriodicScalarEncoder
} // end namespace nupic

#endif // NTA_ENCODERS_SCALAR
