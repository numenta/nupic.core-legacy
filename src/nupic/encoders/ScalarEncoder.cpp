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
 * Implementations of the ScalarEncoder and PeriodicScalarEncoder
 */

#include <algorithm> //std::fill
#include <cmath>

#include <nupic/encoders/ScalarEncoder.hpp>

namespace nupic {

std::vector<UInt> ScalarEncoderBase::encode(Real input) {
    std::vector<UInt> output(getOutputWidth());
    encodeIntoArray(input, output.data());
    return output;
}


ScalarEncoder::ScalarEncoder(int w, double minValue, double maxValue, int n,
                             double radius, double resolution, bool clipInput)
    : ScalarEncoderBase(w,n), minValue_(minValue), maxValue_(maxValue), clipInput_(clipInput) {
  NTA_CHECK(!( (n && radius) || (n && resolution) || (resolution && radius) ))
	  << "Only one of n/radius/resolution can be specified for a "
                 "ScalarEncoder.";

  const double extentWidth = maxValue - minValue;
  NTA_CHECK(extentWidth > 0) 
     << "minValue must be < maxValue. minValue=" << minValue
     << " maxValue=" << maxValue;

  if (n != 0) {
    n_ = n;
    // Distribute nBuckets points along the domain [minValue, maxValue],
    // including the endpoints. The resolution is the width of each band
    // between the points.
    const int nBuckets = n - (w - 1);
    const int nBands = nBuckets - 1;
    bucketWidth_ = extentWidth / nBands;
  } else {
    bucketWidth_ = resolution || radius / w;
    NTA_CHECK(bucketWidth_ > 0) << "One of n/radius/resolution must be nonzero.";
    const int neededBands = ceil(extentWidth / bucketWidth_);
    const int neededBuckets = neededBands + 1;
    n_ = neededBuckets + (w - 1);
  }
  NTA_CHECK(bucketWidth_ > 0);
  NTA_CHECK(n_ > 0);
  NTA_CHECK(w_ < n_);
}


int ScalarEncoder::encodeIntoArray(Real input, UInt output[]) {
  if(clipInput_) {
    input = input < minValue_ ? minValue_ : input;
    input = input > maxValue_ ? maxValue_ : input;
  }

  NTA_CHECK(input >= minValue_ && input <= maxValue_) << "Input must be within [minValue, maxValue]";

  const int iBucket = round((input - minValue_) / bucketWidth_);
  const int firstBit = iBucket;

  std::fill(&output[0], &output[n_ -1], 0);
  std::fill_n(&output[firstBit], w_, 1);
  return iBucket;
}


PeriodicScalarEncoder::PeriodicScalarEncoder(int w, double minValue,
                                             double maxValue, int n,
                                             double radius, double resolution)
    : ScalarEncoder(w, minValue, maxValue, n, radius, resolution, false) {

  if (n != 0) {
    // Distribute nBuckets equal-width bands within the domain [minValue,
    // maxValue]. The resolution is the width of each band.
    const int nBuckets = n;
    const double extentWidth = maxValue - minValue;
    bucketWidth_ = extentWidth / nBuckets;
  } else {
    const int neededBuckets = ceil((maxValue - minValue) / bucketWidth_);
    n_ = (neededBuckets > w_) ? neededBuckets : w_ + 1;
  }

  NTA_CHECK(bucketWidth_ > 0);
  NTA_CHECK(n_ > 0);
  NTA_CHECK(w_ < n_);
}


int PeriodicScalarEncoder::encodeIntoArray(Real input, UInt output[]) {
  NTA_CHECK(input >= minValue_ && input < maxValue_) 
    << "input " << input << " not within range [" << minValue_ << ", " << maxValue_ << ")";

  const int iBucket = (int)((input - minValue_) / bucketWidth_);
  const int middleBit = iBucket;
  const double reach = (w_ - 1) / 2.0;
  const int left = floor(reach);
  const int right = ceil(reach);

  std::fill(&output[0], &output[n_ -1], 0);
  output[middleBit] = 1;
  for (int i = 1; i <= left; i++) {
    const int index = middleBit - i;
    output[(index < 0) ? index + n_ : index] = 1;
  }
  for (int i = 1; i <= right; i++) {
    output[(middleBit + i) % n_] = 1;
  }

  return iBucket;
}
} // end namespace nupic
