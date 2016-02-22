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

#include <math.h>
#include <nupic/encoders/Scalar.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic
{
  static Real64 extractScalar(const ArrayBase & input) {
    Real64 scalar;
    switch (input.getType()) {
    case NTA_BasicType_Real64:
      scalar = ((Real64*) input.getBuffer())[0];
      break;
    default:
      NTA_THROW << "ScalarEncoder: Unsupported type " << input.getType();
    }

    return scalar;
  }

  ScalarEncoder::ScalarEncoder(int w, double minval, double maxval, int n,
                               double radius, double resolution, bool clipInput)
    :w_(w),
     minval_(minval),
     maxval_(maxval),
     clipInput_(clipInput)
  {
    if ((n != 0 && (radius != 0 || resolution != 0)) ||
        (radius != 0 && (n != 0 || resolution != 0)) ||
        (resolution != 0 && (n != 0 || radius != 0))) {
      NTA_THROW << "Only one of n/radius/resolution can be specified for a ScalarEncoder.";
    }

    if (n != 0) {
      n_ = n;
      const int nBuckets = n - (w - 1);
      // Distribute nBuckets points along the domain [minval, maxval], including
      // the endpoints. The resolution is the distance between points.
      resolution_ = (maxval - minval) / (nBuckets - 1);
    }
    else {
      if (resolution != 0) {
        resolution_ = resolution;
      }
      else if (radius != 0) {
        resolution_ = radius / w;
      }
      else {
        NTA_THROW << "One of n/radius/resolution must be nonzero.";
      }

      const int neededBuckets = ceil((maxval - minval) / resolution_) + 1;
      n_ = neededBuckets + (w - 1);
    }
  }

  ScalarEncoder::~ScalarEncoder()
  {
  }

  void ScalarEncoder::encodeIntoArray(const ArrayBase & input, UInt output[],
                                      bool learn)
  {
    Real64 scalar = extractScalar(input);

    if (scalar < minval_) {
      if (clipInput_) {
        scalar = minval_;
      }
      else {
        NTA_THROW << "input (" << scalar << ") less than range [" << minval_ <<
          ", " << maxval_ << "]";
      }
    }
    else if (scalar > maxval_) {
      if (clipInput_) {
        scalar = maxval_;
      }
      else {
        NTA_THROW << "input (" << scalar << ") greater than range [" << minval_ <<
          ", " << maxval_ << "]";
      }
    }

    const int iBucket = round((scalar - minval_) / resolution_);

    const int firstBit = iBucket;

    memset(output, 0, n_*sizeof(output[0]));
    for (int i = 0; i < w_; i++) {
      output[firstBit + i] = 1;
    }
  }

  PeriodicScalarEncoder::PeriodicScalarEncoder(int w, double minval, double maxval,
                                               int n, double radius, double resolution)
    :w_(w),
     minval_(minval),
     maxval_(maxval)
  {
    if ((n != 0 && (radius != 0 || resolution != 0)) ||
        (radius != 0 && (n != 0 || resolution != 0)) ||
        (resolution != 0 && (n != 0 || radius != 0))) {
      NTA_THROW << "Only one of n/radius/resolution can be specified for a ScalarEncoder.";
    }

    if (n != 0) {
      n_ = n;
      const int nBuckets = n;
      // Distribute nBuckets equal-width bands within the domain [minval, maxval].
      // The resolution is the width of each band.
      resolution_ = (maxval - minval) / nBuckets;
    }
    else {
      if (resolution != 0) {
        resolution_ = resolution;
      }
      else if (radius != 0) {
        resolution_ = radius / w;
      }
      else {
        NTA_THROW << "One of n/radius/resolution must be nonzero.";
      }

      const int neededBuckets = ceil((maxval - minval) / resolution_);
      n_ = neededBuckets;
    }
  }

  PeriodicScalarEncoder::~PeriodicScalarEncoder()
  {
  }

  void PeriodicScalarEncoder::encodeIntoArray(const ArrayBase & input, UInt output[],
                                              bool learn)
  {
    Real64 scalar = extractScalar(input);

    if (scalar < minval_ || scalar >= maxval_) {
      NTA_THROW << "input " << scalar << " not within range [" << minval_ <<
        ", " << maxval_ << ")";
    }

    const int iBucket = (int)((scalar - minval_) / resolution_);

    const int middleBit = iBucket;
    const double reach = (w_ - 1) / 2.0;
    const int left = floor(reach);
    const int right = ceil(reach);

    memset(output, 0, n_*sizeof(output[0]));
    output[middleBit] = 1;
    for (int i = 1; i <= left; i++) {
      const int index = middleBit - i;
      output[(index < 0) ? index + n_ : index] = 1;
    }
    for (int i = 1; i <= right; i++) {
      output[(middleBit + i) % n_] = 1;
    }
  }
}
