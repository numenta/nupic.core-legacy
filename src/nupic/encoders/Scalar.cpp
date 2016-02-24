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

#include <cstring> // memset
#include <cmath>
#include <nupic/encoders/Scalar.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic
{
  static Real64 extractScalar(const ArrayBase & input)
  {
    switch (input.getType())
    {
    case NTA_BasicType_Byte:
      return ((Byte*) input.getBuffer())[0];
      break;
    case NTA_BasicType_Int16:
      return ((Int16*) input.getBuffer())[0];
      break;
    case NTA_BasicType_UInt16:
      return ((UInt16*) input.getBuffer())[0];
      break;
    case NTA_BasicType_Int32:
      return ((Int32*) input.getBuffer())[0];
      break;
    case NTA_BasicType_UInt32:
      return ((UInt32*) input.getBuffer())[0];
      break;
    case NTA_BasicType_Int64:
      return ((Int64*) input.getBuffer())[0];
      break;
    case NTA_BasicType_UInt64:
      return ((UInt64*) input.getBuffer())[0];
      break;
    case NTA_BasicType_Real32:
      return ((Real32*) input.getBuffer())[0];
      break;
    case NTA_BasicType_Real64:
      return ((Real64*) input.getBuffer())[0];
      break;
    default:
      NTA_THROW << "extractScalar: Unsupported type " << input.getType();
    }
  }

  ScalarEncoder::ScalarEncoder(
    int w, double minValue, double maxValue, int n, double radius,
    double resolution, bool clipInput)
    :w_(w),
     minValue_(minValue),
     maxValue_(maxValue),
     clipInput_(clipInput)
  {
    if ((n != 0 && (radius != 0 || resolution != 0)) ||
        (radius != 0 && (n != 0 || resolution != 0)) ||
        (resolution != 0 && (n != 0 || radius != 0)))
    {
      NTA_THROW <<
        "Only one of n/radius/resolution can be specified for a ScalarEncoder.";
    }

    const int extentWidth = maxValue - minValue;
    if (extentWidth <= 0)
    {
      NTA_THROW << "minValue must be < maxValue. minValue=" << minValue <<
        " maxValue=" << maxValue;
    }

    if (n != 0)
    {
      n_ = n;

      if (w_ < 1 || w_ >= n_)
      {
        NTA_THROW << "w must be within the range [1, n). w=" << w_ << " n=" << n_;
      }

      // Distribute nBuckets points along the domain [minValue, maxValue],
      // including the endpoints. The resolution is the width of each band
      // between the points.
      const int nBuckets = n - (w - 1);
      const int nBands = nBuckets - 1;
      bucketWidth_ = extentWidth / nBands;
    }
    else
    {
      bucketWidth_ = resolution || radius / w;
      if (bucketWidth_ == 0)
      {
        NTA_THROW << "One of n/radius/resolution must be nonzero.";
      }

      const int neededBands = ceil(extentWidth / bucketWidth_);
      const int neededBuckets =  neededBands + 1;
      n_ = neededBuckets + (w - 1);
    }
  }

  ScalarEncoder::~ScalarEncoder()
  {
  }

  void ScalarEncoder::encodeIntoArray(
    const ArrayBase & input, UInt output[], bool learn)
  {
    Real64 scalar = extractScalar(input);

    if (scalar < minValue_)
    {
      if (clipInput_)
      {
        scalar = minValue_;
      }
      else
      {
        NTA_THROW << "input (" << scalar << ") less than range [" << minValue_ <<
          ", " << maxValue_ << "]";
      }
    } else if (scalar > maxValue_) {
      if (clipInput_)
      {
        scalar = maxValue_;
      }
      else
      {
        NTA_THROW << "input (" << scalar << ") greater than range [" << minValue_ <<
          ", " << maxValue_ << "]";
      }
    }

    const int iBucket = round((scalar - minValue_) / bucketWidth_);

    const int firstBit = iBucket;

    memset(output, 0, n_*sizeof(output[0]));
    for (int i = 0; i < w_; i++)
    {
      output[firstBit + i] = 1;
    }
  }

  PeriodicScalarEncoder::PeriodicScalarEncoder(
    int w, double minValue, double maxValue, int n, double radius, double resolution)
    :w_(w),
     minValue_(minValue),
     maxValue_(maxValue)
  {
    if ((n != 0 && (radius != 0 || resolution != 0)) ||
        (radius != 0 && (n != 0 || resolution != 0)) ||
        (resolution != 0 && (n != 0 || radius != 0)))
    {
      NTA_THROW <<
        "Only one of n/radius/resolution can be specified for a ScalarEncoder.";
    }

    const int extentWidth = maxValue - minValue;
    if (extentWidth <= 0)
    {
      NTA_THROW << "minValue must be < maxValue. minValue=" << minValue <<
        " maxValue=" << maxValue;
    }

    if (n != 0)
    {
      n_ = n;

      if (w_ < 1 || w_ >= n_)
      {
        NTA_THROW << "w must be within the range [1, n). w=" << w_ << " n=" << n_;
      }

      // Distribute nBuckets equal-width bands within the domain [minValue, maxValue].
      // The resolution is the width of each band.
      const int nBuckets = n;
      bucketWidth_ = extentWidth / nBuckets;
    }
    else
    {
      bucketWidth_ = resolution || radius / w;
      if (bucketWidth_ == 0)
      {
        NTA_THROW << "One of n/radius/resolution must be nonzero.";
      }

      const int neededBuckets = ceil((maxValue - minValue) / bucketWidth_);
      n_ = (neededBuckets > w_) ? neededBuckets : w_ + 1;
    }
  }

  PeriodicScalarEncoder::~PeriodicScalarEncoder()
  {
  }

  void PeriodicScalarEncoder::encodeIntoArray(
    const ArrayBase & input, UInt output[], bool learn)
  {
    Real64 scalar = extractScalar(input);

    if (scalar < minValue_ || scalar >= maxValue_)
    {
      NTA_THROW << "input " << scalar << " not within range [" << minValue_ <<
        ", " << maxValue_ << ")";
    }

    const int iBucket = (int)((scalar - minValue_) / bucketWidth_);

    const int middleBit = iBucket;
    const double reach = (w_ - 1) / 2.0;
    const int left = floor(reach);
    const int right = ceil(reach);

    memset(output, 0, n_*sizeof(output[0]));
    output[middleBit] = 1;
    for (int i = 1; i <= left; i++)
    {
      const int index = middleBit - i;
      output[(index < 0) ? index + n_ : index] = 1;
    }
    for (int i = 1; i <= right; i++)
    {
      output[(middleBit + i) % n_] = 1;
    }
  }
} // end namespace nupic
