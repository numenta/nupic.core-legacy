/*
 * Copyright 2016 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef NUPIC_UTIL_MOVING_AVERAGE_HPP
#define NUPIC_UTIL_MOVING_AVERAGE_HPP

#include <vector>

#include <nupic/types/Types.hpp>

namespace nupic {

namespace util {

class MovingAverage {
public:
  MovingAverage(UInt wSize, const std::vector<Real32> &historicalValues);
  MovingAverage(UInt wSize);
  std::vector<Real32> getSlidingWindow() const;
  Real32 getCurrentAvg() const;
  Real32 compute(Real32 newValue);
  Real32 getTotal() const;
  bool operator==(const MovingAverage &r2) const;
  bool operator!=(const MovingAverage &r2) const;

private:
  UInt32 windowSize_;
  std::vector<Real32> slidingWindow_;
  Real32 total_;
};
} // namespace util
} // namespace nupic

#endif // NUPIC_UTIL_MOVING_AVERAGE_HPP
