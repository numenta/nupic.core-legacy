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

#ifndef NUPIC_UTIL_MOVING_AVERAGE_HPP
#define NUPIC_UTIL_MOVING_AVERAGE_HPP

#include <vector>

#include <nupic/types/Types.hpp>


namespace nupic {
  namespace util {

    class MovingAverage
    {
      private:
        std::vector<float>::size_type windowSize_;
        std::vector<float> slidingWindow_;
        float total_;
        float compute(float newVal);
      public:
        MovingAverage(UInt wSize, const std::vector<float>& historicalValues);
        MovingAverage(UInt wSize);
        std::vector<float> getSlidingWindow() const;
        float getCurrentAvg() const;
        void next(float newValue);
        float getTotal() const;
        bool operator==(const MovingAverage& r2) const;
        bool operator!=(const MovingAverage& r2) const;
    };
  }
}

#endif
