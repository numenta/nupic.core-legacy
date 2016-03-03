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

#ifndef UTIL_HPP
#define UTIL_HPP

#include <tuple>
#include <vector>


namespace nupic {
  namespace util {

    class MovingAverage
    {
      private:
        int windowSize_;
        std::vector<float> slidingWindow_;
        float total_;
        bool operator==(MovingAverage& r2);
      public:
        MovingAverage(int wSize, const std::vector<float>& historicalValues);
        MovingAverage(int wSize);
        static std::tuple<float, float> compute(std::vector<float>& slidingWindow, 
                                                float total, float newVal,
                                                unsigned int windowSize);
        std::vector<float> getSlidingWindow();
        float getCurrentAvg();
        float next(float newValue);
        float getTotal();
    };
  }
}

#endif
