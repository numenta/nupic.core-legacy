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
        int window_size;
        std::vector<float> sliding_window;
        float total;
        bool operator==(MovingAverage& r2);
      public:
        MovingAverage(int w_size, const std::vector<float>& historical_values);
        MovingAverage(int w_size);
        static std::tuple<float, float> compute(std::vector<float>& sliding_window, 
                                                float total, float newVal,
                                                unsigned int windowSize);
        std::vector<float> get_sliding_window();
        float get_current_avg();
        float next(float newValue);
        float get_total();
    };
  }
}

#endif
