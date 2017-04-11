/* Numenta Platform for Intelligent Computing (NuPIC)
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


#ifndef NUPIC_UTIL_SLIDING_WINDOW_HPP
#define NUPIC_UTIL_SLIDING_WINDOW_HPP

#include <vector>

#include <nupic/types/Types.hpp>


namespace nupic {
  namespace util {

    class SlidingWindow {
      public:
        SlidingWindow(UInt maxCapacity);
        SlidingWindow(UInt maxCapacity, std::vector<Real> initialData);
        const UInt maxCapacity;
        const Real NEUTRAL_VALUE = 0.0;
        UInt size() const;
        /** append new value to the end of the buffer and handle the "overflows"-may pop the first element if full. 
         :return addition(+) neutral value (that is 0) when size()< maxCapacity; when full - return the value of the dropped element  
        */
        Real append(Real newValue);
        std::vector<Real> getData() const;

        bool operator==(const SlidingWindow& r2) const;
        bool operator!=(const SlidingWindow& r2) const;

      private:
        std::vector<Real> buffer;
}; 
}} //end ns
#endif //header
