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
#include <numeric>

#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/SlidingWindow.hpp>

namespace nupic {
namespace util {

class MovingAverage : public Serializable {
public:
  MovingAverage(UInt wSize, const std::vector<Real> &historicalValues);

  MovingAverage(UInt wSize);

  inline std::vector<Real> getData() const {
    return slidingWindow_.getData(); }

  inline Real getCurrentAvg() const {
    return Real(total_) / Real(slidingWindow_.size()); }

  Real compute(Real newValue);

  inline Real getTotal() const { return total_; }

  inline bool operator==(const MovingAverage& r2) const {
    return (slidingWindow_ == r2.slidingWindow_ &&
          total_ == r2.total_);
  }

  inline bool operator!=(const MovingAverage &r2) const {
    return !operator==(r2);
  }

  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const {
    size_t wSize = slidingWindow_.size();
    ar(CEREAL_NVP(wSize));
    ar(CEREAL_NVP(slidingWindow_));
  }
  template<class Archive>
  void load_ar(Archive & ar) {
    ar(CEREAL_NVP(slidingWindow_));
    const std::vector<Real>&  window = slidingWindow_.getData();
    total_ = Real(std::accumulate(begin(window), end(window), 0.0f));
  }
  // The Class does not have a default constructor so we have to
  // tell Cereal to construct it with an argument if it is used
  // in a smart pointer.
  template <class Archive>
  static void load_and_construct( Archive & ar, cereal::construct<MovingAverage> & construct )
  {
    UInt wSize;
    ar( wSize );  // reads it from the stream
    construct( wSize );
  }


private:
  SlidingWindow<Real> slidingWindow_;
  Real total_;
};
} // namespace util
} // namespace nupic

#endif // NUPIC_UTIL_MOVING_AVERAGE_HPP
