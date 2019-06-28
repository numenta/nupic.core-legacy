/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

#ifndef HTM_UTIL_MOVING_AVERAGE_HPP
#define HTM_UTIL_MOVING_AVERAGE_HPP

#include <vector>
#include <numeric>

#include <htm/types/Serializable.hpp>
#include <htm/types/Types.hpp>
#include <htm/utils/SlidingWindow.hpp>

namespace htm {

class MovingAverage : public Serializable {
public:
  MovingAverage(UInt wSize, const std::vector<Real> &historicalValues);

  MovingAverage(UInt wSize);

  inline std::vector<Real> getData() const {
    return slidingWindow_.getData(); }

  Real getCurrentAvg() const; 

  Real compute(Real newValue);

  inline Real getTotal() const { return total_; }

  inline bool operator==(const MovingAverage& r2) const {
    return (slidingWindow_ == r2.slidingWindow_ &&
          total_ == r2.total_);
  }

  inline bool operator!=(const MovingAverage &r2) const {
    return !operator==(r2);
  }

  CerealAdapter;  // see Serializable.hpp
  template<class Archive>
  void save_ar(Archive & ar) const {
    size_t wSize = slidingWindow_.size();
    ar(CEREAL_NVP(wSize));          // save size of sliding window to stream
    ar(CEREAL_NVP(slidingWindow_)); // save data in sliding window to stream
  }
  template<class Archive>
  void load_ar(Archive & ar) {
    size_t wSize;
    ar(CEREAL_NVP(wSize));          // load size of sliding window to stream, not used
    ar(CEREAL_NVP(slidingWindow_)); // load data in sliding window to stream

    // calculates total_
    const std::vector<Real>&  window = slidingWindow_.getData();
    total_ = Real(std::accumulate(begin(window), end(window), 0.0f));
  }

  friend class cereal::access;

  // The MovingAverage class does not have a default constructor so we have to
  // tell Cereal to construct it with an argument if it is used
  // in a smart pointer.  Called by Cereal when loading unique_ptr<MovingAverage>.
  template <class Archive>
  static void load_and_construct( Archive & ar, cereal::construct<MovingAverage>& construct )
  {
    UInt wSize;
    ar( wSize );                    // reads size of slidingWindow from the stream
    construct(wSize);               // allocates slidingWindow
    ar(construct->slidingWindow_);  // populates sliding window

    // calculates total_
    const std::vector<Real>&  window = construct->slidingWindow_.getData();
    construct->total_ = Real(std::accumulate(begin(window), end(window), 0.0f));
  }

private:
  SlidingWindow<Real> slidingWindow_;
  Real total_ = 0.0f;
};
} // namespace htm

#endif // HTM_UTIL_MOVING_AVERAGE_HPP
