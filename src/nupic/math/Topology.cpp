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
 * ----------------------------------------------------------------------
 */

/** @file
 * Topology helpers
 */

#include <nupic/math/Topology.hpp>
#include <nupic/utils/Log.hpp>

using std::vector;
using namespace nupic;
using namespace nupic::math::topology;

namespace nupic {
namespace math {
namespace topology {

vector<UInt> coordinatesFromIndex(UInt index, const vector<UInt> &dimensions) {
  vector<UInt> coordinates(dimensions.size(), 0);

  UInt shifted = index;
  for (size_t i = dimensions.size() - 1; i > 0; i--) {
    coordinates[i] = shifted % dimensions[i];
    shifted = shifted / dimensions[i];
  }

  NTA_ASSERT(shifted < dimensions[0]);
  coordinates[0] = shifted;

  return coordinates;
}

UInt indexFromCoordinates(const vector<UInt> &coordinates,
                          const vector<UInt> &dimensions) {
  NTA_ASSERT(coordinates.size() == dimensions.size());

  UInt index = 0;
  for (size_t i = 0; i < dimensions.size(); i++) {
    NTA_ASSERT(coordinates[i] < dimensions[i]);
    index *= dimensions[i];
    index += coordinates[i];
  }

  return index;
}

} // end namespace topology
} // namespace math
} // end namespace nupic

// ============================================================================
// NEIGHBORHOOD
// ============================================================================

Neighborhood::Neighborhood(UInt centerIndex, UInt radius,
                           const vector<UInt> &dimensions)
    : centerPosition_(coordinatesFromIndex(centerIndex, dimensions)),
      dimensions_(dimensions), radius_(radius) {}

Neighborhood::Iterator::Iterator(const Neighborhood &neighborhood, bool end)
    : neighborhood_(neighborhood),
      offset_(neighborhood.dimensions_.size(), -neighborhood.radius_),
      finished_(end) {
  // Choose the first offset that has positive resulting coordinates.
  for (size_t i = 0; i < offset_.size(); i++) {
    offset_[i] = std::max(offset_[i], -(Int)neighborhood_.centerPosition_[i]);
  }
}

bool Neighborhood::Iterator::operator!=(const Iterator &other) const {
  return finished_ != other.finished_;
}

UInt Neighborhood::Iterator::operator*() const {
  UInt index = 0;
  for (size_t i = 0; i < neighborhood_.dimensions_.size(); i++) {
    const Int coordinate = neighborhood_.centerPosition_[i] + offset_[i];

    NTA_ASSERT(coordinate >= 0);
    NTA_ASSERT(coordinate < (Int)neighborhood_.dimensions_[i]);

    index *= neighborhood_.dimensions_[i];
    index += coordinate;
  }

  return index;
}

const Neighborhood::Iterator &Neighborhood::Iterator::operator++() {
  advance_();
  return *this;
}

void Neighborhood::Iterator::advance_() {
  // When it overflows, we need to "carry the 1" to the next dimension.
  bool overflowed = true;

  for (Int i = offset_.size() - 1; i >= 0; i--) {
    offset_[i]++;

    overflowed = offset_[i] > (Int)neighborhood_.radius_ ||
                 (((Int)neighborhood_.centerPosition_[i] + offset_[i]) >=
                  (Int)neighborhood_.dimensions_[i]);

    if (overflowed) {
      // Choose the first offset that has a positive resulting coordinate.
      offset_[i] = std::max(-(Int)neighborhood_.radius_,
                            -(Int)neighborhood_.centerPosition_[i]);
    } else {
      // There's no overflow. The remaining coordinates don't need to change.
      break;
    }
  }

  // When the final coordinate overflows, we're done.
  if (overflowed) {
    finished_ = true;
  }
}

Neighborhood::Iterator Neighborhood::begin() const { return {*this, false}; }

Neighborhood::Iterator Neighborhood::end() const { return {*this, true}; }

// ============================================================================
// WRAPPING NEIGHBORHOOD
// ============================================================================

WrappingNeighborhood::WrappingNeighborhood(UInt centerIndex, UInt radius,
                                           const vector<UInt> &dimensions)
    : centerPosition_(coordinatesFromIndex(centerIndex, dimensions)),
      dimensions_(dimensions), radius_(radius) {}

WrappingNeighborhood::Iterator::Iterator(
    const WrappingNeighborhood &neighborhood, bool end)
    : neighborhood_(neighborhood),
      offset_(neighborhood.dimensions_.size(), -neighborhood.radius_),
      finished_(end) {}

bool WrappingNeighborhood::Iterator::operator!=(const Iterator &other) const {
  return finished_ != other.finished_;
}

UInt WrappingNeighborhood::Iterator::operator*() const {
  UInt index = 0;
  for (size_t i = 0; i < neighborhood_.dimensions_.size(); i++) {
    Int coordinate = neighborhood_.centerPosition_[i] + offset_[i];

    // With a large radius, it may have wrapped around multiple times, so use
    // `while`, not `if`.

    while (coordinate < 0) {
      coordinate += neighborhood_.dimensions_[i];
    }

    while (coordinate >= (Int)neighborhood_.dimensions_[i]) {
      coordinate -= neighborhood_.dimensions_[i];
    }

    index *= neighborhood_.dimensions_[i];
    index += coordinate;
  }

  return index;
}

const WrappingNeighborhood::Iterator &WrappingNeighborhood::Iterator::
operator++() {
  advance_();
  return *this;
}

void WrappingNeighborhood::Iterator::advance_() {
  // When it overflows, we need to "carry the 1" to the next dimension.
  bool overflowed = true;

  for (Int i = offset_.size() - 1; i >= 0; i--) {
    offset_[i]++;

    // If the offset has moved by more than the dimension size, i.e. if
    // offset_[i] - (-radius) is greater than the dimension size, then we're
    // about to run into points that we've already seen. This happens when given
    // small dimensions, a large radius, and wrap-around.
    overflowed = offset_[i] > (Int)neighborhood_.radius_ ||
                 offset_[i] + (Int)neighborhood_.radius_ >=
                     (Int)neighborhood_.dimensions_[i];

    if (overflowed) {
      offset_[i] = -neighborhood_.radius_;
    } else {
      // There's no overflow. The remaining coordinates don't need to change.
      break;
    }
  }

  // When the final coordinate overflows, we're done.
  if (overflowed) {
    finished_ = true;
  }
}

WrappingNeighborhood::Iterator WrappingNeighborhood::begin() const {
  return {*this, /*end*/ false};
}

WrappingNeighborhood::Iterator WrappingNeighborhood::end() const {
  return {*this, /*end*/ true};
}
