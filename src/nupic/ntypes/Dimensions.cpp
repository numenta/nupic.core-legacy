/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/utils/Log.hpp>
#include <utility>

using namespace nupic;

Dimensions::Dimensions(){};

Dimensions::Dimensions(std::vector<size_t> v)
    : std::vector<size_t>(std::move(v)){};

Dimensions::Dimensions(size_t x) { push_back(x); }

Dimensions::Dimensions(size_t x, size_t y) {
  push_back(x);
  push_back(y);
}

Dimensions::Dimensions(size_t x, size_t y, size_t z) {
  push_back(x);
  push_back(y);
  push_back(z);
}

size_t Dimensions::getCount() const {
  if (isUnspecified() || isDontcare())
    NTA_THROW << "Attempt to get count from dimensions " << toString();
  size_t count = 1;
  for (size_t i = 0; i < size(); i++)
    count *= at(i);
  if (count == 0)
    NTA_THROW << "Attempt to get count from invalid dimensions " << toString();
  return count;
}

size_t Dimensions::getDimensionCount() const { return size(); }

size_t Dimensions::getDimension(size_t index) const {
  if (index >= size()) {
    NTA_THROW << "Bad request for dimension " << index << " on " << toString();
  }

  return at(index);
}

bool Dimensions::isDontcare() const { return (size() == 1 && at(0) == 0); }

bool Dimensions::isUnspecified() const { return size() == 0; }

bool Dimensions::isOnes() const {
  if (size() == 0)
    return false;

  for (size_t i = 0; i < size(); i++) {
    if (at(i) != 1)
      return false;
  }
  return true;
}

bool Dimensions::isValid() const {
  if (isDontcare() || isUnspecified())
    return true;

  for (size_t i = 0; i < size(); i++)
    if (at(i) == 0)
      return false;
  return true;
}

bool Dimensions::isSpecified() const {
  return isValid() && !isUnspecified() && !isDontcare();
}

// internal helper method
static std::string vecToString(std::vector<size_t> vec) {
  std::stringstream ss;
  for (size_t i = 0; i < vec.size(); i++) {
    ss << vec[i];
    if (i != vec.size() - 1)
      ss << " ";
  }
  return ss.str();
}

std::string Dimensions::toString(bool humanReadable) const {
  if (humanReadable) {
    if (isUnspecified())
      return "[unspecified]";
    if (isDontcare())
      return "[dontcare]";
  }

  std::string s = "[";
  s += vecToString(*this);
  s += "]";
  if (humanReadable && !isValid())
    s += " (invalid)";

  return s;
}

size_t Dimensions::getIndex(const Coordinate &coordinate) const {
  if (coordinate.size() != size()) {
    NTA_THROW << "Invalid coordinate [" << vecToString(coordinate)
              << "] for Dimensions " << toString();
  }

  size_t factor = 1;
  size_t index = 0;

  // We need to return an index based on x major ordering. We can't simply use
  // an unsigned or size_t because vector<>::size_type varies between
  // implementations (it is only required to be an unsigned type, not a
  // specific bit-depth).
  for (Coordinate::size_type dim = 0; dim != coordinate.size(); dim++) {
    size_t thisdim = at(dim);
    if (coordinate[dim] >= thisdim) {
      NTA_THROW << "Invalid coordinate index " << dim << " of "
                << coordinate[dim] << " is too large for region dimensions "
                << toString();
    }
    index += factor * coordinate[dim];
    factor *= thisdim;
  }
  return index;
}

Coordinate Dimensions::getCoordinate(const size_t index) const {
  Coordinate coordinate;
  size_t x = index;

  size_t product = 1;
  for (size_type i = 0; i < size(); i++) {
    product *= at(i);
  }

  for (size_type i = size() - 1; i != (size_type)-1; i--) {
    product /= at(i);
    coordinate.insert(coordinate.begin(), x / product);
    x %= product;
  }

  return coordinate;
}

void Dimensions::promote(size_t newDimensionality) {
  if (!isOnes()) {
    NTA_THROW << "Dimensions::promote -- must be all ones for Dimensions "
              << toString();
  }
  if (size() == newDimensionality)
    return;
  if (size() > newDimensionality)
    resize(newDimensionality);
  for (size_t i = size(); i < newDimensionality; i++)
    push_back(1);
}

bool Dimensions::operator==(const Dimensions &dims2) const {
  if ((std::vector<size_t>)(*this) == (std::vector<size_t>)dims2)
    return true;

  if (isOnes() && dims2.isOnes())
    return true;

  return false;
}

bool Dimensions::operator!=(const Dimensions &dims2) const {
  return !operator==(dims2);
}

namespace nupic {
std::ostream &operator<<(std::ostream &f, const Dimensions &d) {
  // temporary -- this might be hard to de-serialize
  f << d.toString(/* humanReadable: */ false);
  return f;
}
} // namespace nupic
