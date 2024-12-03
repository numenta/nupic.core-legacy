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

#include <nupic/ntypes/Collection.hpp>
#include <nupic/utils/Log.hpp>
#include <string>
#include <vector>

namespace nupic {

/*
 * Implementation of the templated Collection class.
 * This code is used to create explicit instantiations
 * of the Collection class.
 * It is not compiled into the types library because
 * we instantiate for classes outside of the types library.
 * For example, Collection<OutputSpec> is built in the
 * net library where OutputSpec is defined.
 * See nupic/engine/Collections.cpp, which is where the
 * Collection classes are instantiated.
 */

template <typename T> Collection<T>::Collection() {}

template <typename T> Collection<T>::~Collection() {}
template <typename T>
bool Collection<T>::operator==(const Collection<T> &o) const {
  const static auto compare = [](std::pair<std::string, T> a,
                                 std::pair<std::string, T> b) {
    return a.first == b.first && a.second == b.second;
  };
  return std::equal(vec_.begin(), vec_.end(), o.vec_.begin(), compare);
}
template <typename T> size_t Collection<T>::getCount() const {
  return vec_.size();
}

template <typename T>
const std::pair<std::string, T> &Collection<T>::getByIndex(size_t index) const {
  NTA_CHECK(index < vec_.size());
  return vec_[index];
}

template <typename T>
std::pair<std::string, T> &Collection<T>::getByIndex(size_t index) {
  NTA_CHECK(index < vec_.size());
  return vec_[index];
}

template <typename T>
bool Collection<T>::contains(const std::string &name) const {
  typename CollectionStorage::const_iterator i;
  for (i = vec_.begin(); i != vec_.end(); i++) {
    if (i->first == name)
      return true;
  }
  return false;
}

template <typename T>
T Collection<T>::getByName(const std::string &name) const {
  typename CollectionStorage::const_iterator i;
  for (i = vec_.begin(); i != vec_.end(); i++) {
    if (i->first == name)
      return i->second;
  }
  NTA_THROW << "No item named: " << name;
}

template <typename T>
void Collection<T>::add(const std::string &name, const T &item) {
  // make sure we don't already have something with this name
  typename CollectionStorage::const_iterator i;
  for (i = vec_.begin(); i != vec_.end(); i++) {
    if (i->first == name) {
      NTA_THROW << "Unable to add item '" << name << "' to collection "
                << "because it already exists";
    }
  }

  // Add the new item to the vector
  vec_.push_back(std::make_pair(name, item));
}

template <typename T> void Collection<T>::remove(const std::string &name) {
  typename CollectionStorage::iterator i;
  for (i = vec_.begin(); i != vec_.end(); i++) {
    if (i->first == name)
      break;
  }
  if (i == vec_.end())
    NTA_THROW << "No item named '" << name << "' in collection";

  vec_.erase(i);
}

} // namespace nupic
