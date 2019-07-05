/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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

#ifndef NTA_COLLECTION_HPP
#define NTA_COLLECTION_HPP

#include <string>
#include <vector>
#include <map>
#include <htm/utils/Log.hpp>

namespace htm {
// A collection is a templated class that contains items of type t.
// The items are stored in a vector and keys are also stored in a map.
// It supports lookup by name and by index. O(nlogn)
// Iteration is either by consecutive indexes or by iterator. O(1)
// You can add items using the add() method. O(nlogn)
// You can delete itmes using the remove() method. O(n)
//
// The collections are expected to be fairly static and small.
// The deletions are rare so we can affort O(n) for delete.
//
// This has been re-implemnted as an inline header-only class.
// The map holds the key (always a string) and an index into vector.
// The vector holds a std::pair<key, Object>


template <class T>
class Collection {
public:
  Collection() {}
  virtual ~Collection() {}

  typedef typename std::vector<std::pair<std::string, T>>::iterator Iterator;
  typedef typename std::vector<std::pair<std::string, T>>::const_iterator ConstIterator;

  inline bool operator==(const Collection<T> &other) const {
      const static auto compare = [](std::pair<std::string, T> a,
                                     std::pair<std::string, T> b) {
          return a.first == b.first && a.second == b.second;
      };
    return std::equal(vec_.begin(), vec_.end(), other.vec_.begin(), compare);
  }
  inline bool operator!=(const Collection<T> &other) const {
    return !operator==(other);
  }
  inline size_t getCount() const { return vec_.size(); }
  inline size_t size() const { return vec_.size(); }

  // This method provides access by index to the contents of the collection
  // The indices are in insertion order.
  //
  inline const std::pair<std::string, T> &getByIndex(size_t index) const {
  	NTA_CHECK(index < vec_.size()) << "Collection index out-of-range.";
	return vec_[index];
  }
  inline std::pair<std::string, T> &getByIndex(size_t index) {
  	NTA_CHECK(index < vec_.size()) << "Collection index out-of-range.";
	return vec_[index];
  }

  inline bool contains(const std::string &name) const {
    return (map_.find(name) != map_.end());
  }

  inline T getByName(const std::string &name) const {
     auto itr = map_.find(name);
	 NTA_CHECK(itr != map_.end()) << "No item named: " << name;
	 return vec_[itr->second].second;
  }

  inline Iterator begin() {
  	return vec_.begin();
  }
  inline Iterator end() {
  	return vec_.end();
  }
  inline ConstIterator cbegin() const {
  	return vec_.begin();
  }
  inline ConstIterator cend() const {
  	return vec_.end();
  }

  inline void add(const std::string &name, const T &item) {
    NTA_CHECK(!contains(name)) << "Unable to add item '" << name << "' to collection "
                << "because it already exists";
    // Add the new item to the vector
    vec_.push_back(std::make_pair(name, item));
	// Add the new item to the map
	size_t idx = vec_.size() - 1;
	map_[name] = idx;
  }

  void remove(const std::string &name) {
    auto itr = map_.find(name);
    NTA_CHECK(itr != map_.end()) << "No item named '" << name << "' in collection";
	size_t idx = itr->second;
	map_.erase(itr);
	vec_.erase(vec_.begin() + idx);
	// reset the indexes in the map
	for (size_t i = idx; i < vec_.size(); i++) {
		itr = map_.find(vec_[i].first);
		itr->second = i;
	 }
  }


private:
  std::vector<std::pair<std::string, T>> vec_;
  std::map<std::string, size_t> map_;

};

} // namespace htm

#endif
