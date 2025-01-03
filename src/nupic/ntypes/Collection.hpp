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

#ifndef NTA_COLLECTION_HPP
#define NTA_COLLECTION_HPP

#include <string>
#include <vector>

namespace nupic {
// A collection is a templated class that contains items of type t.
// It supports lookup by name and by index. The items are stored in a map
// and copies are also stored in a vector (it's Ok to use pointers).
// You can add items using the add() method.
//
template <typename T> class Collection {
public:
  Collection();
  virtual ~Collection();
  bool operator==(const Collection<T> &other) const;
  inline bool operator!=(const Collection<T> &other) const {
    return !operator==(other);
  }
  size_t getCount() const;

  // This method provides access by index to the contents of the collection
  // The indices are in insertion order.
  //

  const std::pair<std::string, T> &getByIndex(size_t index) const;

  bool contains(const std::string &name) const;

  T getByName(const std::string &name) const;

  // TODO: move add/remove to a ModifiableCollection subclass
  // This method should be internal but is currently tested
  // in net_test.py in test_node_spec
  void add(const std::string &name, const T &item);

  void remove(const std::string &name);

#ifdef NTA_INTERNAL
  std::pair<std::string, T> &getByIndex(size_t index);
#endif

private:
  typedef std::vector<std::pair<std::string, T>> CollectionStorage;
  CollectionStorage vec_;
};
} // namespace nupic

#endif
