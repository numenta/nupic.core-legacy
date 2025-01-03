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

#ifndef NTA_NODESET_HPP
#define NTA_NODESET_HPP

#include "nupic/utils/Log.hpp"

#include <set>

namespace nupic {
/**
 * A NodeSet represents the set of currently-enabled nodes in a Region
 * It is just a set of indexes, with the ability to add/remove an index, and the
 * ability to iterate through enabled nodes.
 *
 * There are many ways to represent such a set, and the best way to represent
 * it depends on what nodes are typically enabled through this mechanism.
 * In NuPIC 1 we used an IndexRangeList, which is a list of index ranges.
 * This is natural, for example, in the original pictures app, where in
 * training level N+1 we would enable a square patch of nodes at level N.
 * (which is a list of ranges). In the NuPIC 1 API such ranges were initially
 * specified with ranges ("1-4"). With new algorithms and new training paradigms
 * I think we may always enable nodes individually.
 *
 * So for NuPIC 2 we're starting with the simplest possible solution (a set) and
 * might switch to something else (e.g. a range list) if needed.
 *
 * TODO: split into hpp/cpp
 */
class NodeSet {
public:
  NodeSet(size_t nnodes) : nnodes_(nnodes) { set_.clear(); }

  typedef std::set<size_t>::const_iterator const_iterator;

  const_iterator begin() const { return set_.begin(); };

  const_iterator end() const { return set_.end(); }

  void allOn() {
    for (size_t i = 0; i < nnodes_; i++) {
      set_.insert(i);
    }
  }

  void allOff() { set_.clear(); }

  void add(size_t index) {
    if (index > nnodes_) {
      NTA_THROW << "Attempt to enable node with index " << index
                << " which is larger than the number of nodes " << nnodes_;
    }
    set_.insert(index);
  }

  void remove(size_t index) {
    auto f = set_.find(index);
    if (f == set_.end())
      return;
    set_.erase(f);
  }

private:
  typedef std::set<size_t>::iterator iterator;
  NodeSet();
  size_t nnodes_;
  std::set<size_t> set_;
};

} // namespace nupic

#endif // NTA_NODESET_HPP
