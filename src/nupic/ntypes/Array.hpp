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

// ---
//
// Definitions for the Array class
//
// It is a sub-class of ArrayBase that owns its buffer
//
// ---

#ifndef NTA_ARRAY_HPP
#define NTA_ARRAY_HPP

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
class Array : public ArrayBase {
public:
  Array(NTA_BasicType type, void *buffer, size_t count)
      : ArrayBase(type, buffer, count) {}

  explicit Array(NTA_BasicType type) : ArrayBase(type) {}

  // Array(const Array & other) : ArrayBase(other)
  //{
  //}

  void invariant() {
    if (!own_)
      NTA_THROW << "Array must own its buffer";
  }

private:
  // Hide base class method (invalid for Array)
  void setBuffer(void *buffer, size_t count);
};
} // namespace nupic

#endif
