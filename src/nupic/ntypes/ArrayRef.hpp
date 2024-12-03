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
// Definitions for the ArrayRef class
//
// It is a sub-class of ArrayBase that doesn't own its buffer
//
// ---

#ifndef NTA_ARRAY_REF_HPP
#define NTA_ARRAY_REF_HPP

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
class ArrayRef : public ArrayBase {
public:
  ArrayRef(NTA_BasicType type, void *buffer, size_t count) : ArrayBase(type) {
    setBuffer(buffer, count);
  }

  explicit ArrayRef(NTA_BasicType type) : ArrayBase(type) {}

  ArrayRef(const ArrayRef &other) : ArrayBase(other) {}

  void invariant() {
    if (own_)
      NTA_THROW << "ArrayRef mmust not own its buffer";
  }

private:
  // Hide base class method (invalid for ArrayRef)
  void allocateBuffer(void *buffer, size_t count);
};
} // namespace nupic

#endif
