/*
 * Copyright 2017 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * The home of the NTA_NumpyArray_API symbol
 */

#define NTA_OVERRIDE_NO_IMPORT_ARRAY
#include <nupic/py_support/NumpyArrayObject.hpp>

#include <stdexcept>

namespace nupic {
void initializeNumpy() {
  // Use _import_array() because import_array() is a macro that contains a
  // return statement.
  if (_import_array() != 0) {
    throw std::runtime_error(
        "initializeNumpy: numpy.core.multiarray failed to import.");
  }
}
} // namespace nupic
