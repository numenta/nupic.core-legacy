/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

/** @file
 * The home of the NTA_NumpyArray_API symbol
 */

#define NTA_OVERRIDE_NO_IMPORT_ARRAY
#include <nupic/py_support/NumpyArrayObject.hpp>

#include <stdexcept>

namespace nupic {
  void initializeNumpy()
  {
    // Use _import_array() because import_array() is a macro that contains a
    // return statement.
    if (_import_array() != 0)
    {
      throw std::runtime_error(
        "initializeNumpy: numpy.core.multiarray failed to import.");
    }
  }
}
