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
 * Wrapper for <numpy/arrayobject.h>.
 *
 * Normally when you include <numpy/arrayobject.h>, it creates a static
 * "PyArray_API" variable for each cpp file. Each of these variables must be
 * initialized via "import_array()".
 *
 * Include this file to use a shared PyArray_API variable across multiple cpp
 * files. Run "initializeNumpy()" to initialize this shared variable.
 */

#ifndef NTA_NUMPY_ARRAY_OBJECT

#ifndef NTA_OVERRIDE_NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NTA_NumpyArray_API
#include <numpy/arrayobject.h>
#undef NO_IMPORT_ARRAY

namespace nupic {
  /**
   * This method needs to be called some time in the process lifetime before
   * calling the numpy C APIs.
   *
   * It initializes the global "NTA_NumpyArray_API" to an array of function
   * pointers -- i.e. it dynamically links with the numpy library.
   */
  void initializeNumpy();
};

#endif // NTA_NUMPY_ARRAY_OBJECT
