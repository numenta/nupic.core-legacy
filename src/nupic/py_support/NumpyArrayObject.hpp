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
}; // namespace nupic

#endif // NTA_NUMPY_ARRAY_OBJECT
