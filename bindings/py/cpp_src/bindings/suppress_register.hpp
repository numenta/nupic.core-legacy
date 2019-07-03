/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, Numenta, Inc.
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
 * Author: @chhenning, 2018
 * --------------------------------------------------------------------- */

/** @file
Suppress warnings about register keyword.
*/

// the use of 'register' keyword is removed in C++17
// Python2.7 uses 'register' in unicodeobject.h
// pybind11.h includes python.h which includes unicodeobject.h

#ifdef _WIN32
#pragma warning( disable : 5033)  // MSVC
#else
  #if __cplusplus >= 201703L  // C++17 or higher
    #pragma GCC diagnostic ignored "-Wregister"  // for GCC and CLang
  #endif
#endif
#include <Python.h>