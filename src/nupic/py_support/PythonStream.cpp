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

#include <nupic/py_support/PythonStream.hpp>
#include <nupic/utils/Log.hpp>

/**
 * Bumps up size to a nicely aligned larger size.
 * Taken for NuPIC2 from PythonUtils.hpp
 */
static size_t NextPythonSize(size_t n) {
  n += 1;
  n += 8 - (n % 8);
  return n;
}

// -------------------------------------------------------------
SharedPythonOStream::SharedPythonOStream(size_t maxSize)
    : target_size_(NextPythonSize(maxSize)), ss_(std::ios_base::out) {}

// -------------------------------------------------------------
std::ostream &SharedPythonOStream::getStream() { return ss_; }

// -------------------------------------------------------------
PyObject *SharedPythonOStream::close() {
  ss_.flush();

  if (ss_.str().length() > target_size_)
    throw std::runtime_error("Stream output larger than allocated buffer.");

  return PyString_FromStringAndSize(ss_.str().c_str(), ss_.str().length());
}
