#ifndef NTA_PYTHON_STREAM_HPP
#define NTA_PYTHON_STREAM_HPP

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

#include <iosfwd>
#include <nupic/py_support/PyHelpers.hpp>
#include <sstream>

///////////////////////////////////////////////////////////////////
/// Provides a stream that outputs a PyString on class close()
///
/// @b Responsibility
/// Must make a PyString object that contains the same string as
/// was passed to the ostream returned by getStream()
///
/// @b Description
/// After instantiation, a call to getStream() returns an ostream
/// that collects the characters fed to it. Any subsequent call
/// to close() will return a PyObject * to a PyString that
/// contains the current contents of the ostream.
///
/// @note
/// A close() before a getStream() will return an empty PyString.
///
///////////////////////////////////////////////////////////////////
class SharedPythonOStream {
public:
  SharedPythonOStream(size_t maxSize);
  std::ostream &getStream();
  PyObject *close();

private:
  size_t target_size_;
  std::stringstream ss_;
};

//------------------------------------------------------------------

#endif // NTA_PYTHON_STREAM_HPP
