/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
 */



#include <Python.h>

// workaround for change in numpy config.h for python2.5 on windows
// Must come after python includes.
#ifndef SIZEOF_FLOAT
#define SIZEOF_FLOAT 32
#endif

#ifndef SIZEOF_DOUBLE
#define SIZEOF_DOUBLE 64
#endif

#include <nupic/py_support/NumpyVector.hpp>

#include <stdexcept>
#include <iostream>

using namespace std;
using namespace nupic;

// --------------------------------------------------------------
// Auto-convert a compile-time type to a Numpy dtype.
// --------------------------------------------------------------

template<typename T>
class NumpyDTypeTraits {};

template<typename T>
int LookupNumpyDTypeT(const T *)
  { return NumpyDTypeTraits<T>::numpyDType; }

#define NTA_DEF_NUMPY_DTYPE_TRAIT(a, b) \
template<> class NumpyDTypeTraits<a> { public: enum { numpyDType=b }; }; \
int nupic::LookupNumpyDType(const a *p) { return LookupNumpyDTypeT(p); }

NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::Byte, NPY_BYTE);
NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::Int16, NPY_INT16);
NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::UInt16, NPY_UINT16);

#if defined(NTA_ARCH_64) && (defined(NTA_OS_LINUX) || defined(NTA_OS_DARWIN) || defined(NTA_OS_SPARC))
NTA_DEF_NUMPY_DTYPE_TRAIT(size_t, NPY_UINT64);
#else
NTA_DEF_NUMPY_DTYPE_TRAIT(size_t, NPY_UINT32);
#endif

NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::Int32, NPY_INT32);

#if !(defined(NTA_ARCH_32) && defined(NTA_OS_LINUX))
// size_t (above) is the same as UInt32 on linux32
NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::UInt32, NPY_UINT32);
#endif

NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::Int64, NPY_INT64);

#if (!(defined(NTA_ARCH_64) && (defined(NTA_OS_LINUX) || defined(NTA_OS_DARWIN) || defined(NTA_OS_SPARC))) && !defined(NTA_OS_WINDOWS))
NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::UInt64, NPY_UINT64);
#endif


NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::Real32, NPY_FLOAT32);
NTA_DEF_NUMPY_DTYPE_TRAIT(nupic::Real64, NPY_FLOAT64);

// --------------------------------------------------------------

NumpyArray::NumpyArray(int nd, const int *ndims, int dtype)
  : p_(0), dtype_(dtype)
{
  // declare static to avoid new/delete with every call
  static npy_intp ndims_intp[NPY_MAXDIMS];

  if(nd < 0)
    throw runtime_error("Negative dimensioned arrays not supported.");

  if (nd > NPY_MAXDIMS)
    throw runtime_error("Too many dimensions specified for NumpyArray()");

  /* copy into array with elements that are the correct size.
   * npy_intp is an integer that can hold a pointer. On 64-bit
   * systems this is not the same as an int.
   */
  for (int i = 0; i < nd; i++)
  {
    ndims_intp[i] = (npy_intp)ndims[i];
  }

  p_ = (PyArrayObject *) PyArray_SimpleNew(nd, ndims_intp, dtype);

}

NumpyArray::NumpyArray(PyObject *p, int dtype, int requiredDimension)
  : p_(0), dtype_(dtype)
{
  PyObject *contiguous = PyArray_ContiguousFromObject(p, NPY_NOTYPE, 0, 0);
  if(!contiguous)
    throw std::runtime_error("Array could not be made contiguous.");
  if(!PyArray_Check(contiguous))
    throw std::logic_error("Failed to convert to array.");

  PyObject *casted = PyArray_Cast((PyArrayObject *) contiguous, dtype);
  Py_CLEAR(contiguous);

  if(!casted) throw std::runtime_error("Array could not be cast to requested type.");
  if(!PyArray_Check(casted)) throw std::logic_error("Array is not contiguous.");
  PyArrayObject *final = (PyArrayObject *) casted;
  if((requiredDimension != 0) && (PyArray_NDIM(final) != requiredDimension))
    throw std::runtime_error("Array is not of the required dimension.");
  p_ = final;
}

NumpyArray::~NumpyArray()
{
  PyObject *generic = (PyObject *) p_;
  p_ = 0;
  Py_CLEAR(generic);
}

int NumpyArray::getRank() const
{
  if(!p_) throw runtime_error("Null NumpyArray.");
  return PyArray_NDIM(p_);
}

int NumpyArray::dimension(int i) const
{
  if(!p_) throw runtime_error("Null NumpyArray.");
  if(i < 0) throw runtime_error("Negative dimension requested.");
  if(i >= PyArray_NDIM(p_)) throw out_of_range("Dimension exceeds number available.");
  return int(PyArray_DIMS(p_)[i]);
}

void NumpyArray::getDims(int *out) const
{
  if(!p_) throw runtime_error("Null NumpyArray.");
  int n = PyArray_NDIM(p_);
  for(int i=0; i<n; ++i) out[i] = int(PyArray_DIMS(p_)[i]); // npy_intp? New type in latest numpy headers.
}

const char *NumpyArray::addressOf0() const
{
  if(!p_) throw runtime_error("Null NumpyArray.");
  return (const char *)PyArray_DATA(p_);
}
char *NumpyArray::addressOf0()
{
  if(!p_) throw runtime_error("Numpy NumpyArray.");
  return (char *)PyArray_DATA(p_);
}

int NumpyArray::stride(int i) const
{
  if(!p_) throw runtime_error("Numpy NumpyArray.");
  return int(PyArray_STRIDES(p_)[i]); // npy_intp? New type in latest numpy headers.
}

PyObject *NumpyArray::forPython() {
  if(p_) {
    Py_XINCREF(p_);
    PyObject *toReturn = PyArray_Return((PyArrayObject *)p_);
    return toReturn;
  }
  else return 0;
}

