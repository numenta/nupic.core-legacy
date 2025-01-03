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

// The Python.h #include MUST always be #included first in every
// compilation unit (.c or .cpp file). That means that PyHelpers.hpp
// must be #included first and transitively every .hpp file that
// #includes directly or indirectly PyHelpers.hpp must be #included
// first.
#include <Python.h>

#include "PyArray.hpp"
#include <nupic/py_support/NumpyArrayObject.hpp>
#include <nupic/utils/Log.hpp>

#include <iostream>
#include <sstream>
#include <string>

namespace nupic {
// -------------------------------------
//
//  G E T   B A S I C   T Y P E
//
// -------------------------------------

NTA_BasicType getBasicType(NTA_Byte) { return NTA_BasicType_Byte; }
NTA_BasicType getBasicType(NTA_Int16) { return NTA_BasicType_Int16; }
NTA_BasicType getBasicType(NTA_UInt16) { return NTA_BasicType_UInt16; }
NTA_BasicType getBasicType(NTA_Int32) { return NTA_BasicType_Int32; }
NTA_BasicType getBasicType(NTA_UInt32) { return NTA_BasicType_UInt32; }
NTA_BasicType getBasicType(NTA_Int64) { return NTA_BasicType_Int64; }
NTA_BasicType getBasicType(NTA_UInt64) { return NTA_BasicType_UInt64; }
NTA_BasicType getBasicType(NTA_Real32) { return NTA_BasicType_Real32; }
NTA_BasicType getBasicType(NTA_Real64) { return NTA_BasicType_Real64; }
NTA_BasicType getBasicType(bool) { return NTA_BasicType_Bool; }

// -------------------------------------
//
//  A R R A Y    2   N U M P Y
//
// -------------------------------------
// Wrap an Array object with a numpy array PyObject
PyObject *array2numpy(const ArrayBase &a) {
  npy_intp dims[1];
  dims[0] = npy_intp(a.getCount());

  NTA_BasicType t = a.getType();
  int dtype;
  switch (t) {
  case NTA_BasicType_Byte:
    dtype = NPY_INT8;
    break;
  case NTA_BasicType_Int16:
    dtype = NPY_INT16;
    break;
  case NTA_BasicType_UInt16:
    dtype = NPY_UINT16;
    break;
  case NTA_BasicType_Int32:
    dtype = NPY_INT32;
    break;
  case NTA_BasicType_UInt32:
    dtype = NPY_UINT32;
    break;
  case NTA_BasicType_Int64:
    dtype = NPY_INT64;
    break;
  case NTA_BasicType_UInt64:
    dtype = NPY_UINT64;
    break;
  case NTA_BasicType_Real32:
    dtype = NPY_FLOAT32;
    break;
  case NTA_BasicType_Real64:
    dtype = NPY_FLOAT64;
    break;
  case NTA_BasicType_Bool:
    dtype = NPY_BOOL;
    break;
  default:
    NTA_THROW << "Unknown basic type: " << t;
  };

  return (PyObject *)PyArray_SimpleNewFromData(1, dims, dtype, a.getBuffer());
}

//// -------------------------------------
////
////  P Y   A R R A Y   B A S E
////
//// -------------------------------------
//  template <typename T, typename A>
//  PyArrayBase<T, A>::PyArrayBase() : A(getType())
//  {
//  }
//
//  //template <typename T, typename A>
//  //PyArrayBase<T, A>::PyArrayBase(ArrayBase * a) : A(*(static_cast<A *>(a)))
//  //{
//  //}
//
//  //template <typename T, typename A>
//  //PyArrayBase<T, A>::PyArrayBase(A * a) : A(*a)
//  //{
//  //}
//
//  template <typename T, typename A>
//  NTA_BasicType PyArrayBase<T, A>::getType()
//  {
//    T t = 0;
//    return getBasicType(t);
//  }
//
//  template <typename T, typename A>
//  T PyArrayBase<T, A>::__getitem__(int i) const
//  {
//    return ((T *)(A::getBuffer()))[i];
//  }
//
//  template <typename T, typename A>
//  void PyArrayBase<T, A>::__setitem__(int i, T x)
//  {
//    ((T *)(A::getBuffer()))[i] = x;
//  }
//
//  template <typename T, typename A>
//  size_t PyArrayBase<T, A>::__len__() const
//  {
//    return A::getCount();
//  }
//
//  template <typename T, typename A>
//  std::string PyArrayBase<T, A>::__repr__() const
//  {
//    std::stringstream ss;
//    ss << "[ ";
//    for (size_t i = 0; i < __len__(); ++i)
//      ss << __getitem__(i) << " ";
//    ss << "]";
//    return ss.str();
//  }
//
//  template <typename T, typename A>
//  std::string PyArrayBase<T, A>::__str__() const
//  {
//    return __repr__();
//  }
//
//  template <typename T, typename A>
//  PyObject * PyArrayBase<T, A>::asNumpyArray() const
//  {
//    return array2numpy(*this);
//  }

// -------------------------------------
//
//  P Y   A R R A Y
//
// -------------------------------------
template <typename T>
PyArray<T>::PyArray()
    : Array(getType())
// PyArray<T>::PyArray() : PyArrayBase<T, Array>()
{}

template <typename T>
// PyArray<T>::PyArray(size_t count) : PyArrayBase<T, Array>()
PyArray<T>::PyArray(size_t count) : Array(getType()) {
  allocateBuffer(count);
}

template <typename T> NTA_BasicType PyArray<T>::getType() {
  T t = 0;
  return getBasicType(t);
}

template <typename T> T PyArray<T>::__getitem__(int i) const {
  return ((T *)(getBuffer()))[i];
  // return PyArrayBase<T, Array>::__getitem__(i);
}

template <typename T> void PyArray<T>::__setitem__(int i, T x) {
  ((T *)(getBuffer()))[i] = x;
  // PyArrayBase<T, Array>::__setitem__(i, x);
}

template <typename T> size_t PyArray<T>::__len__() const {
  return getCount();
  // return PyArrayBase<T, Array>::__len__();
}

template <typename T> std::string PyArray<T>::__repr__() const {
  std::stringstream ss;
  ss << "[ ";
  for (size_t i = 0; i < __len__(); ++i)
    ss << __getitem__(i) << " ";
  ss << "]";
  return ss.str();

  // return PyArrayBase<T, Array>::__repr__();
}

template <typename T> std::string PyArray<T>::__str__() const {
  return __repr__();
  // return PyArrayBase<T, Array>::__str__();
}

template <typename T> PyObject *PyArray<T>::asNumpyArray() const {
  return array2numpy(*this);
  // return PyArrayBase<T, Array>::asNumpyArray();
}

// -------------------------------------
//
//  P Y   A R R A Y   R E F
//
// -------------------------------------
template <typename T> PyArrayRef<T>::PyArrayRef() : ArrayRef(getType()) {}

template <typename T>
PyArrayRef<T>::PyArrayRef(const ArrayRef &a) : ArrayRef(a) {}

template <typename T> NTA_BasicType PyArrayRef<T>::getType() {
  T t = 0;
  return getBasicType(t);
}

template <typename T> T PyArrayRef<T>::__getitem__(int i) const {
  return ((T *)(getBuffer()))[i];
  // return PyArrayBase<T, Array>::__getitem__(i);
}

template <typename T> void PyArrayRef<T>::__setitem__(int i, T x) {
  ((T *)(getBuffer()))[i] = x;
  // PyArrayBase<T, Array>::__setitem__(i, x);
}

template <typename T> size_t PyArrayRef<T>::__len__() const {
  return getCount();
  // return PyArrayBase<T, Array>::__len__();
}

template <typename T> std::string PyArrayRef<T>::__repr__() const {
  std::stringstream ss;
  ss << "[ ";
  for (size_t i = 0; i < __len__(); ++i)
    ss << __getitem__(i) << " ";
  ss << "]";
  return ss.str();

  // return PyArrayBase<T, Array>::__repr__();
}

template <typename T> std::string PyArrayRef<T>::__str__() const {
  return __repr__();
  // return PyArrayBase<T, Array>::__str__();
}

template <typename T> PyObject *PyArrayRef<T>::asNumpyArray() const {
  return array2numpy(*this);
  // return PyArrayBase<T, Array>::asNumpyArray();
}

template class PyArray<NTA_Byte>;
template class PyArray<NTA_Int16>;
template class PyArray<NTA_UInt16>;
template class PyArray<NTA_Int32>;
template class PyArray<NTA_UInt32>;
template class PyArray<NTA_Int64>;
template class PyArray<NTA_UInt64>;
template class PyArray<NTA_Real32>;
template class PyArray<NTA_Real64>;
template class PyArray<bool>;

template class PyArrayRef<NTA_Byte>;
template class PyArrayRef<NTA_Int16>;
template class PyArrayRef<NTA_UInt16>;
template class PyArrayRef<NTA_Int32>;
template class PyArrayRef<NTA_UInt32>;
template class PyArrayRef<NTA_Int64>;
template class PyArrayRef<NTA_UInt64>;
template class PyArrayRef<NTA_Real32>;
template class PyArrayRef<NTA_Real64>;
template class PyArrayRef<bool>;
} // namespace nupic
