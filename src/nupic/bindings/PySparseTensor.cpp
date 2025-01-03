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

/** @file
 */

#include <Python.h>

#include <nupic/bindings/PySparseTensor.hpp>
#include <nupic/py_support/NumpyVector.hpp>

using namespace std;
using namespace nupic;

typedef nupic::SparseTensor<PyTensorIndex, nupic::Real> STBase;

PySparseTensor::PySparseTensor(PyObject *numpyArray)
    // TODO: Switch to rank 0 (or at least dimension 0) default.
    : tensor_(PyTensorIndex(1)) {
  NumpyNDArray a(numpyArray);
  int rank = a.getRank();
  if (rank > PYSPARSETENSOR_MAX_RANK)
    throw invalid_argument(
        "Array rank exceeds max rank for SparseTensor bindings.");
  int dims[PYSPARSETENSOR_MAX_RANK]; // Never larger than max ND array rank.
  a.getDims(dims);
  tensor_ = STBase(PyTensorIndex(rank, dims));
  tensor_.fromDense(a.getData());
}

void PySparseTensor::set(const PyTensorIndex &i, PyObject *x) {
  PyObject *num = PyNumber_Float(x);
  if (!num)
    throw std::invalid_argument("value is not a float.");
  nupic::Real y = (nupic::Real)PyFloat_AsDouble(num);
  Py_CLEAR(num);
  set(i, y);
}

PyObject *PySparseTensor::toDense() const {
  const PyTensorIndex &bounds = tensor_.getBounds();
  int rank = bounds.size();
  int dims[PYSPARSETENSOR_MAX_RANK];
  if (rank > PYSPARSETENSOR_MAX_RANK)
    throw std::logic_error("Rank exceeds max rank.");
  for (int i = 0; i < rank; ++i)
    dims[i] = bounds[i];
  NumpyNDArray a(rank, dims);
  tensor_.toDense(a.getData());
  return a.forPython();
}

PyObject *PySparseTensor::__str__() const {
  PyObject *a = toDense();
  PyObject *s = PyObject_Str(a);
  Py_CLEAR(a);
  return s;
}

string PySparseTensor::__getstate__() const {
  stringstream s;
  tensor_.toStream(s);
  return s.str();
}

inline STBase SparseTensorFromString(const string &s) {
  size_t rank = 0;
  {
    stringstream forRank(s);
    forRank.exceptions(ios::failbit | ios::badbit);
    forRank >> rank;
  };
  PyTensorIndex index(rank, (const size_t *)0);
  for (size_t i = 0; i < rank; ++i) {
    index[i] = 1;
  }
  STBase tensor(index);
  stringstream toRead(s);
  tensor.fromStream(toRead);
  return tensor;
}

PySparseTensor::PySparseTensor(const string &s)
    : tensor_(SparseTensorFromString(s)) {}

double PySparseTensor::marginalize() const { return tensor_.sum(); }

PyTensorIndex PySparseTensor::argmax() const { return tensor_.max().first; }

nupic::Real PySparseTensor::max() const { return tensor_.max().second; }

PySparseTensor PySparseTensor::__mul__(const nupic::Real &x) const {
  PySparseTensor out(tensor_.getBounds());
  tensor_.multiply(x, out.tensor_);
  return out;
}
