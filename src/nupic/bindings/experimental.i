/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
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

%module(package="bindings") experimental
%import <nupic/bindings/algorithms.i>

%pythoncode %{
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os

_EXPERIMENTAL = _experimental

%}

%{
/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <Python.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>

#include <nupic/experimental/ExtendedTemporalMemory.hpp>

#include <nupic/proto/ExtendedTemporalMemoryProto.capnp.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <nupic/py_support/NumpyVector.hpp>
#if !CAPNP_LITE
#include <nupic/py_support/PyCapnp.hpp>
#endif
#include <nupic/py_support/PythonStream.hpp>
#include <nupic/py_support/PyHelpers.hpp>

// Hack to fix SWIGPY_SLICE_ARG not found bug
#if PY_VERSION_HEX >= 0x03020000
# define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
#else
# define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
#endif

using namespace nupic::experimental::extended_temporal_memory;
using namespace nupic;

#define CHECKSIZE(var) \
  NTA_ASSERT(PyArray_DESCR(var)->elsize == 4) << " elsize:" << PyArray_DESCR(var)->elsize

%}

%pythoncode %{
  uintDType = "uint32"
%}

%naturalvar;

//--------------------------------------------------------------------------------
// Extended Temporal Memory
//--------------------------------------------------------------------------------
%inline %{
  template <typename IntType>
  inline PyObject* vectorToList(const vector<IntType> &cellIdxs)
  {
    PyObject *list = PyList_New(cellIdxs.size());
    for (size_t i = 0; i < cellIdxs.size(); i++)
    {
      PyObject *pyIdx = PyInt_FromLong(cellIdxs[i]);
      PyList_SET_ITEM(list, i, pyIdx);
    }
    return list;
  }
%}

%extend nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory
{
  %pythoncode %{
    def __init__(self,
                 columnDimensions=(2048,),
                 cellsPerColumn=32,
                 activationThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 minThreshold=10,
                 maxNewSynapseCount=20,
                 permanenceIncrement=0.10,
                 permanenceDecrement=0.10,
                 predictedSegmentDecrement=0.00,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255,
                 seed=42):
      self.this = _EXPERIMENTAL.new_ExtendedTemporalMemory()
      _EXPERIMENTAL.ExtendedTemporalMemory_initialize(
        self, columnDimensions, cellsPerColumn, activationThreshold,
        initialPermanence, connectedPermanence,
        minThreshold, maxNewSynapseCount, permanenceIncrement,
        permanenceDecrement, predictedSegmentDecrement, seed,
        maxSegmentsPerCell, maxSynapsesPerSegment)

    def __getstate__(self):
      # Save the local attributes but override the C++ temporal memory with the
      # string representation.
      d = dict(self.__dict__)
      d["this"] = self.getCState()
      return d

    def __setstate__(self, state):
      # Create an empty C++ temporal memory and populate it from the serialized
      # string.
      self.this = _EXPERIMENTAL.new_ExtendedTemporalMemory()
      if isinstance(state, str):
        self.loadFromString(state)
        self.valueToCategory = {}
      else:
        self.loadFromString(state["this"])
        # Use the rest of the state to set local Python attributes.
        del state["this"]
        self.__dict__.update(state)

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance
  %}

  inline PyObject* getActiveCells()
  {
    const vector<CellIdx> cellIdxs = self->getActiveCells();
    return vectorToList(cellIdxs);
  }

  inline PyObject* getPredictiveCells()
  {
    const vector<CellIdx> cellIdxs = self->getPredictiveCells();
    return vectorToList(cellIdxs);
  }

  inline PyObject* getWinnerCells()
  {
    const vector<CellIdx> cellIdxs = self->getWinnerCells();
    return vectorToList(cellIdxs);
  }

  inline PyObject* getMatchingCells()
  {
    const vector<CellIdx> cellIdxs = self->getMatchingCells();
    return vectorToList(cellIdxs);
  }

  inline PyObject* cellsForColumn(UInt columnIdx)
  {
    const vector<CellIdx> cellIdxs = self->cellsForColumn(columnIdx);
    return vectorToList(cellIdxs);
  }

  inline void write(PyObject* pyBuilder) const
  {
%#if !CAPNP_LITE
    ExtendedTemporalMemoryProto::Builder proto =
        getBuilder<ExtendedTemporalMemoryProto>(pyBuilder);
    self->write(proto);
  %#else
    throw std::logic_error(
        "ExtendedTemporalMemory.write is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }

  inline void convertedRead(PyObject* pyReader)
  {
%#if !CAPNP_LITE
    ExtendedTemporalMemoryProto::Reader proto =
        getReader<ExtendedTemporalMemoryProto>(pyReader);
    self->read(proto);
  %#else
    throw std::logic_error(
        "ExtendedTemporalMemory.read is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }

  PyObject* getCState()
  {
    SharedPythonOStream py_s(self->persistentSize());
    std::ostream& s = py_s.getStream();
    // TODO: Consider writing floats as binary instead.
    s.flags(ios::scientific);
    s.precision(numeric_limits<double>::digits10 + 1);
    self->save(s);
    return py_s.close();
  }
}

%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::getActiveCells;
%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::getPredictiveCells;
%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::getWinnerCells;
%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::getMatchingCells;
%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::cellsForColumn;
%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::columnForCell;

%include <nupic/experimental/ExtendedTemporalMemory.hpp>
