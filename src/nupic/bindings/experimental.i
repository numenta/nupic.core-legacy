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

%pythoncode %{
  import numpy

  # Without this, Python scripts that haven't imported nupic.bindings.algorithms
  # will get a SwigPyObject rather than a SWIG-wrapped Connections instance
  # when accessing the ExtendedTemporalMemory's connections.
  import nupic.bindings.algorithms
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
                 formInternalBasalConnections=True,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255,
                 seed=42,
                 learnOnOneCell=False):
      self.this = _EXPERIMENTAL.new_ExtendedTemporalMemory(
        columnDimensions, cellsPerColumn, activationThreshold,
        initialPermanence, connectedPermanence,
        minThreshold, maxNewSynapseCount, permanenceIncrement,
        permanenceDecrement, predictedSegmentDecrement,
        formInternalBasalConnections, learnOnOneCell, seed, maxSegmentsPerCell,
        maxSynapsesPerSegment)

      self.activeExternalCellsBasal = numpy.array([], dtype=uintDType)
      self.activeExternalCellsApical = numpy.array([], dtype=uintDType)
      self.connections = self.basalConnections

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


    def activateCells(self,
                      activeColumns,
                      prevActiveExternalCellsBasal,
                      prevActiveExternalCellsApical,
                      learn=True):
      """
      Calculate the active cells, using the current active columns and dendrite
      segments. Grow and reinforce synapses.

      @param activeColumns (iterable)
      Indices of active columns.

      @param prevActiveExternalCellsBasal (iterable)
      External cells that were used to calculate the current basal segment
      excitation.

      @param prevActiveExternalCellsApical (iterable)
      External cells that were used to calculate the current apical segment
      excitation.

      @param learn (boolean)
      Whether to grow / reinforce / punish synapses.
      """
      columnsArray = numpy.array(sorted(activeColumns), dtype=uintDType)
      basalArray = numpy.array((sorted(prevActiveExternalCellsBasal)
                                if prevActiveExternalCellsBasal is not None
                                  else []),
                               dtype=uintDType)
      apicalArray = numpy.array((sorted(prevActiveExternalCellsApical)
                                if prevActiveExternalCellsApical is not None
                                  else []),
                               dtype=uintDType)

      self.convertedActivateCells(columnsArray, basalArray, apicalArray, learn)


    def activateDendrites(self,
                          activeExternalCellsBasal,
                          activeExternalCellsApical,
                          learn=True):
      """
      Calculate dendrite segment activity, using the current active cells.

      @param activeExternalCellsBasal (iterable)
      Active external cells for activating basal dendrites.

      @param activeExternalCellsApical (iterable)
      Active external cells for activating apical dendrites.

      @param learn (boolean)
      If true, segment activations will be recorded. This information is used
      during segment cleanup.
      """
      basalArray = numpy.array((sorted(activeExternalCellsBasal)
                                if activeExternalCellsBasal is not None
                                  else []),
                               dtype=uintDType)
      apicalArray = numpy.array((sorted(activeExternalCellsApical)
                                if activeExternalCellsApical is not None
                                  else []),
                               dtype=uintDType)

      self.convertedActivateDendrites(basalArray, apicalArray, learn)


    def compute(self,
                activeColumns,
                activeExternalCells=None,
                activeApicalCells=None,
                formInternalConnections=True,
                learn=True):
      """
      Perform one time step of the Temporal Memory algorithm.

      This method calls activateCells, then calls activateDendrites. Using
      the TemporalMemory via its compute method ensures that you'll always
      be able to call getPredictiveCells to get predictions for the next
      time step.

      @param activeColumns (iterable)
      Indices of active columns.

      @param activeExternalCellsBasal (iterable)
      Active external cells that should be used for activating basal dendrites
      in this timestep.

      @param activeExternalCellsApical (iterable)
      Active external cells that should be used for activating apical dendrites
      in this timestep.

      @param formInternalConnections (boolean)
      Whether to grow synapses to other cells within this temporal memory.

      @param learn (boolean)
      Whether or not learning is enabled.
      """
      activeColumnsArray = numpy.array(sorted(activeColumns), dtype=uintDType)
      activeExternalCellsBasal = numpy.array((sorted(activeExternalCells)
                                              if activeExternalCells is not None
                                              else []),
                                             dtype=uintDType)
      activeExternalCellsApical = numpy.array((sorted(activeApicalCells)
                                               if activeApicalCells is not None
                                               else []),
                                              dtype=uintDType)

      _EXPERIMENTAL.ExtendedTemporalMemory_setFormInternalBasalConnections(
        self, formInternalConnections)

      self.convertedCompute(
        activeColumnsArray,
        self.activeExternalCellsBasal,
        activeExternalCellsBasal,
        self.activeExternalCellsApical,
        activeExternalCellsApical,
        learn)

      self.activeExternalCellsBasal = activeExternalCellsBasal
      self.activeExternalCellsApical = activeExternalCellsApical

    def reset(self):
      self.activeExternalCellsBasal = numpy.array([], dtype=uintDType)
      self.activeExternalCellsApical = numpy.array([], dtype=uintDType)
      _EXPERIMENTAL.ExtendedTemporalMemory_reset(self)

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

  inline PyObject* cellsForColumn(UInt columnIdx)
  {
    const vector<CellIdx> cellIdxs = self->cellsForColumn(columnIdx);
    return vectorToList(cellIdxs);
  }

  inline void convertedActivateCells(PyObject *py_activeColumns,
                                     PyObject *py_prevActiveExternalCellsBasal,
                                     PyObject *py_prevActiveExternalCellsApical,
                                     bool learn)
  {
    PyArrayObject* _activeColumns =
      (PyArrayObject*) py_activeColumns;
    size_t activeColumnsSize =
      PyArray_DIMS(_activeColumns)[0];
    UInt32* activeColumns =
      (UInt32*)PyArray_DATA(_activeColumns);

    PyArrayObject* _prevActiveExternalCellsBasal =
      (PyArrayObject*) py_prevActiveExternalCellsBasal;
    size_t prevActiveExternalCellsBasalSize =
      PyArray_DIMS(_prevActiveExternalCellsBasal)[0];
    CellIdx* prevActiveExternalCellsBasal =
      (CellIdx*)PyArray_DATA(_prevActiveExternalCellsBasal);

    PyArrayObject* _prevActiveExternalCellsApical =
      (PyArrayObject*) py_prevActiveExternalCellsApical;
    size_t prevActiveExternalCellsApicalSize =
      PyArray_DIMS(_prevActiveExternalCellsApical)[0];
    CellIdx* prevActiveExternalCellsApical =
      (CellIdx*)PyArray_DATA(_prevActiveExternalCellsApical);

    self->activateCells(activeColumnsSize,
                        activeColumns,
                        prevActiveExternalCellsBasalSize,
                        prevActiveExternalCellsBasal,
                        prevActiveExternalCellsApicalSize,
                        prevActiveExternalCellsApical,
                        learn);
  }

  inline void convertedActivateDendrites(PyObject *py_activeExternalCellsBasal,
                                         PyObject *py_activeExternalCellsApical,
                                         bool learn)
{
    PyArrayObject* _activeExternalCellsBasal =
      (PyArrayObject*) py_activeExternalCellsBasal;
    size_t activeExternalCellsBasalSize =
      PyArray_DIMS(_activeExternalCellsBasal)[0];
    CellIdx* activeExternalCellsBasal =
      (CellIdx*)PyArray_DATA(_activeExternalCellsBasal);

    PyArrayObject* _activeExternalCellsApical =
      (PyArrayObject*) py_activeExternalCellsApical;
    size_t activeExternalCellsApicalSize =
      PyArray_DIMS(_activeExternalCellsApical)[0];
    CellIdx* activeExternalCellsApical =
      (CellIdx*)PyArray_DATA(_activeExternalCellsApical);

    self->activateDendrites(activeExternalCellsBasalSize,
                            activeExternalCellsBasal,
                            activeExternalCellsApicalSize,
                            activeExternalCellsApical,
                            learn);
  }

  inline void convertedCompute(PyObject *py_activeColumns,
                               PyObject *py_prevActiveExternalCellsBasal,
                               PyObject *py_activeExternalCellsBasal,
                               PyObject *py_prevActiveExternalCellsApical,
                               PyObject *py_activeExternalCellsApical,
                               bool learn)
  {
    PyArrayObject* _activeColumns =
      (PyArrayObject*) py_activeColumns;
    size_t activeColumnsSize =
      PyArray_DIMS(_activeColumns)[0];
    UInt32* activeColumns =
      (UInt32*)PyArray_DATA(_activeColumns);

    PyArrayObject* _prevActiveExternalCellsBasal =
      (PyArrayObject*) py_prevActiveExternalCellsBasal;
    size_t prevActiveExternalCellsBasalSize =
      PyArray_DIMS(_prevActiveExternalCellsBasal)[0];
    CellIdx* prevActiveExternalCellsBasal =
      (CellIdx*)PyArray_DATA(_prevActiveExternalCellsBasal);

    PyArrayObject* _activeExternalCellsBasal =
      (PyArrayObject*) py_activeExternalCellsBasal;
    size_t activeExternalCellsBasalSize =
      PyArray_DIMS(_activeExternalCellsBasal)[0];
    CellIdx* activeExternalCellsBasal =
      (CellIdx*)PyArray_DATA(_activeExternalCellsBasal);

    PyArrayObject* _prevActiveExternalCellsApical =
      (PyArrayObject*) py_prevActiveExternalCellsApical;
    size_t prevActiveExternalCellsApicalSize =
      PyArray_DIMS(_prevActiveExternalCellsApical)[0];
    CellIdx* prevActiveExternalCellsApical =
      (CellIdx*)PyArray_DATA(_prevActiveExternalCellsApical);

    PyArrayObject* _activeExternalCellsApical =
      (PyArrayObject*) py_activeExternalCellsApical;
    size_t activeExternalCellsApicalSize =
      PyArray_DIMS(_activeExternalCellsApical)[0];
    CellIdx* activeExternalCellsApical =
      (CellIdx*)PyArray_DATA(_activeExternalCellsApical);

    self->compute(activeColumnsSize,
                  activeColumns,
                  prevActiveExternalCellsBasalSize,
                  prevActiveExternalCellsBasal,
                  activeExternalCellsBasalSize,
                  activeExternalCellsBasal,
                  prevActiveExternalCellsApicalSize,
                  prevActiveExternalCellsApical,
                  activeExternalCellsApicalSize,
                  activeExternalCellsApical,
                  learn);
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
%ignore nupic::experimental::extended_temporal_memory::ExtendedTemporalMemory::cellsForColumn;

%include <nupic/experimental/ExtendedTemporalMemory.hpp>
