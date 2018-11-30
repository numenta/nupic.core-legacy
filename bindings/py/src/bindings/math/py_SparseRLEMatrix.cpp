/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
 *
 * Author: @chhenning, 2018
 * ---------------------------------------------------------------------
 */

/** @file
PyBind11 bindings for SparseRLEMatrix class
*/


#include <fstream>

// the use of 'register' keyword is removed in C++17
// Python2.7 uses 'register' in unicodeobject.h
#ifdef _WIN32
#pragma warning( disable : 5033)  // MSVC
#else
#pragma GCC diagnostic ignored "-Wregister"  // for GCC and CLang
#endif

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseRLEMatrix.hpp>

#include "helpers/engine/py_utils.hpp"

namespace py = pybind11;

namespace nupic_ext
{
    void init_SparseRLEMatrix(py::module& m)
    {
        // ////////////////////////////
        // nupic::SparseRLEMatrix<nupic::UInt16, unsigned char>
        // ////////////////////////////
        // def __str__(self):
        // 
        // def __setstate__(self, inString):
        // 
        // inline PyObject* __getstate__() const
        // 
        // inline void readState(PyObject* str)
        // 
        // inline void appendRow(PyObject* py_x)
        // 
        // inline PyObject* getRowToDense(nupic::UInt32 row) const
        // 
        // inline nupic::UInt16 firstRowCloserThan(PyObject* py_x, nupic::Real32 d) const
        // 
        // inline void fromDense(PyObject* py_m)
        // 
        // inline PyObject* toDense() const
        // 
        // inline PyObject* toCSR() const
        // 
        // inline void fromCSR(PyObject* str)
        // 
        // inline void CSRSaveToFile(const std::string& filename) const
        // 
        // inline void CSRLoadFromFile(const std::string& filename)
        // 
        // 
        // 
        // ////////////////////////////
        // nupic::SparseRLEMatrix<nupic::UInt16, nupic::UInt16>
        // ////////////////////////////
        // def __str__(self):
        // 
        // def __setstate__(self, inString):
        // 
        // inline PyObject* __getstate__() const
        // 
        // inline void readState(PyObject* str)
        // 
        // inline void appendRow(PyObject* py_x)
        // 
        // inline PyObject* getRowToDense(nupic::UInt32 row) const
        // 
        // inline nupic::UInt16 firstRowCloserThan(PyObject* py_x, nupic::Real32 d) const
        // 
        // inline void fromDense(PyObject* py_m)
        // 
        // inline PyObject* toDense() const
        // 
        // inline PyObject* toCSR() const
        // 
        // inline void fromCSR(PyObject* str)
        // 
        // inline void CSRSaveToFile(const std::string& filename) const
        // 
        // inline void CSRLoadFromFile(const std::string& filename)
        // 
        // 
        // ////////////////////////////  
        // nupic::SparseRLEMatrix<nupic::UInt32, nupic::Real32>
        // ////////////////////////////
        // 
        // def __str__(self):
        // return self.toDense().__str__()
        // 
        // def __setstate__(self, inString):
        // 
        // inline PyObject* __getstate__() const
        // 
        // inline void readState(PyObject* str)
        // 
        // inline void appendRow(PyObject* py_x)
        // 
        // inline PyObject* getRowToDense(nupic::UInt32 row) const
        // 
        // inline nupic::UInt32 firstRowCloserThan(PyObject* py_x, nupic::Real32 d) const
        // 
        // inline void fromDense(PyObject* py_m)
        // 
        // inline PyObject* toDense() const
        // 
        // inline PyObject* toCSR() const
        // 
        // inline void fromCSR(PyObject* str)
        // 
        // inline void CSRSaveToFile(const std::string& filename) const
        // 
        // inline void CSRLoadFromFile(const std::string& filename)    
    }
} // namespace nupic_ext
