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
PyBind11 Module for Math
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nupic_ext
{
    void init_array_algo(py::module&);
    void init_Domain(py::module&);
    void init_Math_Functions(py::module&);
    void init_Random(py::module&);
    void init_reals(py::module&);
    void init_Set(py::module&);
    void init_SM_01_32_32(py::module&);
    void init_SM32(py::module&);
    void init_SparseTensor(py::module&);
    void init_TensorIndex(py::module&);
} // namespace nupic_ext

using namespace nupic_ext;

PYBIND11_MODULE(math, m) {
    m.doc() = "nupic.bindings.math plugin"; // optional module docstring

    init_array_algo(m);
    init_Domain(m);
    init_SM32(m); // has to be defined before NearestNeighbor
    init_Math_Functions(m);
    init_Random(m);
    init_reals(m);
    init_Set(m);
    init_SM_01_32_32(m);
    init_SparseTensor(m);
    init_TensorIndex(m);
}
