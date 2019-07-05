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
Algorithm bindings Module file for pybind11
*/

#include <bindings/suppress_register.hpp>  //must be before pybind11.h
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace htm_ext
{
    void init_Connections(py::module&);
    void init_TemporalMemory(py::module&);
    void init_SDR_Classifier(py::module&);
    void init_Spatial_Pooler(py::module&);

} // namespace htm_ext

using namespace htm_ext;

PYBIND11_MODULE(algorithms, m) {
    m.doc() = "htm.core.algorithms plugin"; // optional module docstring

    init_Connections(m);
    init_TemporalMemory(m);
    init_SDR_Classifier(m);
    init_Spatial_Pooler(m);
}
