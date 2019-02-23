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
Algorithm bindings Module file for pybind11
*/

#include <bindings/suppress_register.hpp>  //must be before pybind11.h
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nupic_ext
{
    void init_SDR(py::module&);
    void init_SDR_Metrics(py::module&);
    void init_algorithms(py::module&);
    void init_Cells4(py::module&);
    void init_HTM(py::module&);
    void init_SDR_Classifier(py::module&);
    void init_Spatial_Pooler(py::module&);

} // namespace nupic_ext

using namespace nupic_ext;

PYBIND11_MODULE(algorithms, m) {
    m.doc() = "nupic.core.algorithms plugin"; // optional module docstring

    init_SDR(m);
    init_SDR_Metrics(m);
    init_algorithms(m);
    init_HTM(m);
    init_Cells4(m);
    init_SDR_Classifier(m);
    init_Spatial_Pooler(m);
}
