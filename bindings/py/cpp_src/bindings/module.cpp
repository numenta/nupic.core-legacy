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
Bindings Module file
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nupic_ext
{
    void init_Random(py::module&);
    void init_SM32(py::module&);
    void init_SM_01_32_32(py::module&);
    void init_Math_Functions(py::module&);

    void init_Engine(py::module&);
    void init_HTM(py::module&);
    void init_reals(py::module&);
    void init_Cells4(py::module&);
    void init_SDR_Classifier(py::module&);
    void init_Spatial_Pooler(py::module&);
    void init_SparseTensor(py::module&);
    void init_Domain(py::module&);
    void init_TensorIndex(py::module&);
    void init_algorithms(py::module&);
    void init_array_algo(py::module&);
    void init_Regions(py::module&);
} // namespace nupic_ext


using namespace nupic_ext;

PYBIND11_MODULE(bindings, m) {
    m.doc() = "nupic python plugin"; // optional module docstring

    init_reals(m);

    auto utils_module = m.def_submodule("utils");

    auto math_module = m.def_submodule("math");
    init_reals(math_module);
    init_Random(math_module);
    init_SM32(math_module);
    init_SM_01_32_32(math_module);
    init_Math_Functions(math_module);
    init_SMC(math_module);
    init_SparseTensor(math_module);
    init_TensorIndex(math_module);
    init_Domain(math_module);
    init_array_algo(math_module);

    auto engine_module = m.def_submodule("engine");
    init_Engine(engine_module);

    auto regions_module = m.def_submodule("regions");
    init_Regions(regions_module);

    auto algorithms_module = m.def_submodule("algorithms");
    init_algorithms(algorithms_module);
    init_HTM(algorithms_module);
    init_Cells4(algorithms_module);
    init_SDR_Classifier(algorithms_module);
    init_Spatial_Pooler(algorithms_module);
}

