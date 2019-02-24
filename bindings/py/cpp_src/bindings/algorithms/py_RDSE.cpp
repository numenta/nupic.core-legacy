/* ----------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
 * The following terms and conditions apply:
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
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>

#include <nupic/encoders/RandomDistributedScalarEncoder.hpp>

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_RDSE(py::module& m)
    {
        py::class_<RDSE> py_RDSE(m, "RDSE");
        py_RDSE.def(py::init<UInt, Real, Real, UInt>(),
            py::arg("size"),
            py::arg("sparsity"),
            py::arg("radius"),
            py::arg("seed") = 0u);
        py_RDSE.def("encode", &RDSE::encode);

        // TODO encode into SDR
    }
}
