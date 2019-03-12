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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <nupic/encoders/ScalarEncoder.hpp>

using namespace nupic::encoders;

namespace nupic_ext
{
  void init_ScalarEncoder(py::module& m)
  {
    py::class_<ScalarEncoderParameters> py_ScalarEncParams(m, "ScalarEncoderParameters",
        "TODO DOC");
    py_ScalarEncParams.def_readwrite("minimum", &ScalarEncoderParameters::minimum, "TODO DOC");
    py_ScalarEncParams.def_readwrite("maximum", &ScalarEncoderParameters::maximum, "TODO DOC");
    py_ScalarEncParams.def_readwrite("clipInput", &ScalarEncoderParameters::clipInput, "TODO DOC");
    py_ScalarEncParams.def_readwrite("periodic", &ScalarEncoderParameters::periodic, "TODO DOC");
    py_ScalarEncParams.def_readwrite("active", &ScalarEncoderParameters::active, "TODO DOC");
    py_ScalarEncParams.def_readwrite("sparsity", &ScalarEncoderParameters::sparsity, "TODO DOC");
    py_ScalarEncParams.def_readwrite("size", &ScalarEncoderParameters::size, "TODO DOC");
    py_ScalarEncParams.def_readwrite("radius", &ScalarEncoderParameters::radius, "TODO DOC");
    py_ScalarEncParams.def_readwrite("resolution", &ScalarEncoderParameters::resolution, "TODO DOC");
    py_ScalarEncParams.def(py::init<>(), "TODO DOC");

    py::class_<ScalarEncoder> py_ScalarEnc(m, "ScalarEncoder", "TODO DOC");
    py_ScalarEnc.def(py::init<ScalarEncoderParameters&>(), "TODO DOC");
    py_ScalarEnc.def_property_readonly("parameters",
        [](const ScalarEncoder &self) { return self.parameters; });
    py_ScalarEnc.def("encode", &ScalarEncoder::encode, "TODO DOC");
  }
}
