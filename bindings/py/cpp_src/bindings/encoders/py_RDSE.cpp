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
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>

#include <nupic/encoders/RandomDistributedScalarEncoder.hpp>

namespace py = pybind11;

using namespace nupic;
using namespace nupic::encoders;

namespace nupic_ext
{
    void init_RDSE(py::module& m)
    {
        py::class_<RDSE_Parameters> py_RDSE_args(m, "RDSE_Parameters", "TODO DOCS");
        py_RDSE_args.def(py::init<>());
        py_RDSE_args.def_readwrite("size", &RDSE_Parameters::size, "TODO DOCS");
        py_RDSE_args.def_readwrite("sparsity", &RDSE_Parameters::sparsity, "TODO DOCS");
        py_RDSE_args.def_readwrite("activeBits", &RDSE_Parameters::activeBits, "TODO DOCS");
        py_RDSE_args.def_readwrite("radius", &RDSE_Parameters::radius, "TODO DOCS");
        py_RDSE_args.def_readwrite("resolution", &RDSE_Parameters::resolution, "TODO DOCS");
        py_RDSE_args.def_readwrite("seed", &RDSE_Parameters::seed, "TODO DOCS");

        py::class_<RDSE> py_RDSE(m, "RDSE", "TODO DOCS");
        py_RDSE.def(py::init<RDSE_Parameters>());

        py_RDSE.def("encode", &RDSE::encode, "TODO DOCS");

        py_RDSE.def("encode", [](RDSE &self, Real64 value) {
            auto sdr = new sdr::SDR({self.size});
            self.encode(value, *sdr);
            return sdr;
        });
    }
}
