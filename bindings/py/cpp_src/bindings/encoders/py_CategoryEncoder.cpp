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

#include <nupic/encoders/CategoryEncoder.hpp>

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_CategoryEncoder(py::module& m)
    {
        py::class_<CategoryEncoder<UInt>> py_CategoryEncoder(m, "CategoryEncoder",
R"(
TODO: DOCSTRINGS!
)");

        py_CategoryEncoder.def(py::init<UInt, Real, UInt>(),
        // TODO DOCSTRINGS FOR INIT
            py::arg("size"),
            py::arg("sparsity"),
            py::arg("seed") = 0u);

        // TODO DOCSTRINGS FOR ENCODE
        py_CategoryEncoder.def("encode", &CategoryEncoder<UInt>::encode);

        py_CategoryEncoder.def("encode", []
            (CategoryEncoder<UInt> &self, UInt value) {
                auto sdr = new SDR({ self.size });
                self.encode( value, *sdr );
                return sdr;
        });
    }
}
