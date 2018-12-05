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
PyBind11 bindings for Gaussian_2D class
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

#include "bindings/engine/py_utils.hpp"

#include <nupic/math/Math.hpp>

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_Gaussian_2D(py::module& m)
    {
        typedef Gaussian2D<Real32> Gaussian2D_t;

        py::class_<Gaussian2D_t> py_Gaussian2D(m, "Gaussian2D");

        // T c_x_, T c_y_, T s00_, T s01_, T s10_, T s11_
        py_Gaussian2D.def(py::init<Real32, Real32, Real32, Real32, Real32, Real32>()
            , py::arg("c_x_"), py::arg("c_y_"), py::arg("s00_"), py::arg("s01_"), py::arg("s10_"), py::arg("s11_"));

        // inline nupic::Real32 eval(nupic::Real32 x, nupic::Real32 y) const
        py_Gaussian2D.def("construct", [](const Gaussian2D_t& self, Real32 x, Real32 y)
        {
            return self.operator()(x, y);
        });
    }
} // namespace nupic_ext
