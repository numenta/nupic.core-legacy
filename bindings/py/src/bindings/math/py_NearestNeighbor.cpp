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
PyBind11 bindings for NearestNeighbor class
*/

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

#include <nupic/math/NearestNeighbor.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;

namespace nupic_ext
{
    void init_NearestNeighbor(py::module& m)
    {
        // the actual type NearestNeighbor is defined by a function, see sparse_matrix.i[2781]

        typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32>> Parent32_t;
        typedef nupic::NearestNeighbor<Parent32_t> NearestNeighbor32_t;

        py::class_<NearestNeighbor32_t, Parent32_t> py_nn32(m, "NearestNeighbor");

        py_nn32.def(py::init<>())
            .def(py::init<Parent32_t::size_type, Parent32_t::size_type>(), py::arg("nrows"), py::arg("ncols"));


        // python slots
        py_nn32.def("__str__", [](NearestNeighbor32_t& self)
        {
            Parent32_t p;

            //@todo doesn't compile
            //self.toDense(p);

            std::ostringstream s;
            p.toCSR(s);

            return s.str();
        });

        py_nn32.def("rowDist", [](const NearestNeighbor32_t& self, int row, py::array_t<nupic::Real32> x)
        {
            self.rowL2Dist(row, get_it(x));
        });

        // vecLpDist
        py_nn32.def("vecLpDist", [](const NearestNeighbor32_t& self, nupic::Real32 p, py::array_t<nupic::Real32>& xIn, bool take_root)
        {
            py::array_t<nupic::Real32> out(self.nRows());
            self.LpDist(p, get_it(xIn), get_it(out), take_root);

            return out;

        }, "", py::arg("p"), py::arg("xIn"), py::arg("take_root") = true);


        // LpNearest
        py_nn32.def("LpNearest", [](const NearestNeighbor32_t& self, nupic::Real32 p, py::array_t<nupic::Real32>& row, nupic::UInt32 k, bool take_root)
        {
            std::vector<std::pair<nupic::UInt32, nupic::Real32>> nn(k);

            self.LpNearest(p, get_it(row), nn.begin(), k, take_root);

            py::tuple result;
            for (int i = 0; i < nn.size(); ++i)
            {
                result[i] = nn[i];
            }

            return result;

        }, "", py::arg("p"), py::arg("row"), py::arg("k") = 1, py::arg("take_root") = true);


        // closestLp_w
        py_nn32.def("closestLp_w", [](NearestNeighbor32_t& self, nupic::Real32 p, py::array_t<nupic::Real32>& row)
        {
            std::pair<int, nupic::Real32> nn;

            self.LpNearest_w(p, get_it(row), &nn, true);

            //@todo not sure this correct
            return Py_BuildValue("(if)", nn.first, nn.second);
        });


        // closestDot
        py_nn32.def("closestDot", [](const NearestNeighbor32_t& self, py::array_t<nupic::Real32>& row)
        {
            auto r = self.dotNearest(get_it(row));

            //@todo not sure this correct
            return Py_BuildValue("(if)", r.first, r.second);
        });


        // projLpNearest
        py_nn32.def("projLpNearest", [](const NearestNeighbor32_t& self, nupic::Real32 p, py::array_t<nupic::Real32>& row, nupic::UInt32 k, bool take_root)
        {
            std::vector<std::pair<nupic::UInt32, nupic::Real32>> nn(k);

            self.projLpNearest(p, get_it(row), nn.begin(), k, take_root);

            py::tuple result;
            for (int i = 0; i < nn.size(); ++i)
            {
                result[i] = nn[i];
            }

            return result;
        });


        // projRbf
        py_nn32.def("projRbf", [](const NearestNeighbor32_t& self, nupic::Real32 p, nupic::Real32 k, py::array_t<nupic::Real32>& row)
        {
            py::array_t<nupic::Real32> y(self.nRows());

            self.projRbf(p, k, get_it(row), get_it(y));

            return y;
        });


        typedef nupic::NearestNeighbor<nupic::SparseMatrix<nupic::UInt32, nupic::Real64, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real64>>> NearestNeighbor64_t;
    }

} // namespace nupic_ext
