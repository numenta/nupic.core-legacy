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
PyBind11 bindings for ArrayAlgo class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseMatrix.hpp>
#include "bindings/engine/py_utils.hpp"

#include <fstream>

namespace py = pybind11;
typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32>> SparseMatrix32_t;
typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real64, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real64>> _SparseMatrix64;

namespace nupic_ext
{
    void init_array_algo(py::module& m)
    {
        // count_gt
        m.def("count_gt", [](py::array_t<nupic::Real32>& x, nupic::Real32 threshold)
        {
            return nupic::count_gt(get_it(x), get_end(x), threshold);
        });

        // count_gte
        m.def("count_gte", [](py::array_t<nupic::Real32>& x, nupic::Real32 threshold)
        {
            return nupic::count_gte(get_it(x), get_end(x), threshold);
        });

        // count_lt
        m.def("count_lt", [](py::array_t<nupic::Real32>& x, nupic::Real32 threshold)
        {
            return nupic::count_lt(get_it(x), get_end(x), threshold);
        });

        // partialArgsort
        m.def("partialArgsort", [](size_t k, py::array_t<nupic::Real32>& x, py::array_t<nupic::UInt32>& r, int direction)
        {
            return nupic::partial_argsort(k, get_it(x), get_end(x), get_it(r), get_end(r), direction);
        }, "", py::arg("k"), py::arg("x"), py::arg("r"), py::arg("direction") = -1);

        // positiveLearningPartialArgsort
        m.def("positiveLearningPartialArgsort", [](size_t k, py::array_t<nupic::Real32>& x, py::array_t<nupic::UInt32>& r, nupic::Random& rng, bool real_random)
        {
            return nupic::partial_argsort_rnd_tie_break(k, get_it(x), get_end(x), get_it(r), get_end(r), rng, real_random);
        }, "", py::arg("k"), py::arg("x"), py::arg("r"), py::arg("rng"), py::arg("real_random") = false);

        // logicalAnd
        m.def("logicalAnd", [](py::array_t<nupic::Real32>& x, py::array_t<nupic::Real32>& y)
        {
            py::array_t<nupic::Real32> z(x.size());

            nupic::logical_and(get_it(x), get_end(x), get_it(y), get_end(y), get_it(z), get_end(z));

            return z;
        });

        // logicalAnd2
        m.def("logicalAnd2", [](py::array_t<nupic::Real32>& x, py::array_t<nupic::Real32>& y)
        {
            nupic::in_place_logical_and(get_it(x), get_end(x), get_it(y), get_end(y));
        });

        //inline PyObject* binarize_with_threshold(nupic::Real32 threshold, PyObject* py_x)
        m.def("binarize_with_threshold", [](nupic::Real32 threshold, py::array_t<nupic::Real32>& x)
        {
            py::array_t<nupic::Real32> y(x.size());

            nupic::UInt32 c = nupic::binarize_with_threshold(threshold, get_it(x), get_end(x), get_it(y), get_end(y));

            return py::make_tuple(c, y);
        }, "A function that binarizes a dense vector.");
    }
} // namespace nupic_ext
