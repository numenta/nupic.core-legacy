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
PyBind11 bindings for Math classes
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/ArrayAlgo.hpp>
#include <nupic/math/Functions.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;

namespace nupic_ext
{
	void init_Math_Functions(py::module& m)
	{
		m.def("lgamma", &nupic::lgamma<nupic::Real64>)
		 .def("digamma", &nupic::digamma<nupic::Real64>)
		 .def("beta", &nupic::beta<nupic::Real64>)
		 .def("erf", &nupic::erf<nupic::Real64>)
		 .def("beta", &nupic::digamma<nupic::Real64>);

        m.def("getGlobalEpsilon", []() {return nupic::Epsilon; });


        m.def("nearlyZeroRange", [](py::array_t<nupic::Real32>& x, nupic::Real32 eps)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            return nupic::nearlyZeroRange(get_it(x), get_end(x), eps);
        }, "", py::arg("x"), py::arg("eps") = nupic::Epsilon);


        m.def("nearlyEqualRange", [](py::array_t<nupic::Real32>& x, py::array_t<nupic::Real32>& y, nupic::Real32 eps)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }
            if (y.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            return nupic::nearlyEqualRange(get_it(x), get_end(x), get_it(y), get_end(y), eps);
        }, "", py::arg("x"), py::arg("y"), py::arg("eps") = nupic::Epsilon);


        m.def("positive_less_than", [](py::array_t<nupic::Real32>& x, nupic::Real32 eps)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            return nupic::positive_less_than(get_it(x), get_end(x), eps);
        }, "", py::arg("x"), py::arg("eps") = nupic::Epsilon);


        m.def("quantize_255", [](py::array_t<nupic::Real32>& x, nupic::Real32 x_min, nupic::Real32 x_max)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            py::array_t<nupic::Real32> y(x.size());

            throw std::runtime_error("No quantize function found.");

            //@todo
            //nupic::quantize(get_it(x), get_end(x), get_it(y), get_end(y), x_min, x_max, 1, 255);

            //return y;
        });

        m.def("quantize_65535", [](py::array_t<nupic::Real32>& x, nupic::Real32 x_min, nupic::Real32 x_max)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            py::array_t<nupic::Real32> y(x.size());

            throw std::runtime_error("No quantize function found.");

            //@todo
            //nupic::quantize(get_it(x), get_end(x), get_it(y), get_end(y), x_min, x_max, 1, 255);

            //return y;
        });

        m.def("winnerTakesAll_3", [](size_t k, size_t seg_size, py::array_t<nupic::Real32>& x)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            std::vector<int> ind;
            std::vector<nupic::Real32> nz;

            nupic::winnerTakesAll3(k, seg_size, get_it(x), get_end(x),
                std::back_inserter(ind), std::back_inserter(nz));

            return py::make_tuple(ind, nz);
        });

        m.def("min_score_per_category", [](nupic::UInt32 maxCategoryIdx
            , py::array_t<nupic::UInt32>& c
            , py::array_t<nupic::Real32>& d)
        {
            int n = int(maxCategoryIdx + 1);

            std::vector<nupic::Real32> s(n, std::numeric_limits<nupic::Real32>::max());

            // @todo Not sure why no just take the size()?
            // int nScores = int(c.end() - c.begin());
            int nScores = c.size();
            for (int i = 0; i != nScores; ++i)
            {
                if (i >= c.size()) { throw std::runtime_error("Buffer access out of bounds."); }
                if (i >= d.size()) { throw std::runtime_error("Buffer access out of bounds."); }

                auto c_i = *(get_it(c) + i);
                auto d_i = *(get_it(d) + i);

                s[c_i] = std::min(s[c_i], d_i);
            }

            return py::array_t<nupic::Real32>(s.size(), s.data());
        });

        // inline nupic::Real32 l2_norm(PyObject* py_x)
        m.def("l2_norm", [](py::array_t<nupic::Real32>& x)
        {
            return nupic::l2_norm(get_it(x), get_end(x));
        });
    }

//#ifndef _WIN32
//#pragma GCC diagnostic pop  // for GCC and CLang: resume command-line arguments, ends ignored -Wregister
//#endif


} // namespace nupic_ext
