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
PyBind11 bindings for Domain class
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/Domain.hpp>

#include "SparseTensorIndex.hpp"
#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;
using namespace nupic;

namespace nupic_ext
{
    void init_Domain(py::module& m)
    {
        typedef nupic::Domain<nupic::UInt32> Domain_t;

        py::class_<Domain_t> py_Domain(m, "Domain");

        py_Domain.def(py::init([](const TIV& lowerHalfSpace)
        {
            return Domain_t(lowerHalfSpace);
        }),"",py::arg("lowerHalfSpace"));

        py_Domain.def(py::init([](const TIV &lower, const TIV &upper)
        {
            return Domain_t(lower, upper);
        }), "", py::arg("lower"), py::arg("upper"));

        py_Domain.def("getLowerBound", [](const Domain_t& self)
        {
            PyBindTensorIndex bounds(self.rank(), (const nupic::UInt32 *) 0);
            self.getLB(bounds);

            return bounds;
        });

        py_Domain.def("getUpperBound", [](const Domain_t& self)
        {
            PyBindTensorIndex bounds(self.rank(), (const nupic::UInt32 *) 0);
            self.getUB(bounds);

            return bounds;
        });

        py_Domain.def("getDimensions", [](const Domain_t& self)
        {
            PyBindTensorIndex bounds(self.rank(), (const nupic::UInt32 *) 0);
            self.getDims(bounds);

            return bounds;
        });

        py_Domain.def("getNumOpenDims", [](const Domain_t& self)
        {
            return self.getNOpenDims();
        });

        py_Domain.def("getOpenDimensions", [](const Domain_t& self)
        {
            PyBindTensorIndex bounds(self.getNOpenDims(), (const nupic::UInt32 *) 0);
            self.getOpenDims(bounds);

            return bounds;
        });

        py_Domain.def("getSliceBounds", [](const Domain_t& self)
        {
            PyBindTensorIndex bounds(self.getNOpenDims(), (const nupic::UInt32 *) 0);
            nupic::UInt32 n = self.rank();
            nupic::UInt32 cur = 0;
            for (nupic::UInt32 i = 0; i<n; ++i) {
                nupic::DimRange<nupic::UInt32> r = self[i];
                if (!(r.getDim() == i)) throw std::invalid_argument("Out-of-order dims.");
                if (r.empty()) {}
                else {
                    bounds[cur++] = r.getUB() - r.getLB();
                }
            }

            return bounds;
        });

        py_Domain.def("doesInclude", [](const Domain_t& self, const TIV &x)
        {
            return self.includes(x);
        });

        py_Domain.def("__getitem__", [](Domain_t& self, size_t i)
        {
            nupic::DimRange<nupic::UInt32> r = self[i];
            nupic::UInt32 v[3];
            v[0] = r.getDim();
            v[1] = r.getLB();
            v[2] = r.getUB();
            return std::vector<nupic::UInt32>(v, v + 3);
        });

        py_Domain.def("__str__", [](Domain_t& self)
        {
            std::stringstream s;
            s << "(";
            nupic::UInt32 n = self.rank();
            for (nupic::UInt32 i = 0; i<n; ++i) {
                if (i) s << ", ";
                nupic::DimRange<nupic::UInt32> r = self[i];
                s << "(" << r.getDim() << ", " << r.getLB() << ", " << r.getUB() << ")";
            }
            s << ")";

            return s.str();

        });

    }

} // namespace nupic_ext
