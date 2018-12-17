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
PyBind11 bindings for Set class
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bindings/engine/py_utils.hpp"

#include <nupic/math/Set.hpp>
#include <nupic/types/Types.hpp>

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_Set(py::module& m)
    {
        typedef Set<nupic::UInt32> Set_t;

        py::class_<Set_t> py_Set(m, "Set");

        py_Set.def(py::init<>());
        py_Set.def(py::init<>([](nupic::UInt32 m, py::array_t<nupic::UInt32>& a)
        {
            return Set_t(m, static_cast<nupic::UInt32>(a.shape(0)), get_it(a));
        }));

        py_Set.def("n_elements", &Set_t::n_elements);
        py_Set.def("max_index", &Set_t::max_index);
        py_Set.def("n_bytes", &Set_t::n_bytes);


        // inline void construct(nupic::UInt32 m, PyObject* py_a)
        py_Set.def("construct", [](Set_t& self, nupic::UInt32 m, py::array_t<nupic::UInt32>& a)
        {
            self.construct(m, static_cast<UInt32>(a.shape(0)), get_it(a));
        });

        //
        // inline nupic::UInt32 intersection(PyObject* py_s2, PyObject* py_r) const
        py_Set.def("intersection", [](const Set_t& self, py::array_t<nupic::UInt32>& s2, py::array_t<nupic::UInt32>& r)
        {
            return self.intersection(static_cast<UInt32>(s2.shape(0)), get_it(s2), get_it(r));
        });
    }
} // namespace nupic_ext
