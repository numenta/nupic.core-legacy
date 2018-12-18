#ifndef NUPIC_EXT_BINDINGS_PY_UTILS_HPP
#define NUPIC_EXT_BINDINGS_PY_UTILS_HPP
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
Utility functions for PyBind11 bindings
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace nupic_ext {

    template<typename T> T* get_it(py::array_t<T>& a) { return (T*)a.request().ptr; }
    template<typename T> T* get_end(py::array_t<T>& a) { return ((T*)a.request().ptr) + a.size(); }

    template<typename T> T* get_it(py::array& a) { return (T*)a.request().ptr; }
    template<typename T> T* get_end(py::array& a) { return ((T*)a.request().ptr) + a.size(); }

    template<typename T> T* get_row_it(py::array_t<T>& a, int row)
    {
        auto buffer_info = a.request();

        return (T*)((char*)buffer_info.ptr + (buffer_info.strides[0] * row));
    }

    inline
    void enable_cout()
    {
        py::scoped_ostream_redirect stream(
            std::cout,                               // std::ostream&
            py::module::import("sys").attr("stdout") // Python output
        );
    }


} // namespace nupic_ext


#endif //NUPIC_EXT_BINDINGS_PY_UTILS_HPP