#ifndef HTM_EXT_BINDINGS_PY_UTILS_HPP
#define HTM_EXT_BINDINGS_PY_UTILS_HPP
/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, Numenta, Inc.
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
 * Author: @chhenning, 2018
 * --------------------------------------------------------------------- */

/** @file
Utility functions for PyBind11 bindings
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace htm_ext {

    // Check that the precision in bytes matches the data size of the array
    template<typename T> T* get_it(py::array& a) //TODO add explanation why array, and not array_t
	{
	if (sizeof(T) != a.request().itemsize)
		{throw std::invalid_argument("Invalid numpy array precision used.");}

	return static_cast<T*>(a.request().ptr);
	}

    template<typename T> T* get_end(py::array& a) { return (static_cast<T*>(a.request().ptr)) + a.size(); }

    inline void enable_cout()
    {
        py::scoped_ostream_redirect stream(
            std::cout,                               // std::ostream&
            py::module::import("sys").attr("stdout") // Python output
        );
    }


} // namespace htm_ext


#endif //HTM_EXT_BINDINGS_PY_UTILS_HPP
