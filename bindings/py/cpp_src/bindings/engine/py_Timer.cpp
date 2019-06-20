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
PyBind11 bindings for Timer class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htm/os/Timer.hpp>

namespace py = pybind11;
using namespace htm;
namespace htm_ext {


    void init_Timer(py::module& m)
    {
        py::class_<Timer> py_Timer(m, "Timer");

        py_Timer.def(py::init<bool>(), py::arg("startme") = false);

        py_Timer.def("start", &Timer::start);
        py_Timer.def("stop", &Timer::stop);
        py_Timer.def("elapsed", &Timer::getElapsed);
        py_Timer.def("reset", &Timer::reset);
        py_Timer.def("startCount", &Timer::getStartCount);
        py_Timer.def("isStarted", &Timer::isStarted);
        py_Timer.def("toString", &Timer::toString);

        py_Timer.def("__str__", &Timer::toString);

    }

} // namespace htm_ext