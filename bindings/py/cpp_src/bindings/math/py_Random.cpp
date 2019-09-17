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
PyBind11 bindings for Random class
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htm/utils/Random.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;

namespace htm_ext {

    void init_Random(py::module& m)
    {
        typedef htm::Random Random_t;

        py::class_<Random_t> Random(m, "Random");

        Random.def(py::init<htm::UInt64>(), py::arg("seed") = 0)
            .def("getUInt32", &Random_t::getUInt32, py::arg("max") = (htm::UInt32)-1l)
            .def("getReal64", &Random_t::getReal64)
			.def("getSeed", &Random_t::getSeed)
            .def("max", &Random_t::max)
            .def("min", &Random_t::min)
        	.def("__eq__", [](Random_t const & self, Random_t const & other) {//wrapping operator==
            	return self == other;
        	}, py::is_operator());

        Random.def_property_readonly_static("MAX32", [](py::object) {
				return Random_t::MAX32;
			});

        //////////////////
        // sample
        /////////////////


        Random.def("sample",
            [](Random_t& r, py::array& population, const htm::UInt32 nSelect)
        {
            if (population.ndim() != 1 )
            {
                throw std::invalid_argument("Number of dimensions must be one.");
            }

	    std::vector<htm::UInt32> tmp_pop(get_it<htm::UInt32>(population), get_it<htm::UInt32>(population) + population.size()); //vector from numpy.array
            return r.sample(tmp_pop, nSelect);
        });

        //////////////////
        // shuffle
        /////////////////

        Random.def("shuffle",
            [](Random_t& r, py::array& a)
        {
            //py::scoped_ostream_redirect stream(
            //    std::cout,                               // std::ostream&
            //    py::module::import("sys").attr("stdout") // Python output
            //);

            if (a.ndim() != 1)
            {
                throw std::invalid_argument("Number of dimensions must be one.");
            }

            r.shuffle(get_it<htm::UInt32>(a), get_end<htm::UInt32>(a));
        });

        ////////////////////

        Random.def("initializeUInt32Array", [](Random_t& self, py::array& a, htm::UInt32 max_value)
        {
            auto array_data = get_it<htm::UInt32>(a);

            for (htm::UInt32 i = 0; i != a.size(); ++i)
                array_data[i] = self.getUInt32() % max_value;

        });


        ////////////////////

        Random.def("initializeReal64Array", [](Random_t& self, py::array& a)
        {
            auto array_data = get_it<htm::Real64>(a);

            for (htm::UInt32 i = 0; i != a.size(); ++i)
                array_data[i] = self.getReal64();

        });


        //////////////////
        // serialization
        /////////////////
				Random.def("saveToFile", [](Random_t& self, const std::string& name, int fmt) { 
				  htm::SerializableFormat fmt1;
				  switch(fmt) {                                             
			    case 0: fmt1 = htm::SerializableFormat::BINARY; break;
			    case 1: fmt1 = htm::SerializableFormat::PORTABLE; break;
					case 2: fmt1 = htm::SerializableFormat::JSON; break;
					case 3: fmt1 = htm::SerializableFormat::XML; break;
					default: NTA_THROW << "unknown serialization format.";
					} 
					self.saveToFile(name, fmt1); 
				}, "serialize to a File, using BINARY=0, PORTABLE=1, JSON=2, or XML=3 format.",
			  py::arg("name"), py::arg("fmt") = 0);
				
				Random.def("loadFromFile", [](Random_t& self, const std::string& name, int fmt) { 
				  htm::SerializableFormat fmt1;
				  switch(fmt) {                                             
			    case 0: fmt1 = htm::SerializableFormat::BINARY; break;
			    case 1: fmt1 = htm::SerializableFormat::PORTABLE; break;
					case 2: fmt1 = htm::SerializableFormat::JSON; break;
					case 3: fmt1 = htm::SerializableFormat::XML; break;
					default: NTA_THROW << "unknown serialization format.";
					} 
				  self.loadFromFile(name, fmt1);
				}, "load from a File, using BINARY, PORTABLE, JSON, or XML format.",
				py::arg("name"), py::arg("fmt") = 0);

        Random.def(py::pickle(
            [](const Random_t& r)
        {
            std::stringstream ss;
            ss << r;
            return ss.str();
        },
            [](const std::string& str)
        {
            if (str.empty())
            {
                throw std::runtime_error("Empty state");
            }

            std::stringstream ss(str);
            Random_t r;
            ss >> r;

            return r;
        }
        ));

    }

} // namespace htm_ext
