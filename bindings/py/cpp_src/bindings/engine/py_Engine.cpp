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
PyBind11 bindings for Engine classes
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
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


#include <nupic/os/OS.hpp>
#include <nupic/os/Timer.hpp>

#include <nupic/ntypes/ArrayRef.hpp>

#include <nupic/engine/Link.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Spec.hpp>
#include <plugin/PyBindRegion.hpp>
#include <plugin/RegisteredRegionImplPy.hpp>

namespace py = pybind11;
using namespace nupic;

namespace nupic_ext
{
    void init_Engine(py::module& m)
    {
        ///////////////////
        // Dimensions
        ///////////////////

        py::class_<Dimensions> py_Dimensions(m, "Dimensions");

        // constructors
        py_Dimensions.def(py::init<>())
            .def(py::init<std::vector<size_t>>())
            .def(py::init<size_t>())
            .def(py::init<size_t, size_t>())
            .def(py::init<size_t, size_t, size_t>());

        // members
        py_Dimensions.def("getCount", &Dimensions::getCount)
            .def("getDimensionCount", &Dimensions::getDimensionCount)
            .def("getDimension", &Dimensions::getDimension)
            .def("isUnspecified", &Dimensions::isUnspecified)
            .def("isDontcare", &Dimensions::isDontcare)
            .def("isSpecified", &Dimensions::isSpecified)
            .def("isOnes", &Dimensions::isOnes)
            .def("isValid", &Dimensions::isValid)
            .def("getIndex", &Dimensions::getIndex)
            .def("getCoordinate", &Dimensions::getCoordinate)
            .def("toString", &Dimensions::toString, "", py::arg("humanReadable") = true)
            .def("promote", &Dimensions::promote)
            ;

        // operator overloading
        py_Dimensions.def(py::self == py::self)
            .def(py::self != py::self);

        // python slots
        py_Dimensions.def("__str__", &Dimensions::toString)
            .def("__repr__", &Dimensions::toString);

        ///////////////////
        // Link
        ///////////////////
        py::class_<Link, std::shared_ptr<Link>> py_Link(m, "Link");

        // constructors
        py_Link.def(py::init<>())
            .def(py::init<const std::string&, const std::string&
                , const std::string&, const std::string&
                , const std::string&, const std::string&
                , size_t>()
                , ""
                , py::arg("linkType"), py::arg("linkParams")
                , py::arg("srcRegionName"), py::arg("destRegionName")
                , py::arg("srcOutputName") = "", py::arg("destInputName") = ""
                , py::arg("propagationDelay") = 0);




        ///////////////////
        // Spec
        ///////////////////
        py::class_<Spec> py_Spec(m, "Spec");

        ///////////////////
        // Region
        ///////////////////

        py::class_<Region, std::shared_ptr<Region>> py_Region(m, "Region");

        py_Region.def("getName", &Region::getName)
            .def("getType", &Region::getType)
            .def("getDimensions", &Region::getDimensions)
            .def("setDimensions", &Region::setDimensions);

        py_Region.def("__setattr__", [](Region& r, const std::string& Name, py::dict& d)
        {
            //r.python_attributes.insert(std::pair<std::string, py::object>(Name, d));
        });


        py_Region.def("__setattr__", [](Region& r, py::args args)
        {
            auto num_args = args.size();

            for (size_t i = 0; i < args.size(); ++i)
            {
                auto arg = args[i];
                std::string as_string = py::str(arg.get_type());

                if (py::isinstance<py::str>(arg))
                {
                    auto str = arg.cast<std::string>();
                }
                else if (py::isinstance<py::dict>(arg))
                {
                    auto dict = arg.cast<std::map<std::string, std::string>>();
                }
            }
        });



        py_Region.def("__getattr__", [](const Region& r, py::args args)
        {
            auto num_args = args.size();

            for (size_t i = 0; i < args.size(); ++i)
            {
                auto arg = args[i];
                std::string as_string = py::str(arg.get_type());

                if (py::isinstance<py::str>(arg))
                {
                    std::stringstream ss;
                    ss << "Attribute " << arg.cast<std::string>() << " not found";

                    throw std::runtime_error(ss.str());
                }
                else
                {
                    throw std::runtime_error("Unknown attribute.");
                }
            }
        });


        /*

        getSpec

        static member
        getSpecFromType
        */

        py_Region.def("getSelf", [](const Region& self)
        {
            return self.getParameterHandle("self");
        });

        py_Region.def("getInputArray", [](const Region& self, const std::string& name)
        {
            auto array_ref = self.getInputData(name);

            return py::array_t<nupic::Byte>();
        });

        py_Region.def("getOutputArray", [](const Region& self, const std::string& name)
        {
            auto array_ref = self.getInputData(name);

            return py::array_t<nupic::Byte>();
        });


        ///////////////////
        // Network
        ///////////////////

        py::class_<Network> py_Network(m, "Network");

        // constructors
        py_Network.def(py::init<>())
            .def(py::init<std::string>());

        py_Network.def("getRegions", &nupic::Network::getRegions)
            .def("getRegion", &nupic::Network::getRegion)
            .def("getLinks", &nupic::Network::getLinks)
            .def("getMinPhase", &nupic::Network::getMinPhase)
            .def("getMaxPhase", &nupic::Network::getMaxPhase)
            .def("setMinEnabledPhase", &nupic::Network::getMinPhase)
            .def("setMaxEnabledPhase", &nupic::Network::getMaxPhase)
            .def("getMinEnabledPhase", &nupic::Network::getMinPhase)
            .def("getMaxEnabledPhase", &nupic::Network::getMaxPhase);

        py_Network.def("initialize", &nupic::Network::initialize);

		py_Network.def("addRegion",
			(Region_Ptr_t (nupic::Network::*)(
					const std::string&,
  					const std::string&,
                    const std::string&))
					&nupic::Network::addRegion,
					"Normal add region."
					, py::arg("name")
					, py::arg("nodeType" )
					, py::arg("nodeParams"));
		py_Network.def("addRegion",
			(Region_Ptr_t (nupic::Network::*)(
			 		std::istream &stream,
                    std::string))
					&nupic::Network::addRegion,
					"add deserialized region."
					, py::arg("stream")
					, py::arg("name") = "");


		py_Network.def("addRegionFromBundle", &nupic::Network::addRegionFromBundle
			, "A function to load a serialized region into a Network framework."
			, py::arg("name")
			, py::arg("nodeType")
			, py::arg("dimensions")
			, py::arg("filename")
			, py::arg("label") = "");

        py_Network.def("link", &nupic::Network::link
            , "Defines a link between regions"
            , py::arg("srcName"), py::arg("destName")
            , py::arg("linkType"), py::arg("linkParams")
            , py::arg("srcOutput") = "", py::arg("destInput") = "", py::arg("propagationDelay") = 0);


        // plugin registration
        //     (note: we are re-directing these to static functions on the PyBindRegion class)
        py_Network.def_static("registerPyRegion",
		                 [](const std::string& nodeType,
							const std::string& module) {
				nupic::RegisteredRegionImplPy::registerPyRegion(nodeType, module, "");
			});
		py_Network.def_static("registerPyRegion",
		                 [](const std::string& nodeType,
							const std::string& module,
                            const std::string& className) {
				nupic::RegisteredRegionImplPy::registerPyRegion(nodeType, module, className);
			});


        py_Network.def_static("unregisterPyRegion",
			             [](const std::string& nodeType) {
				nupic::RegisteredRegionImplPy::unregisterPyRegion(nodeType);
			});



        ///////////////////
        // Collection
        ///////////////////

        // Regions
        typedef Collection<std::shared_ptr<Region>> Region_Collection_t;
        py::class_<Region_Collection_t> py_RegionCollection(m, "RegionCollection");
        py_RegionCollection.def("getByName", &Region_Collection_t::getByName);
        py_RegionCollection.def("contains", &Region_Collection_t::contains);
        py_RegionCollection.def("getCount", &Region_Collection_t::getCount);

        // bare bone sequence protocol
        py_RegionCollection.def("__len__", &Region_Collection_t::getCount);
        py_RegionCollection.def("__getitem__", [](Region_Collection_t& coll, size_t i)
        {
            if (i >= coll.getCount())
            {
                throw py::index_error();
            }

            return coll.getByIndex(i);
        });

        // Links
        typedef Collection<std::shared_ptr<Link>> Link_Collection_t;
        py::class_<Link_Collection_t> py_LinkCollection(m, "Link_Collection_t");
        py_LinkCollection.def("getByName", &Link_Collection_t::getByName);
        py_LinkCollection.def("contains", &Link_Collection_t::contains);
        py_LinkCollection.def("getCount", &Link_Collection_t::getCount);

        // bare bone sequence protocol
        py_LinkCollection.def("__len__", &Link_Collection_t::getCount);
        py_LinkCollection.def("__getitem__", [](Link_Collection_t& coll, size_t i)
        {
            if (i >= coll.getCount())
            {
                throw py::index_error();
            }

            return coll.getByIndex(i);
        });

        // not sure we need __iter__
        //py_LinkCollection.def("__iter__", [](Link_Collection_t& coll) { return py::make_iterator(coll.begin(), coll.end()); }, py::keep_alive<0, 1>());
    }

} // namespace nupic_ext
