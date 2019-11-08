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
PyBind11 bindings for Engine classes
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htm/os/Timer.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/utils/Log.hpp>

#include <htm/engine/Link.hpp>
#include <htm/engine/Network.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/types/Sdr.hpp>

#include <plugin/PyBindRegion.hpp>
#include <plugin/RegisteredRegionImplPy.hpp>

namespace py = pybind11;
using namespace htm;

namespace htm_ext
{

    typedef std::shared_ptr<Region> Region_Ptr_t;

    void init_Engine(py::module& m)
    {
        ///////////////////
        // Dimensions
        ///////////////////

        py::class_<Dimensions> py_Dimensions(m, "Dimensions");

        // constructors
        py_Dimensions.def(py::init<>())
            .def(py::init<std::vector<UInt>>())
            .def(py::init<UInt>())
            .def(py::init<UInt, UInt>())
            .def(py::init<UInt, UInt, UInt>());

        // members
        py_Dimensions.def("getCount", &Dimensions::getCount)
            .def("size", &Dimensions::size)
            .def("isUnspecified", &Dimensions::isUnspecified)
            .def("isDontcare", &Dimensions::isDontcare)
            .def("isSpecified", &Dimensions::isSpecified)
            .def("isInvalid", &Dimensions::isInvalid)
            .def("toString", &Dimensions::toString, "", py::arg("humanReadable") = true)
            ;

        // operator overloading
        py_Dimensions.def(py::self == py::self)
            .def(py::self != py::self);

        // python slots
        py_Dimensions.def("__str__", &Dimensions::toString)
            .def("__repr__", &Dimensions::toString);

        ///////////////////
        // Array
        ///////////////////

        // The Array object will be presented to python as a Buffer object.
        // To create from a numpy array, use
        //      a = Array(numpy_array)
        //    optional parameter copy = False to avoid copying.
        //
        //     can be cast to a numPy object for direct access.
        //      np.array(this_instance)
        //     optional parameter copy = False to avoid copying.

        py::class_<Array>(m, "Array", py::buffer_protocol())
            .def_buffer([](Array &m) -> py::buffer_info {
                if (!m.has_buffer())
                    throw std::runtime_error("Array object is not initalized.");
                /* determine the Python struct-style format descriptor, see pybind11/buffer_info.h */
                std::string fmt;
                if      (m.getType() == NTA_BasicType_Int32)  fmt = py::format_descriptor<Int32>::format();
                else if    (m.getType() == NTA_BasicType_UInt32) fmt = py::format_descriptor<UInt32>::format();
                else if    (m.getType() == NTA_BasicType_Int64)  fmt = py::format_descriptor<Int64>::format();
                else if    (m.getType() == NTA_BasicType_UInt64) fmt = py::format_descriptor<UInt64>::format();
                else if    (m.getType() == NTA_BasicType_Real32) fmt = py::format_descriptor<Real32>::format();
                else if    (m.getType() == NTA_BasicType_Real64) fmt = py::format_descriptor<Real64>::format();
                else if    (m.getType() == NTA_BasicType_Bool)   fmt = py::format_descriptor<bool>::format();
                else  fmt =  py::format_descriptor<Byte>::format();  /* for Byte and SDR data */
                return py::buffer_info(
                    m.getBuffer(),                            /* Pointer to buffer */
                    BasicType::getSize(m.getType()),          /* Size of one scalar */
                    fmt,                                      /* Python struct-style format descriptor */
                    1,                                        /* Number of dimensions */
                    { m.getCount() },                         /* Buffer dimensions */
                    { BasicType::getSize(m.getType())}        /* Strides (in bytes) for each index */
                );
            })
            .def("__init__", [](Array &m, py::buffer b, bool copy){
                py::buffer_info info = b.request();  /* Request a buffer descriptor from Python */
                if (info.ndim != 1)
                    throw std::runtime_error("Expected a one dimensional array!");
                size_t size = static_cast<size_t>(info.shape[0]);
                NTA_BasicType type;
                if      (((info.format == "i") || (info.format == "l") ) && info.itemsize == 4) type = NTA_BasicType_Int32;
                else if (((info.format == "I") || (info.format == "L") ) && info.itemsize == 4) type = NTA_BasicType_UInt32;
                else if ((info.format == "l") || (info.format == "q") ) type = NTA_BasicType_Int64;
                else if ((info.format == "L") || (info.format == "Q") ) type = NTA_BasicType_UInt64;
                else if (info.format == "f") type = NTA_BasicType_Real32;
                else if (info.format == "d") type = NTA_BasicType_Real64;
                else if (info.format == py::format_descriptor<bool>::format()) type = NTA_BasicType_Bool;
                else if (info.format == py::format_descriptor<Byte>::format()) type = NTA_BasicType_Byte;
                else NTA_THROW << "Unexpected data type in the array!  info.format=" << info.format;
                // for info.format codes, see https://docs.python.org/3.7/library/array.html

                if (copy) {
                    new(&m) Array(type);
                    m.allocateBuffer(size);
                    std::memcpy(m.getBuffer(), info.ptr, size*info.itemsize);
                }
                else
                    new (&m) Array(type, info.ptr, info.shape[0]);
            }, py::arg("b"), py::arg("copy")=true)
            .def(py::init<>(),           "Create an empty Array object.")
            .def("zeroBuffer", &Array::zeroBuffer, "Fills array with zeros")
            .def("getCount", &Array::getCount, "Returns the number of elements.")

            // boolean functions to determine the type of value in the Array
            .def("getType", &Array::getType, "Returns an enum representing the data type.")
            .def("isInt32",  [](const Array &self) { return self.getType() == NTA_BasicType_Int32; })
            .def("isUInt32", [](const Array &self) { return self.getType() == NTA_BasicType_UInt32; })
            .def("isInt64",  [](const Array &self) { return self.getType() == NTA_BasicType_Int64; })
            .def("isUInt64", [](const Array &self) { return self.getType() == NTA_BasicType_UInt64; })
            .def("isReal32", [](const Array &self) { return self.getType() == NTA_BasicType_Real32; })
            .def("isReal64", [](const Array &self) { return self.getType() == NTA_BasicType_Real64; })
            .def("isBool",   [](const Array &self) { return self.getType() == NTA_BasicType_Bool; })
            .def("isSDR",    [](const Array &self) { return self.getType() == NTA_BasicType_SDR; })
            .def(py::pickle(
                [](const Array& self) {
                    std::stringstream ss;
                    self.save(ss);
                    return py::bytes(ss.str());
                },
                [](const py::bytes& s) {
                    std::istringstream ss(s);
                    Array self;
                    self.load(ss);
                    return self;
            }))
            .def("__init__", [](Array &m, SDR* sdr, bool copy){
                NTA_CHECK(sdr != NULL) << "SDR pointer is null.";
                if (copy) {
                    new(&m) Array(*sdr); // makes a copy of the SDR
                }
                else {
                  new (&m) Array();
                    m.setBuffer(*sdr);   // Does not copy the SDR.
                }
              }, "Create an Array object from an SDR object.", 
                 py::arg("sdr"), py::arg("copy")=true)
            .def("getSDR", [](Array& self){ return self.getSDR(); 
              },  "Returns an SDR object if the Array contains type SDR.");






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

                // member functions
        py_Link.def("toString", &Link::toString);
        py_Link.def("getDestRegionName", &Link::getDestRegionName);
        py_Link.def("getSrcRegionName",  &Link::getSrcRegionName);
        py_Link.def("getSrcOutputName",  &Link::getSrcOutputName);
        py_Link.def("getDestInputName",  &Link::getDestInputName);
        py_Link.def("getLinkType",       &Link::getLinkType);



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
            .def("setDimensions", &Region::setDimensions)
            .def("getInputDimensions", &Region::getInputDimensions)
            .def("getOutputDimensions", &Region::getOutputDimensions)
            .def("setInputDimensions", &Region::setInputDimensions)
            .def("setOutputDimensions", &Region::setOutputDimensions)
            .def("getOutputElementCount", &Region::getNodeOutputElementCount)
            .def("getInputElementCount", &Region::getNodeInputElementCount)
            .def("askImplForOutputDimensions", &Region::askImplForOutputDimensions)
            .def("askImplForInputDimensions", &Region::askImplForInputDimensions);
                        
        py_Region.def("getInputArray", &Region::getInputData)
            .def("getOutputArray", &Region::getOutputData)
            .def("setInputArray", [](Region& r, const std::string& name, py::buffer& b)
            { 
                py::buffer_info info = b.request();  /* Request a buffer descriptor from Python */
                if (info.ndim != 1)
                    throw std::runtime_error("Expected a one dimensional array!");
                size_t size = static_cast<size_t>(info.shape[0]);
                NTA_BasicType type;
                if      (((info.format == "i") || (info.format == "l") ) && info.itemsize == 4) type = NTA_BasicType_Int32;
                else if (((info.format == "I") || (info.format == "L") ) && info.itemsize == 4) type = NTA_BasicType_UInt32;
                else if ((info.format == "l") || (info.format == "q") ) type = NTA_BasicType_Int64;
                else if ((info.format == "L") || (info.format == "Q") ) type = NTA_BasicType_UInt64;
                else if (info.format == "f") type = NTA_BasicType_Real32;
                else if (info.format == "d") type = NTA_BasicType_Real64;
                else if (info.format == py::format_descriptor<bool>::format()) type = NTA_BasicType_Bool;
                else if (info.format == py::format_descriptor<Byte>::format()) type = NTA_BasicType_Byte;
                else NTA_THROW << "setInputArray(): Unexpected data type in the array!  info.format=" << info.format;
                // for info.format codes, see https://docs.python.org/3.7/library/array.html
                Array s(type, info.ptr, size);
                std::cout << "src: " << s << std::endl;

                r.setInputData(name, s);
            });
            
        py_Region.def(py::pickle(
            [](const Region& self) {
                std::stringstream ss;
                  self.save(ss);
                return py::bytes(ss.str());
            },
            // Note: a de-serialized Region will need to be reattached to a Network
            //           before it could be used.  See Network::addRegion( Region*)
            [](const py::bytes& s) {
                std::istringstream ss(s);
                Region self;
                self.load(ss);
                return self;
        }));


        py_Region.def("getParameterInt32", &Region::getParameterInt32)
            .def("getParameterUInt32", &Region::getParameterUInt32)
            .def("getParameterInt64",  &Region::getParameterInt64)
            .def("getParameterUInt64", &Region::getParameterUInt64)
            .def("getParameterReal32", &Region::getParameterReal32)
            .def("getParameterReal64", &Region::getParameterReal64)
            .def("getParameterBool",   &Region::getParameterBool)
            .def("getParameterString", &Region::getParameterString)
            .def("getParameterArray", &Region::getParameterArray);

        py_Region.def("getParameterArrayCount", &Region::getParameterArrayCount);

        py_Region.def("setParameterInt32", &Region::setParameterInt32)
            .def("setParameterUInt32", &Region::setParameterUInt32)
            .def("setParameterInt64",  &Region::setParameterInt64)
            .def("setParameterUInt64", &Region::setParameterUInt64)
            .def("setParameterReal32", &Region::setParameterReal32)
            .def("setParameterReal64", &Region::setParameterReal64)
            .def("setParameterBool",   &Region::setParameterBool)
            .def("setParameterString", &Region::setParameterString)
            .def("setParameterArray",  &Region::setParameterArray);
                
                
        py_Region.def("executeCommand", [](Region& r, const std::string& command, py::args args)
        {
            // Note: The arguments to executeCommand( ) must be convertable to a string.
            //            The Region implementation must know how to handle the command you are requesting.
            //             So it must implement executeCommand( ) and handle the command string.
            //             For an example, see TestNode, VectorFileSensor and VectorFileEffector regions.
            std::vector<std::string> v_args;
            //std::cout << "command: " << command << std::endl;
            v_args.push_back(command);
            for (size_t i = 0; i < args.size(); ++i)
            {
                auto arg = args[i];
                std::string s = py::str(arg);
                //std::cout << "arg: " << s << std::endl;
                v_args.push_back(s);
            }
            std::string result = r.executeCommand(v_args);
            //std::cout << "result: " << result << std::endl;
            return result;
        });
            
        py_Region.def("__setattr__", [](Region& r, const std::string& Name, py::dict& d)
        {
            //r.python_attributes.insert(std::pair<std::string, py::object>(Name, d));
        });


        py_Region.def("__setattr__", [](Region& r, py::args args)
        {
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


        // TODO: do we need a function like getSelf()?
        //    This is used to allow python apps to access the python written region implimentations
        //    so they can call arbitrary functions on them.
        //    This breaks the plugin API such that apps or regions implemented in C++ or any other
        //    language will not work.  Recommend using getParameter, setParmeter and setParameterArray
        //    or executeCommand to accomplish the same objectives...not very clean but it works cross languages.
        //    Alternative is to add a new feature to accomplish this with more style.
        //py_Region.def("getSelf", [](const Region& self)
        //{
        //    return self.getParameterHandle("self");
        //});



        ///////////////////
        // Serialization
        ///////////////////
        py::enum_<SerializableFormat>(m, "SerializableFormat")
            .value("BINARY", SerializableFormat::BINARY)
            .value("PORTABLE", SerializableFormat::PORTABLE)
            .value("JSON", SerializableFormat::JSON)
            .value("XML", ::SerializableFormat::XML)
            .export_values();
        // NOTE: Registered python regions will be serialized in BINARY (i.e. pickle) regardless of the format specified.


        ///////////////////
        // Network
        ///////////////////

        py::class_<Network> py_Network(m, "Network");

        // constructors
        py_Network.def(py::init<>())
            .def(py::init<std::string>());


        py_Network.def("addRegion", (Region_Ptr_t (htm::Network::*)(
                    const std::string&,
                      const std::string&,
                    const std::string&))
                    &htm::Network::addRegion,
                    "Normal add region."
                    , py::arg("name")
                    , py::arg("nodeType" )
                    , py::arg("nodeParams"));
        py_Network.def("addRegion", (Region_Ptr_t (htm::Network::*)(
                    Region_Ptr_t&))
                    &htm::Network::addRegion,
                    "add region for deserialization."
                    , py::arg("region"));

        py_Network.def("getRegions", &htm::Network::getRegions)
            .def("getRegion",          &htm::Network::getRegion)
            .def("getLinks",           &htm::Network::getLinks)
            .def("getMinPhase",        &htm::Network::getMinPhase)
            .def("getMaxPhase",        &htm::Network::getMaxPhase)
            .def("setMinEnabledPhase", &htm::Network::getMinPhase)
            .def("setMaxEnabledPhase", &htm::Network::getMaxPhase)
            .def("getMinEnabledPhase", &htm::Network::getMinPhase)
            .def("getMaxEnabledPhase", &htm::Network::getMaxPhase)
            .def("setPhases",          &htm::Network::setPhases)
            .def("run",                &htm::Network::run);

        py_Network.def("initialize", &htm::Network::initialize);

        py_Network.def("save",      &htm::Network::save)
            .def("load",            &htm::Network::load)
            .def("saveToFile",      &htm::Network::saveToFile, py::arg("file"), py::arg("fmt") = SerializableFormat::BINARY)
            .def("loadFromFile",    &htm::Network::loadFromFile, py::arg("file"), py::arg("fmt") = SerializableFormat::BINARY)
            .def("__eq__",          &htm::Network::operator==);
            
        py_Network.def(py::pickle(
            [](const Network& self) {
                std::stringstream ss;
                self.save(ss);
                return py::bytes(ss.str());
            },
            [](const py::bytes& s) {
                std::istringstream ss(s);
                Network self;
                self.load(ss);
                return self;  
        }));

        py_Network.def("link", &htm::Network::link
            , "Defines a link between regions"
            , py::arg("srcName"), py::arg("destName")
            , py::arg("linkType") = "", py::arg("linkParams") = ""
            , py::arg("srcOutput") = "", py::arg("destInput") = ""
            , py::arg("propagationDelay") = 0);

        py::enum_<htm::LogLevel>(m, "LogLevel", "An enumeration of logging levels.")
                     .value("None",    htm::LogLevel::LogLevel_None)        // default
                     .value("Minimal", htm::LogLevel::LogLevel_Minimal)
                     .value("Normal",  htm::LogLevel::LogLevel_Normal)
                     .value("Verbose", htm::LogLevel::LogLevel_Verbose)
                     .export_values();
        py_Network.def_static("setLogLevel", &htm::Network::setLogLevel, py::arg("level") = htm::LogLevel::LogLevel_None);
                
                
        // plugin registration
        //     (note: we are re-directing these to static functions on the PyBindRegion class)
        //     (node: the typeName is "py."+className )
        py_Network.def_static("registerPyRegion",
                         [](const std::string& module,
                            const std::string& className) {
                htm::RegisteredRegionImplPy::registerPyRegion(module, className);
            });

        py_Network.def_static("unregisterPyRegion", [](const std::string& typeName) {
                htm::RegisteredRegionImplPy::unregisterPyRegion(typeName);
            });
        py_Network.def_static("cleanup", &htm::Network::cleanup);



        ///////////////////
        // Collection
        ///////////////////

        // Regions
        typedef Collection<std::shared_ptr<Region>> Region_Collection_t;
        py::class_<Region_Collection_t> py_RegionCollection(m, "RegionCollection");
        py_RegionCollection.def("getByName", &Region_Collection_t::getByName);
        py_RegionCollection.def("contains", &Region_Collection_t::contains);
        py_RegionCollection.def("getCount", &Region_Collection_t::getCount);
        py_RegionCollection.def("size", &Region_Collection_t::size);

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
        typedef std::vector<std::shared_ptr<Link>> Links_t;
        py::class_<Links_t> py_LinkCollection(m, "Links_t");
        py_LinkCollection.def("getCount", &Links_t::size);

        // bare bone sequence protocol
        py_LinkCollection.def("__len__", &Links_t::size);
        py_LinkCollection.def("__getitem__", [](Links_t& coll, size_t i)
        {
            if (i >= coll.size())
            {
                throw py::index_error();
            }

            return coll[i];
        });

        // not sure we need __iter__
        py_LinkCollection.def("__iter__", [](Links_t& coll) { return py::make_iterator(coll.begin(), coll.end()); }, py::keep_alive<0, 1>());
    }

} // namespace htm_ext
