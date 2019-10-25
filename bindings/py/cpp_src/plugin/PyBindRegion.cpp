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
Implementation for the PyBindRegion class.  This class acts as the base class for all Python implemented Regions.
In this case, the C++ engine is actually calling into the Python code.
*/

#include "PyBindRegion.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <regex>
#include <vector>

#include <htm/engine/Region.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/BasicType.hpp>
#include <htm/utils/Log.hpp>
#include <htm/os/Path.hpp>


namespace htm
{
using namespace htm;
namespace py = pybind11;

    py::array create_numpy_view(const ArrayBase& a)
    {
        switch (a.getType())
        {
        case NTA_BasicType_Bool: { return py::array({ a.getCount() }, { sizeof(bool) }, (bool*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Byte: { return py::array({ a.getCount() }, { sizeof(Byte) }, (Byte*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Int16: { return py::array({ a.getCount() }, { sizeof(Int16) }, (Int16*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_UInt16: { return py::array({ a.getCount() }, { sizeof(UInt16) }, (UInt16*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Int32: { return py::array({ a.getCount() }, { sizeof(Int32) }, (Int32*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_UInt32: { return py::array({ a.getCount() }, { sizeof(UInt32) }, (UInt32*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Int64: { return py::array({ a.getCount() }, { sizeof(Int64) }, (Int64*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_UInt64: { return py::array({ a.getCount() }, { sizeof(UInt64) }, (UInt64*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Real32: { return py::array({ a.getCount() }, { sizeof(Real32) }, (Real32*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Real64: { return py::array({ a.getCount() }, { sizeof(Real64) }, (Real64*)a.getBuffer(), py::capsule(a.getBuffer())); }

        default:
            throw Exception(__FILE__, __LINE__, "Data type not implemented");

        } // switch
    }
    
    template<class T>
    py::array_t<T> create_matrix(size_t width, T *data_ptr  = nullptr) {
      return py::array_t<T>(
        	py::buffer_info( data_ptr,
		                       sizeof(T), //itemsize
		                       py::format_descriptor<T>::format(),
		                       1, // ndim
		                       std::vector<size_t> { width }, // shape
		                       std::vector<size_t> { sizeof(T)} ));// strides
    }

    // recurseive helper for prepareCreationParams
    static py::object make_args(const Value& vm, NTA_BasicType dataType) {
      //std::cerr << "[" << vm.key() << "]; make_args(\"" << vm << "\", " << BasicType::getName(dataType) << ")\n";
      if (vm.isScalar() ) {
        switch(dataType) {
        case NTA_BasicType_Int16:
        case NTA_BasicType_UInt16:
        case NTA_BasicType_Int32:
        case NTA_BasicType_UInt32:
        case NTA_BasicType_Int64:
        case NTA_BasicType_UInt64:
          return py::int_(vm.as<Int64>());
        case NTA_BasicType_Real32:
        case NTA_BasicType_Real64:
          return py::float_(vm.as<Real64>());
        case NTA_BasicType_Bool:
          return py::bool_(vm.as<bool>());
        case NTA_BasicType_Byte:
          return py::str(vm.str());
        default:   // NTA_BasicType_Last
          // use the format of the data in the string to determine type.
          std::string s = vm.str();
          if (std::regex_match(s, std::regex("^[-+]?[0-9]+$"))) {
            // it is an integer.
            return py::int_(vm.as<Int64>());
          }
          if (std::regex_match(s, std::regex("^[-+]?[0-9]+([.][0-9]+)?$"))) {
            // it is floating point.
            return py::float_(vm.as<double>());
          }
          else if (std::regex_match(s, std::regex("^(true|false|off|on)$", std::regex::icase))) {
            return py::bool_(vm.as<bool>());
          }
          else 
            return py::str(s);
        }
      }
      else if (vm.isMap()) {
        py::kwargs kw;
        for (auto it = vm.begin(); it != vm.end(); ++it)
        {
          std::string key = it->first.c_str();
          Value v = it->second;
          kw[key.c_str()] = make_args(v, NTA_BasicType_Last);  // recursive call
        }
        return std::move(kw);
      }
      else if (vm.isSequence()) {
        switch(dataType) {
          case NTA_BasicType_Int16:
          case NTA_BasicType_UInt16:
          case NTA_BasicType_Int32:
          case NTA_BasicType_UInt32:
          case NTA_BasicType_Int64:
          case NTA_BasicType_UInt64:
          {
            py::array_t<Int64> arr = create_matrix<Int64>(vm.size());
            for (size_t i = 0; i < vm.size(); i++) {
              arr[py::int_(i)] = make_args(vm[i], dataType); // recursive call
            }
            return std::move(arr);
          }
          case NTA_BasicType_Real32:
          case NTA_BasicType_Real64:
          {
            py::array_t<double> arr = create_matrix<double>(vm.size());
            for (size_t i = 0; i < vm.size(); i++) {
              arr[py::int_(i)] = make_args(vm[i], dataType); // recursive call
            }
            return std::move(arr);
          }
          case NTA_BasicType_Bool:
          {
            py::array_t<bool> arr = create_matrix<bool>(vm.size());
            for (size_t i = 0; i < vm.size(); i++) {
              arr[py::int_(i)] = make_args(vm[i], dataType); // recursive call
            }
            return std::move(arr);
          }
          case NTA_BasicType_Byte:
          {
            // return a py::list
            py::list lst;
            for (size_t i = 0; i < vm.size(); i++) {
              if (vm[i].isScalar())
                lst.append(vm[i].str());
              else
                lst.append(vm[i].to_json());
            }
            return std::move(lst);
          }
          default:   // NTA_BasicType_Last
          {
            // return a py::list
            py::list lst;
            for (size_t i = 0; i < vm.size(); i++) {
              lst.append(make_args(vm[i], NTA_BasicType_Last)); // recursive call
            }
            return std::move(lst);
          }
        }
      }
      throw Exception(__FILE__, __LINE__, "Not implemented.");
    }


    // make kwargs from ValueMap.
    // Basically this will construct the kwargs based on the Spec provided by the
    // py implemented region.  For each parameter it uses the type specification in the Spec.
    //  - integer types (Intxx & UIntxx) will be passed as python ints.
    //  - floating point types (Realxx) will be passed as python floats.
    //  - boolean types (Bool) will be passed as a python boolean.
    //  - string types (Byte) will be passed as a python string.
    //  - NTA_BasicType_Last in Spec as type will mean type based on inspecting the value
    //    and can contain a nesting of Map, Sequence, and Scalar formatting.
    // If the ValueMap contains a Sequence for a parameter, numeric types will be 
    // passed as a numpy array.  String types or anything else will be passed as a python list.
    // If the ValueMap conatins a Map for a parameter, the spec is ignored and it is 
    // returned as a python dict (a kwargs) with each element type based on inspecting the value.
    // 
    static void prepareCreationParams(const ValueMap & vm, py::kwargs& kwargs, Spec* ns)
    {
      // Look for parameters that don't belong
      if (vm.isMap()) {
        for (auto p: vm) {
          std::string key = p.first;
          if (key == "dim")
            continue;
          if (!ns->parameters.contains(key))
            NTA_THROW << "Parameter '" << key << "' is not expected for this Region.";
            
          // Prevent parameters with ReadOnlyAccess from being used
          // to initialize a region.  They must have CreateAccess or ReadWriteAccess.
          auto ps = ns->parameters.getByName(key);
          if (ps.accessMode == ParameterSpec::ReadOnlyAccess)
            NTA_THROW << "Parameter '" << key << "' is ReadOnly. Cannot be used for creation.";
        }
      }
      
      // apply defaults and encode into py types
      for (auto p : ns->parameters) {
        std::string key = p.first;
        try {
          ParameterSpec &ps = p.second;
          if (vm.contains(key) && !(vm[key].isScalar() && vm[key].str().empty())) {
            kwargs[key.c_str()] = make_args(vm[key], ps.dataType);
          }
          else if (!ps.defaultValue.empty()) {
              // a missing or empty parameter that has a default value.
              //std::cerr << "  parse default for " << key << ": " << ps.defaultValue << "\n";
              try {
                Value default_value;
                default_value.parse(ps.defaultValue);
                kwargs[key.c_str()] = make_args(default_value, ps.dataType);
              } catch(Exception& e) {
                NTA_THROW << "In default value for " << key << "; " << e.what();
              }
          }
        } 
        catch(Exception& e) {
          NTA_THROW << "Unable to create a Python object for parameter '"
                     << key << ": " << e.what();
        }
        catch(std::exception& e) {
          NTA_THROW << "Unable to create a Python object for parameter '"
                     << key << ": " << e.what();
        }
      }
    };

    PyBindRegion::PyBindRegion(const char * module, const ValueMap & nodeParams, Region * region, const char* className)
        : RegionImpl(region)
        , module_(module)
        , className_(className)
    {
        NTA_CHECK(region != NULL);

        std::string realClassName(className);
        if (realClassName.empty())
        {
            realClassName = Path::getExtension(module_);
        }
        //std::cerr << "calling creation of " << module_ << "\n";

        // Make a local copy of the Spec
        createSpec(module_.c_str(), nodeSpec_, className_.c_str());
        //std::cerr << nodeSpec_ << "\n";    

        // Prepare the creation params as a tuple of PyObject pointers
        py::args args;
        py::kwargs kwargs;

        //std::cerr << "nodeParams: " << nodeParams << "\n";
        try {
            prepareCreationParams(nodeParams, kwargs, &nodeSpec_);
        }
        catch(Exception &e) {
            NTA_THROW << "Python region: " << module_ << "; " << e.what();
        }
        // std::cerr << "calling arguments: " << kwargs << "\n";
        // Instantiate a node and assign it  to the node_ member
        // node_.assign(py::Instance(module_, realClassName, args, kwargs));
        node_ = py::module::import(module_.c_str()).attr(realClassName.c_str())(*args, **kwargs);
        NTA_CHECK(node_);
        //std::cerr << "return from creation\n";

    }

    PyBindRegion::PyBindRegion(const char* module, ArWrapper& wrapper, Region * region, const char* className)
        : RegionImpl(region)
        , module_(module)
        , className_(className)

    {

        cereal_adapter_load(wrapper);
    }

    PyBindRegion::~PyBindRegion()
    {
    }

    std::string PyBindRegion::pickleSerialize() const
    {
        // 1. serialize main state using pickle
        // 2. call class method to serialize external state

        // 1. Serialize main state of the Python module
				//    We want this to end up in the open stream obtained from bundle.
				//    a. We first pickle the python into a temporary file.
				//    b. copy the file into our open stream.

				std::string tmp_pickle = "pickle.tmp";
		    py::tuple args = py::make_tuple(tmp_pickle, "wb");
		    auto f = py::module::import("__builtin__").attr("file")(*args);

#if PY_MAJOR_VERSION >= 3
		    auto pickle = py::module::import("pickle");
#else
		    auto pickle = py::module::import("cPickle");
#endif
		    args = py::make_tuple(node_, f, 2);   // use type 2 protocol
		    pickle.attr("dump")(*args);
		    pickle.attr("close")();

				// copy the pickle into the out string
				std::ifstream pfile(tmp_pickle.c_str(), std::ios::binary);
				std::string content((std::istreambuf_iterator<char>(pfile)),
				                     std::istreambuf_iterator<char>());
				pfile.close();
		 		Path::remove(tmp_pickle);
		    return content;
    }
    std::string PyBindRegion::extraSerialize() const
    {
		    std::string tmp_extra = "extra.tmp";

        // 2. External state
        // Call the Python serializeExtraData() method to write additional data.

        py::tuple args = py::make_tuple(tmp_extra);
        // Need to put the None result in py::Ptr to decrement the ref count
        node_.attr("serializeExtraData")(*args);

				// copy the extra data into the extra string
				std::ifstream efile(tmp_extra.c_str(), std::ios::binary);
				std::string extra((std::istreambuf_iterator<char>(efile)),
				                   std::istreambuf_iterator<char>());
				efile.close();
				Path::remove(tmp_extra);
		    return extra;

    }

		void PyBindRegion::pickleDeserialize(std::string p) {
        // 1. deserialize main state using pickle
        // 2. call class method to deserialize external state

				std::ofstream des;
				std::string tmp_pickle = "pickle.tmp";


			  std::ofstream pfile(tmp_pickle.c_str(), std::ios::binary);
				pfile.write(p.c_str(), p.size());
				pfile.close();


		// Tell Python to un-pickle using what is now in the pickle.tmp file.
        py::args args = py::make_tuple(tmp_pickle, "rb");
        auto f = py::module::import("__builtin__").attr("file")(*args);

#if PY_MAJOR_VERSION >= 3
        auto pickle = py::module::import("pickle");
#else
        auto pickle = py::module::import("cPickle");
#endif

        args = py::make_tuple(node_, f);
        pickle.attr("load")(*args);

        pickle.attr("close")();
				Path::remove(tmp_pickle);
		}

		void PyBindRegion::extraDeserialize(std::string e) {
        // 2. External state
		    std::string tmp_extra = "extra.tmp";
			  std::ofstream efile(tmp_extra.c_str(), std::ios::binary);
				efile.write(e.c_str(), e.size());
				efile.close();

        // Call the Python deSerializeExtraData() method
        py::tuple args = py::make_tuple(tmp_extra);
        node_.attr("deSerializeExtraData")(*args);
				Path::remove(tmp_extra);
    }





    template<typename T>
    T PyBindRegion::getParameterT(const std::string & name, Int64 index)
    {
        try
        {
            py::args args = py::make_tuple(name, index);
            return node_.attr("getParameter")(*args).cast<T>();
        }
        catch (const py::error_already_set& e)
        {
            std::cout << e.what() << std::endl;
            throw Exception(__FILE__, __LINE__, e.what());
        }
    }

    template <typename T>
    void PyBindRegion::setParameterT(const std::string & name, Int64 index, T value)
    {
        NTA_CHECK(nodeSpec_.parameters.contains(name)) 
               << "module " << module_ << "; Parameter '" << name 
               << "' is not known. Cannot be set.";
        auto ps = nodeSpec_.parameters.getByName(name);
        NTA_CHECK(ps.accessMode == ParameterSpec::ReadWriteAccess) 
               << "module " << module_ << "; Parameter '" << name 
               << "' does not have ReadWriteAccess. Cannot be set.";

        try
        {
            py::args args = py::make_tuple(name, index, value);
            node_.attr("setParameter")(*args);
        }
        catch (const py::error_already_set& e)
        {
            std::cout << e.what() << std::endl;
            throw Exception(__FILE__, __LINE__, e.what());
        }
    }

    bool PyBindRegion::getParameterBool(const std::string& name, Int64 index)
    {
        return getParameterT<bool>(name, index);
    }

    Byte PyBindRegion::getParameterByte(const std::string& name, Int64 index)
    {
        return getParameterT<Byte>(name, index);
    }

    Int32 PyBindRegion::getParameterInt32(const std::string& name, Int64 index)
    {
        return getParameterT<Int32>(name, index);
    }

    UInt32 PyBindRegion::getParameterUInt32(const std::string& name, Int64 index)
    {
        return getParameterT<UInt32>(name, index);
    }

    Int64 PyBindRegion::getParameterInt64(const std::string& name, Int64 index)
    {
        return getParameterT<Int64>(name, index);
    }

    UInt64 PyBindRegion::getParameterUInt64(const std::string& name, Int64 index)
    {
        return getParameterT<UInt64>(name, index);
    }

    Real32 PyBindRegion::getParameterReal32(const std::string& name, Int64 index)
    {
        return getParameterT<Real32>(name, index);
    }

    Real64 PyBindRegion::getParameterReal64(const std::string& name, Int64 index)
    {
        return getParameterT<Real64>(name, index);
    }



    void PyBindRegion::setParameterBool(const std::string& name, Int64 index, bool value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterByte(const std::string& name, Int64 index, Byte value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterInt32(const std::string& name, Int64 index, Int32 value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterUInt32(const std::string& name, Int64 index, UInt32 value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterInt64(const std::string& name, Int64 index, Int64 value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterUInt64(const std::string& name, Int64 index, UInt64 value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterReal32(const std::string& name, Int64 index, Real32 value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::setParameterReal64(const std::string& name, Int64 index, Real64 value)
    {
        setParameterT(name, index, value);
    }

    void PyBindRegion::getParameterArray(const std::string& name, Int64 index, Array & a)
    {
        auto args = py::make_tuple(name, index, create_numpy_view(a));
        node_.attr("getParameterArray")(*args);
    }

    void PyBindRegion::setParameterArray(const std::string& name, Int64 index, const Array & a)
    {
        auto args = py::make_tuple(name, index, create_numpy_view(a));
        node_.attr("setParameterArray")(*args);
    }

    std::string PyBindRegion::getParameterString(const std::string& name, Int64 index)
    {
        py::args args = py::make_tuple(name, index);
        return node_.attr("setParameter")(*args).cast<std::string>();
    }

    void PyBindRegion::setParameterString(const std::string& name, Int64 index, const std::string& value)
    {
        py::args args = py::make_tuple(name, index, value);
        node_.attr("setParameter")(*args);
    }



    size_t PyBindRegion::getParameterArrayCount(const std::string& name, Int64 index)
    {
        py::args args = py::make_tuple(name, index);
        return node_.attr("getParameterArrayCount")(*args).cast<size_t>();
    }
    
    
    
    
    

    size_t PyBindRegion::getNodeOutputElementCount(const std::string& outputName) const
    {
        py::args args = py::make_tuple(outputName);
        return (size_t)node_.attr("getOutputElementCount")(*args).cast<int>();
    }
    
    
    

    std::string PyBindRegion::executeCommand(const std::vector<std::string>& args, Int64 index)
    {
        //py::Tuple t(args.size() - 1);
        //for (size_t i = 1; i < args.size(); ++i)
        //{
        //    py::String s(args[i]);
        //    t.setItem(i - 1, s);
        //}
        std::vector<std::string> t(args.begin() + 1, args.end());

        py::args commandArgs = py::make_tuple(args[0], t);
        auto result = node_.attr("executeMethod")(*commandArgs);

        auto s = result.attr("__str__")().cast<std::string>();
        return s;
    }

    void PyBindRegion::compute()
    {
        const Spec& ns = nodeSpec_;

        // Prepare the inputs dict
        py::dict inputs;
        for (size_t i = 0; i < ns.inputs.getCount(); ++i)
        {
            const std::pair<std::string, InputSpec> & p = ns.inputs.getByIndex(i);

            // Get the corresponding input buffer
            auto inp = region_->getInput(p.first);
            NTA_CHECK(inp);

            // Set pa to point to the original input array
            const htm::Array * pa = &(inp->getData());

            // Skip unlinked inputs of size 0
    		    if (pa->getCount() == 0)
                continue;


            // Create a numpy array from pa, which wil be either
            // the original input array or a stored input array copy
            // (if a splitter map is needed)
            inputs[p.first.c_str()] = create_numpy_view(*pa);
        }

        // Prepare the outputs dict
        py::dict outputs;

        for (size_t i = 0; i < ns.outputs.getCount(); ++i)
        {
            // Get the current OutputSpec object
            const std::pair<std::string, OutputSpec> & p = ns.outputs.getByIndex(i);

            // Get the corresponding output buffer
            Output * out = region_->getOutput(p.first);
            // Skip optional outputs
            if (!out)
                continue;

            const Array & data = out->getData();
            outputs[p.first.c_str()] = create_numpy_view(data);
        }

        py::args args = py::make_tuple(inputs, outputs);
        node_.attr("guardedCompute")(*args);
    }


    //
    // Get the node spec from the underlying Python node
    // and populate the provided node spec object.
    // There is a local copy (used by compute()), and RegisteredRegionImplPy.hpp caches a copy.
    //
    void PyBindRegion::createSpec(const char * module, Spec& ns, const char* className)
    {
    
        std::string realClassName(className);
        if (realClassName.empty())
        {
            realClassName = Path::getExtension(module);
        }
        //std::cerr << "createSpec for " << std::string(module) << "." << realClassName << "\n";

        try
        {
            auto m = py::module::import(module);
            auto pyClass = m.attr(realClassName.c_str());

            auto pyNodeSpec = pyClass.attr("getSpec")();
            ns.description = pyNodeSpec["description"].cast<std::string>();

            if (pyNodeSpec.contains("inputs"))
            {
                auto inputs = pyNodeSpec["inputs"];

                // Add inputs
                for (auto it = inputs.begin(); it != inputs.end(); ++it)
                {
                    auto name = it->cast<std::string>();
                    auto input = inputs[*it];

                    // Add an InputSpec object for each input spec dict
                    std::ostringstream inputMessagePrefix;
                    inputMessagePrefix << "Region " << realClassName
                        << " spec has missing key for input section " << name << ": ";

                    NTA_ASSERT(input.contains("description"))
                        << inputMessagePrefix.str() << "description";
                    auto description = input["description"].cast<std::string>();

                    NTA_ASSERT(input.contains("dataType"))
                        << inputMessagePrefix.str() << "dataType";
                    auto dt = input["dataType"].cast<std::string>();

                    NTA_BasicType dataType;
                    try {
                        dataType = BasicType::parse(dt);
                    }
                    catch (Exception &) {
                        std::stringstream stream;
                        stream << "Invalid 'dataType' specificed for input '" << name
                            << "' when getting spec for region '" << realClassName << "'.";
                        throw Exception(__FILE__, __LINE__, stream.str());
                    }

                    NTA_ASSERT(input.contains("count")) << inputMessagePrefix.str() << "count";
                    auto count = input["count"].cast<UInt32>();

										bool required = false;
                    if (input.contains("required"))
										{
                    	required = input["required"].cast<bool>();
										}

                    // make regionLevel optional and default to true.
                    bool regionLevel = true;
                    if (input.contains("regionLevel"))
                    {
                        regionLevel = input["regionLevel"].cast<bool>();
                    }

										bool isDefaultInput = false;
                    if (input.contains("isDefaultInput"))
                    {
                      isDefaultInput = input["isDefaultInput"].cast<bool>();
										}

                    ns.inputs.add(
                        name,
                        InputSpec(
                            description,
                            dataType,
                            count,
                            required,
                            regionLevel,
                            isDefaultInput )
                    );
                }
            }

            if (pyNodeSpec.contains("outputs"))
            {
                auto outputs = pyNodeSpec["outputs"];

                // Add outputs
                for (auto it = outputs.begin(); it != outputs.end(); ++it)
                {
                    auto name = it->cast<std::string>();
                    auto output = outputs[*it];

                    // Add an OutputSpec object for each output spec dict
                    std::ostringstream outputMessagePrefix;
                    outputMessagePrefix << "Region " << realClassName
                        << " spec has missing key for output section " << name << ": ";

                    NTA_ASSERT(output.contains("description"))
                        << outputMessagePrefix.str() << "description";
                    auto description = output["description"].cast<std::string>();

                    NTA_ASSERT(output.contains("dataType"))
                        << outputMessagePrefix.str() << "dataType";
                    auto dt = output["dataType"].cast<std::string>();
                    NTA_BasicType dataType;
                    try {
                        dataType = BasicType::parse(dt);
                    }
                    catch (Exception &) {
                        std::stringstream stream;
                        stream << "Invalid 'dataType' specificed for output '" << name
                            << "' when getting spec for region '" << realClassName << "'.";
                        throw Exception(__FILE__, __LINE__, stream.str());
                    }

                    NTA_ASSERT(output.contains("count"))
                        << outputMessagePrefix.str() << "count";
                    auto count = output["count"].cast<UInt32>();

                    // make regionLevel optional and default to true.
                    bool regionLevel = true;
                    if (output.contains("regionLevel"))
                    {
                        regionLevel = output["regionLevel"].cast<bool>();
                    }
										bool isDefaultOutput = false;
										if (output.contains("isDefaultOutput"))
										{
                    	isDefaultOutput = output["isDefaultOutput"].cast<bool>();
										}

                    ns.outputs.add(
                        name,
                        OutputSpec(
                            description,
                            dataType,
                            count,
                            regionLevel,
                            isDefaultOutput)
                    );
                }
            }

            if (pyNodeSpec.contains("parameters"))
            {
                auto parameters = pyNodeSpec["parameters"];

                // Add parameters
                for (auto it = parameters.begin(); it != parameters.end(); ++it)
                {
                    std::string name;
                    std::string description;
                    UInt32 count;
                    std::string constraints;
                    ParameterSpec::AccessMode accessMode;
                    std::string defaultValue;
                    
                    name = std::string(py::str(*it));
                    
                    auto parameter = parameters[*it];
                    // Add an ParameterSpec object for each output spec dict
                    std::string parameterMessagePrefix = " parameter " + name + ": ";

                    NTA_ASSERT(parameter.contains("description"))
                        << parameterMessagePrefix << "missing description";
                    try {
                      description = parameter["description"].cast<std::string>();
                    }
                    catch (const py::cast_error& e) {
                        NTA_THROW << parameterMessagePrefix << ":description; " << e.what();
                    }

                    NTA_ASSERT(parameter.contains("dataType"))
                        << parameterMessagePrefix << "missing dataType";
                    NTA_BasicType dataType;
                    try {
                        auto dt = parameter["dataType"].cast<std::string>();
                        dataType = BasicType::parse(dt);
                    }
                    catch (const py::cast_error& e) {
                      NTA_THROW << parameterMessagePrefix << "dataType " << e.what();
                    }
                    catch (Exception &e) {
                      NTA_THROW << parameterMessagePrefix << "dataType " << e.what();
                    }

                    NTA_ASSERT(parameter.contains("count"))
                        << parameterMessagePrefix << "missing count";
                    try {
                        count = (UInt32)parameter["count"].cast<int>();
                    }
                    catch (const py::cast_error& e) {
                      NTA_THROW << parameterMessagePrefix << "dataType " << e.what();
                    }

                    // This parameter is optional
                    if (parameter.contains("constraints")) {
                      try {
                        constraints = parameter["constraints"].cast<std::string>();
                      }
                      catch (const py::cast_error& e) {
                        NTA_THROW << parameterMessagePrefix << "constraints " << e.what();
                      }
                    }

                    NTA_ASSERT(parameter.contains("accessMode"))
                        << parameterMessagePrefix << "missing accessMode";
                    std::string am;
                    try {
                      am = parameter["accessMode"].cast<std::string>();
                    }
                    catch (const py::cast_error& e) {
                      NTA_THROW << parameterMessagePrefix << "accessMode " << e.what();
                    }                      
                    if (am == "Create" || am == "CreateAccess")
                        accessMode = ParameterSpec::CreateAccess;
                    else if (am == "Read" || am == "ReadOnly" || am == "ReadOnlyAccess")
                        accessMode = ParameterSpec::ReadOnlyAccess;
                    else if (am == "ReadWrite" || am == "ReadWriteAccess")
                        accessMode = ParameterSpec::ReadWriteAccess;
                    else
                        NTA_THROW << parameterMessagePrefix << "Invalid access mode: " << am;

                    // Get default value as a string
                    if (parameter.contains("defaultValue")) {
                      try {
                          defaultValue = std::string(py::str(parameter["defaultValue"]));
                      } catch (const py::cast_error& e) {
                        NTA_THROW << parameterMessagePrefix << "defaultValue " << e.what();
                      }
                        
                    }
                    if (defaultValue == "None")
                        defaultValue = "";

                    ns.parameters.add(
                        name,
                        ParameterSpec(
                            description,
                            dataType,
                            count,
                            constraints,
                            defaultValue,
                            accessMode));
                }
            }

            if (pyNodeSpec.contains("commands"))
            {
                auto commands = pyNodeSpec["commands"];

                // Add commands
                for (auto it = commands.begin(); it != commands.end(); ++it)
                {
                    auto name = it->cast<std::string>();
                    auto command = commands[*it];

                    std::ostringstream commandsMessagePrefix;
                    commandsMessagePrefix << "Region " << realClassName
                        << " spec has missing key for commands section " << name << ": ";

                    NTA_ASSERT(command.contains("description"))
                        << commandsMessagePrefix.str() << "description";
                    auto description = command["description"].cast<std::string>();

                    ns.commands.add(
                        name,
                        CommandSpec(description));
                }
            }
        }
        catch (const py::error_already_set& e) {
            NTA_THROW << "createSpec() Region: " << module << " " << e.what();
        }
        catch (const py::cast_error& e) {
            NTA_THROW << "createSpec() Region: " << module << " " << e.what();
        }
        catch (std::exception& e) {
            NTA_THROW << "createSpec() Region: " << module << " " << e.what();
        }
        catch (...) {
            NTA_THROW << "createSpec()  Region: " << module << " Unknown error.";
        }
    }

    void PyBindRegion::initialize()
    {
        node_.attr("initialize")();
    }


} // namespace htm

