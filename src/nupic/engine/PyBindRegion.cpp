#include "PyBindRegion.hpp"

#include <map>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/engine/Spec.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/ntypes/ObjectModel.hpp> // IWrite/ReadBuffer
#include <nupic/ntypes/Value.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/os/Path.hpp>

using namespace nupic;
namespace py = pybind11;


extern "C"
{
    bool Initialized = false;

    // NTA_initPython() must be called by the MultinodeFactory before any call to
    // NTA_createPyNode()
    void PyBindRegion::NTA_initPython()
    {
        try
        {
            if (Py_IsInitialized() == false)
            {
                Initialized = true;
                py::initialize_interpreter();
            }
        }
        catch (const std::exception& e)
        {
            throw Exception(__FILE__, __LINE__, e.what());
        }
    }

    // NTA_finalizePython() must be called before unloading the pynode dynamic library
    // to ensure proper cleanup.
    void PyBindRegion::NTA_finalizePython()
    {
        if (Initialized)
        {
            py::finalize_interpreter();
        }
    }

    // createPyNode is used by the MultinodeFactory to create a C++ PyNode instance
    // That references a Python instance. The function tries to create a NuPIC 2.0
    // Py node first and if it fails it tries to create a NuPIC 1.x Py node
    void * PyBindRegion::NTA_createPyNode(const char * module, void * nodeParams,
        void * region, void ** exception, const char* className)
    {
        try
        {
            NTA_CHECK(nodeParams != NULL);
            NTA_CHECK(region != NULL);

            ValueMap * valueMap = static_cast<nupic::ValueMap *>(nodeParams);
            Region * r = static_cast<nupic::Region*>(region);
            RegionImpl * p = NULL;
            p = new nupic::PyBindRegion(module, *valueMap, r, className);

            return p;
        }
        catch (const py::error_already_set& e)
        {
            throw Exception(__FILE__, __LINE__, e.what());
        }
        catch (nupic::Exception & e)
        {
            *exception = new nupic::Exception(e);
            return NULL;
        }
        catch (...)
        {
            return NULL;
        }
    }

    // deserializePyNode is used by the MultinodeFactory to create a C++ PyNode instance
    // that references a Python instance which has been deserialized from saved state
    void * PyBindRegion::NTA_deserializePyNode(const char * module, void * bundle,
        void * region, void ** exception, const char* className)
    {
        try
        {
            NTA_CHECK(region != NULL);

            Region * r = static_cast<nupic::Region*>(region);
            BundleIO *b = static_cast<nupic::BundleIO*>(bundle);
            RegionImpl * p = NULL;
            p = new PyBindRegion(module, *b, r, className);
            return p;
        }
        catch (nupic::Exception & e)
        {
            *exception = new nupic::Exception(e);
            return NULL;
        }
        catch (...)
        {
            return NULL;
        }
    }

    void * PyBindRegion::NTA_deserializePyNodeProto(const char * module, void * proto,
        void * region, void ** exception, const char* className)
    {
        throw std::runtime_error("Not implemented");

        //try
        //{
        //    NTA_CHECK(region != NULL);

        //    Region * r = static_cast<nupic::Region*>(region);
        //    capnp::AnyPointer::Reader *c = static_cast<capnp::AnyPointer::Reader*>(proto);
        //    RegionImpl * p = NULL;
        //    p = new PyBindRegion(module, *c, r, className);
        //    return p;
        //}
        //catch (nupic::Exception & e)
        //{
        //    *exception = new nupic::Exception(e);
        //    return NULL;
        //}
        //catch (...)
        //{
        //    return NULL;
        //}
    }

    // getLastError() returns the last error message
    const char * PyBindRegion::NTA_getLastError()
    {
        return last_error.c_str();
    }

    // createSpec is used by the RegionImplFactory to get the node spec
    // and cache it. It is a static function so there is no need to instantiate
    // a dummy node, just to get its node spec.
    void * PyBindRegion::NTA_createSpec(const char * nodeType, void ** exception, const char* className)
    {
        try
        {
            return PyBindRegion::createSpec(nodeType, className);
        }
        catch (nupic::Exception & e)
        {
            NTA_WARN << "PyRegion::createSpec failed: " << exception;

            *exception = new nupic::Exception(e);
            return NULL;
        }
        catch (...)
        {
            return NULL;
        }
    }

    // destroySpec is used by the RegionImplFactory to destroy
    // a cached node spec.
    int PyBindRegion::NTA_destroySpec(const char * nodeType, const char* className)
    {
        try
        {
            PyBindRegion::destroySpec(nodeType, className);
            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }
}

// This map stores the node specs for all the Python nodes
std::map<std::string, Spec> PyBindRegion::specs_;
std::string PyBindRegion::last_error;

//
// Get the node spec from the underlying Python node
// and populate a dynamically C++ node spec object.
// Return the node spec pointer (that will be owned
// by RegionImplFactory.
//
Spec * PyBindRegion::createSpec(const char * nodeType, const char* className)
{
    // If the node spec for a node type is requested more than once
    // return the exisiting one from the map.
    std::string name(nodeType);
    std::string realClassName(className);
    name = name + ".";
    if (!realClassName.empty())
    {
        name = name + realClassName;
    }

    if (specs_.find(name) != specs_.end())
    {
        Spec & ns = specs_[name];
        return &ns;
    }

    Spec ns;
    createSpec(nodeType, ns, className);

    specs_[name] = ns;
    //NTA_DEBUG << "node type: " << nodeType << std::endl;
    //NTA_DEBUG << specs_[name].toString() << std::endl;
    return &specs_[name];
}


void PyBindRegion::destroySpec(const char * nodeType, const char* className)
{
    std::string name(nodeType);
    std::string realClassName(className);
    name = name + ".";
    if (!realClassName.empty())
    {
        name = name + realClassName;
    }

    specs_.erase(name);
}


namespace nupic
{
    py::array create_numpy_view(const ArrayBase& a)
    {
        switch (a.getType())
        {
        case NTA_BasicType_Byte: { return py::array({ a.getCount() }, { sizeof(NTA_Byte) }, (NTA_Byte*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Int16: { return py::array({ a.getCount() }, { sizeof(NTA_Int16) }, (NTA_Int16*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_UInt16: { return py::array({ a.getCount() }, { sizeof(NTA_UInt16) }, (NTA_UInt16*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Int32: { return py::array({ a.getCount() }, { sizeof(NTA_Int32) }, (NTA_Int32*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_UInt32: { return py::array({ a.getCount() }, { sizeof(NTA_UInt32) }, (NTA_UInt32*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Int64: { return py::array({ a.getCount() }, { sizeof(NTA_Int64) }, (NTA_Int64*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_UInt64: { return py::array({ a.getCount() }, { sizeof(NTA_UInt64) }, (NTA_UInt64*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Real32: { return py::array({ a.getCount() }, { sizeof(NTA_Real32) }, (NTA_Real32*)a.getBuffer(), py::capsule(a.getBuffer())); }
        case NTA_BasicType_Real64: { return py::array({ a.getCount() }, { sizeof(NTA_Real64) }, (NTA_Real64*)a.getBuffer(), py::capsule(a.getBuffer())); }

        default:
        {
            throw Exception(__FILE__, __LINE__, "Not implemented");

            break;
        }

        } // switch
    }

    static void prepareCreationParams(const ValueMap & vm, py::kwargs& kwargs)
    {
        if (std::distance(vm.begin(), vm.end()) > 1)
        {
            int pp = 9;
        }

        ValueMap::const_iterator it;
        for (it = vm.begin(); it != vm.end(); ++it)
        {
            try
            {
                auto key = it->first.c_str();
                
                auto value = it->second;
                if (value->isScalar())
                {
                    auto s = value->getScalar();
                    switch (s->getType())
                    {
                        case NTA_BasicType_Bool: { kwargs[key] = s->getValue<bool>(); break; }
                        case NTA_BasicType_Byte: { kwargs[key] = s->getValue<NTA_Byte>(); break; }
                        case NTA_BasicType_Int16: { kwargs[key] = s->getValue<NTA_Int16>(); break; }
                        case NTA_BasicType_UInt16: { kwargs[key] = s->getValue<NTA_UInt16>(); break; }
                        case NTA_BasicType_Int32: { kwargs[key] = s->getValue<NTA_Int32>(); break; }
                        case NTA_BasicType_UInt32: { kwargs[key] = s->getValue<NTA_UInt32>(); break; }
                        case NTA_BasicType_Int64: { kwargs[key] = s->getValue<NTA_Int64>(); break; }
                        case NTA_BasicType_UInt64: { kwargs[key] = s->getValue<NTA_UInt64>(); break; }
                        case NTA_BasicType_Real32: { kwargs[key] = s->getValue<NTA_Real32>(); break; }
                        case NTA_BasicType_Real64: { kwargs[key] = s->getValue<NTA_Real64>(); break; }

                        default:
                            NTA_THROW << "Invalid type: " << s->getType();
                    }
                }
                else if(value->isString())
                {
                    kwargs[key] = value->getString().get();
                }
                else if (value->isArray())
                {
                    auto a = value->getArray();
                    kwargs[key] = create_numpy_view(*a.get());
                    
                }
                else
                {
                    throw Exception(__FILE__, __LINE__, "Not implemented.");
                }
            }
            catch (Exception& e) {
                NTA_THROW << "Unable to create a Python object for parameter '"
                    << it->first << ": " << e.what();
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

        // Prepare the creation params as a tuple of PyObject pointers
        py::args args;
        py::kwargs kwargs;

        prepareCreationParams(nodeParams, kwargs);

        // Instantiate a node and assign it  to the node_ member
        // node_.assign(py::Instance(module_, realClassName, args, kwargs));
        node_ = py::module::import(module_.c_str()).attr(realClassName.c_str())(*args, **kwargs);
        NTA_CHECK(node_);
    }

    PyBindRegion::PyBindRegion(const char* module, BundleIO& bundle, Region * region, const char* className)
        : RegionImpl(region)
        , module_(module)
        , className_(className)

    {
        deserialize(bundle);
        // XXX ADD CHECK TO MAKE SURE THE TYPE MATCHES!
    }

    PyBindRegion::~PyBindRegion()
    {
        for (std::map<std::string, Array*>::iterator i = inputArrays_.begin();
            i != inputArrays_.end();
            i++)
        {
            delete i->second;
            i->second = NULL;
        }
    }

    void PyBindRegion::serialize(BundleIO& bundle)
    {
        // 1. serialize main state using pickle
        // 2. call class method to serialize external state

        // 1. Serialize main state

        // f = open(path, "wb")
        std::string path = bundle.getPath("pkl");
        py::tuple args = py::make_tuple(path, "wb");

        auto f = py::module::import("__builtin__").attr("file")(*args);

#if PY_MAJOR_VERSION >= 3
        auto pickle = py::module::import("pickle");
#else
        auto pickle = py::module::import("cPickle");
#endif
        args = py::make_tuple(node_, f, 2);
        pickle.attr("dump")(*args);

        pickle.attr("close")();

        // 2. External state
        // Call the Python serializeExtraData() method
        std::string externalPath = bundle.getPath("xtra");
        args = py::make_tuple(externalPath);
        
        // Need to put the None result in py::Ptr to decrement the ref count
        node_.attr("serializeExtraData")(*args);
    }

    void PyBindRegion::deserialize(BundleIO& bundle)
    {
        // 1. deserialize main state using pickle
        // 2. call class method to deserialize external state

        // 1. de-serialize main state using pickle
        // f = open(path, "rb")  # binary mode needed on windows
        std::string path = bundle.getPath("pkl");
        py::args args = py::make_tuple(path, "rb");

        auto f = py::module::import("__builtin__").attr("file")(*args);

#if PY_MAJOR_VERSION >= 3
        auto pickle = py::module::import("pickle");
#else
        auto pickle = py::module::import("cPickle");
#endif

        args = py::make_tuple(node_, f);
        pickle.attr("load")(*args);

        pickle.attr("close")();

        // 2. External state
        // Call the Python deSerializeExtraData() method
        std::string externalPath = bundle.getPath("xtra");
        args = py::make_tuple(externalPath);

        node_.attr("deSerializeExtraData")(*args);
    }

    const Spec & PyBindRegion::getSpec()
    {
        return *(PyBindRegion::createSpec(module_.c_str(), className_.c_str()));
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

    pybind11::object PyBindRegion::getParameterHandle(const std::string& name, Int64 index)
    {
        if (name == "self")
        {
            return node_;
        }
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

    void PyBindRegion::setParameterHandle(const std::string& name, Int64 index, Handle value)
    {
        throw std::runtime_error("Not implemented");
    }

    void PyBindRegion::getParameterArray(const std::string& name, Int64 index, Array & a)
    {
        auto args = py::make_tuple(name, index, create_numpy_view(a));
        node_.attr("getParameterArray")(*args);

        //py::array_t<Int64> aa({ a.getCount() }, { sizeof(Int64) }, (Int64*)a.getBuffer());
        //assert(a.getBuffer() == aa.request().ptr);

        //auto args = py::make_tuple(name, index, aa);
        //node_.attr("getParameterArray")(*args);

        //assert(a.getBuffer() == aa.request().ptr);
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


    void PyBindRegion::getParameterFromBuffer(const std::string& name, Int64 index, IWriteBuffer& value)
    {
        // we override getParameterX for every type, so this should never
        // be called
        NTA_THROW << "::getParameterFromBuffer should not have been called";
    }

    void PyBindRegion::setParameterFromBuffer(const std::string& name, Int64 index, IReadBuffer& value)
    {
        // we override getParameterX for every type, so this should never
        // be called
        NTA_THROW << "::setParameterFromBuffer should not have been called";
    }

    size_t PyBindRegion::getParameterArrayCount(const std::string& name, Int64 index)
    {
        py::args args = py::make_tuple(name, index);
        return node_.attr("getParameterArrayCount")(*args).cast<size_t>();
    }

    size_t PyBindRegion::getNodeOutputElementCount(const std::string& outputName)
    {
        py::args args = py::make_tuple(outputName);
        return node_.attr("getOutputElementCount")(*args).cast<size_t>();
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
        const Spec & ns = getSpec();

        // Prepare the inputs dict
        py::dict inputs;
        for (size_t i = 0; i < ns.inputs.getCount(); ++i)
        {
            const std::pair<std::string, InputSpec> & p = ns.inputs.getByIndex(i);

            // Get the corresponding input buffer
            auto inp = region_->getInput(p.first);
            NTA_CHECK(inp);

            // Set pa to point to the original input array
            const Array * pa = &(inp->getData());

            // Skip unlinked inputs of size 0
            if (pa->getCount() == 0)
                continue;

            // If the input requires a splitter map then
            // Copy the original input array to the stored input array, which is larger
            // by one element and put 0 in the extra element. This is needed for splitter map
            // access.
            if (p.second.requireSplitterMap)
            {
                // Verify that this input has a stored input array
                NTA_ASSERT(inputArrays_.find(p.first) != inputArrays_.end());
                Array & a = *(inputArrays_[p.first]);

                // Verify that the stored input array is larger by 1  then the original input
                NTA_ASSERT(a.getCount() == pa->getCount() + 1);

                // Work at the char * level because there is no good way
                // to work with the actual data type of the input (since the buffer is void *)
                size_t itemSize = BasicType::getSize(p.second.dataType);
                char * begin1 = (char *)pa->getBuffer();
                char * end1 = begin1 + pa->getCount() * itemSize;
                char * begin2 = (char *)a.getBuffer();
                char * end2 = begin2 + a.getCount() * itemSize;
            
                // Copy the original input array to the stored array
                std::copy(begin1, end1, begin2);

                // Put 0 in the last item (the sentinel value)
                std::fill(end2 - itemSize, end2, 0);

                // Change pa to point to the stored input array (with the sentinel)
                pa = &a;
            }

            // Create a numpy array from pa, which wil be either
            // the original input array or a stored input array
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
    // and populate a dynamically C++ node spec object.
    // Return the node spec pointer (that will be owned
    // by RegionImplFactory.
    //
    void PyBindRegion::createSpec(const char * nodeType, Spec& ns, const char* className)
    {
        std::string realClassName(className);
        if (realClassName.empty())
        {
            realClassName = Path::getExtension(nodeType);
        }

        try
        {
            auto m = py::module::import(nodeType);
            auto pyClass = m.attr(realClassName.c_str());

            //@todo Not sure these are required
            //auto pyList = pyClass.attr("getList")();
            //auto List = pyList.cast<std::vector<std::string>>();

            //auto pyDict = pyClass.attr("getDict")();
            //auto Description = pyDict["description"].cast<std::string>();
            //auto SingleNodeOnly = pyDict["singleNodeOnly"].cast<bool>();

            auto pyNodeSpec = pyClass.attr("getSpec")();
            ns.description = pyNodeSpec["description"].cast<std::string>();
            ns.singleNodeOnly = pyNodeSpec["singleNodeOnly"].cast<bool>();

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

                    NTA_ASSERT(input.contains("count") != nullptr)
                        << inputMessagePrefix.str() << "count";
                    auto count = input["count"].cast<UInt32>();

                    NTA_ASSERT(input.contains("required"))
                        << inputMessagePrefix.str() << "required";
                    auto required = input["required"].cast<bool>();

                    // make regionLevel optional and default to true.
                    bool regionLevel = true;
                    if (input.contains("regionLevel"))
                    {
                        regionLevel = input["regionLevel"].cast<bool>();
                    }

                    NTA_ASSERT(input.contains("isDefaultInput"))
                        << inputMessagePrefix.str() << "isDefaultInput";
                    auto isDefaultInput = input["isDefaultInput"].cast<bool>();

                    // make requireSplitterMap optional and default to false.
                    bool requireSplitterMap = false;
                    if (input.contains("requireSplitterMap"))
                    {
                        requireSplitterMap = input["requireSplitterMap"].cast<bool>();
                    }

                    ns.inputs.add(
                        name,
                        InputSpec(
                            description,
                            dataType,
                            count,
                            required,
                            regionLevel,
                            isDefaultInput,
                            requireSplitterMap));
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

                    NTA_ASSERT(output.contains("isDefaultOutput"))
                        << outputMessagePrefix.str() << "isDefaultOutput";
                    bool isDefaultOutput = output["isDefaultOutput"].cast<bool>();

                    ns.outputs.add(
                        name,
                        OutputSpec(
                            description,
                            dataType,
                            count,
                            regionLevel,
                            isDefaultOutput));
                }
            }

            if (pyNodeSpec.contains("parameters"))
            {
                auto parameters = pyNodeSpec["parameters"];

                // Add parameters
                for (auto it = parameters.begin(); it != parameters.end(); ++it)
                {
                    auto name = it->cast<std::string>();
                    auto parameter = parameters[*it];

                    // Add an ParameterSpec object for each output spec dict
                    std::ostringstream parameterMessagePrefix;
                    parameterMessagePrefix << "Region " << realClassName
                        << " spec has missing key for parameter section " << name << ": ";

                    NTA_ASSERT(parameter.contains("description"))
                        << parameterMessagePrefix.str() << "description";
                    auto description = parameter["description"].cast<std::string>();

                    NTA_ASSERT(parameter.contains("dataType"))
                        << parameterMessagePrefix.str() << "dataType";
                    auto dt = parameter["dataType"].cast<std::string>();
                    NTA_BasicType dataType;
                    try {
                        dataType = BasicType::parse(dt);
                    }
                    catch (Exception &) {
                        std::stringstream stream;
                        stream << "Invalid 'dataType' specificed for parameter '" << name
                            << "' when getting spec for region '" << realClassName << "'.";
                        throw Exception(__FILE__, __LINE__, stream.str());
                    }

                    NTA_ASSERT(parameter.contains("count"))
                        << parameterMessagePrefix.str() << "count";
                    auto count = parameter["count"].cast<UInt32>();

                    std::string constraints = "";
                    // This parameter is optional
                    if (parameter.contains("constraints")) {
                        constraints = parameter["constraints"].cast<std::string>();
                    }

                    NTA_ASSERT(parameter.contains("accessMode"))
                        << parameterMessagePrefix.str() << "accessMode";
                    ParameterSpec::AccessMode accessMode;
                    auto am = parameter["accessMode"].cast<std::string>();
                    if (am == "Create")
                        accessMode = ParameterSpec::CreateAccess;
                    else if (am == "Read")
                        accessMode = ParameterSpec::ReadOnlyAccess;
                    else if (am == "ReadWrite")
                        accessMode = ParameterSpec::ReadWriteAccess;
                    else
                        NTA_THROW << "Invalid access mode: " << am;

                    // Get default value as a string if it's a create parameter
                    std::string defaultValue;
                    if (am == "Create")
                    {
                        NTA_ASSERT(parameter.getItem("defaultValue") != nullptr)
                            << parameterMessagePrefix.str() << "defaultValue";
                        auto dv = parameter["defaultValue"];
                        defaultValue = dv.attr("__str__").cast<std::string>();
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

                // Add the automatic "self" parameter
                ns.parameters.add(
                    "self",
                    ParameterSpec(
                        "The PyObject * of the region's Python classd",
                        NTA_BasicType_Handle,
                        1,
                        "",
                        "",
                        ParameterSpec::ReadOnlyAccess));
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
        catch (const py::error_already_set& e)
        {
            std::cout << e.what() << std::endl;
        }
        catch (const py::cast_error& e)
        {
            std::cout << e.what() << std::endl;
        }
        catch (...)
        {
            throw std::runtime_error("Unknown error.");
        }
    }

    void PyBindRegion::initialize()
    {
        node_.attr("initialize")();
    }

} // namespace nupic

