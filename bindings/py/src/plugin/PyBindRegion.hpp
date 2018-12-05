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
Definition of the PyBindRegion class.  The base class for all Python Region implementations.
*/

#ifndef NTA_PYBIND_REGION_HPP
#define NTA_PYBIND_REGION_HPP

#include <string>
#include <vector>
#include <set>


// the use of 'register' keyword is removed in C++17
// Python2.7 uses 'register' in unicodeobject.h
#ifdef _WIN32
#pragma warning( disable : 5033)  // MSVC
#else
#pragma GCC diagnostic ignored "-Wregister"  // for GCC and CLang
#endif

#include <pybind11/pybind11.h>

#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Value.hpp>


namespace nupic
{
    class PyBindRegion : public RegionImpl
    {
        typedef std::map<std::string, Spec> SpecMap;

    public:
		/**
		* Registers a python region implementation class so that it can be instantiated
		* when its name is used in a Network::addRegion() call.
		*
		* @param nodeType -- a name for the region implementation type. This is normally 
		*                                   the class name prefixed with 'py_' to  avoid name conflicts.
                  * @param className -- the name of the Python class that implements the region.
                  * @param module      -- the module (shared library) in which the class resides.
                  */
		static void registerPyRegion(const std::string& nodeType,
									 const std::string& module,
                                     const std::string& className) {
			RegisteredRegionImplPy *reg = new RegisteredRegionImplPy(className, module);
			RegionImplFactory::registerRegion(nodeType, reg);
		} 
		
		/*
		  * Removes a region from RegionImplFactory's packages
		  */
		static void unregisterPyRegion(const std::string nodeType);


        // Used by RegionImplFactory via RegisterRegionImplPy  to create and cache a nodespec
        static void createSpec(const char * nodeType, Spec& ns, const char* className = "");


        // Used by RegisterRegionImplPy to destroy a node spec when clearing its cache
        static void destroySpec(const char * nodeType, const char* className = "");

        // Constructors
        PyBindRegion() = delete;
        PyBindRegion(const char * module, const ValueMap & nodeParams, Region * region, const char* className);
        PyBindRegion(const char * module, BundleIO& bundle, Region * region, const char* className);

        // no copy constructor
        PyBindRegion(const Region &) = delete;

        // Destructor
        virtual ~PyBindRegion();

        ////////////////////////////
        // DynamicPythonLibrary
        ////////////////////////////

        // DynamicPythonLibrary functions. Originally used NTA_EXPORT
        //static void NTA_initPython();
        //static void NTA_finalizePython();
        //static void * NTA_createPyNode(const char * module, void * nodeParams, void * region, void ** exception, const char* className = "");
        //static void * NTA_deserializePyNode(const char * module, void * bundle, void * region, void ** exception, const char* className = "");
        //static void * NTA_deserializePyNodeProto(const char * module, void * proto, void * region, void ** exception, const char* className = "");
        //static const char * NTA_getLastError();
        //static void * NTA_createSpec(const char * nodeType, void ** exception, const char* className = "");
        //static int NTA_destroySpec(const char * nodeType, const char* className = "");


        // Manual serialization methods. Current recommended method.
        void serialize(BundleIO& bundle) override;
        void deserialize(BundleIO& bundle) override;



        ////////////////////////////
        // RegionImpl 
        ////////////////////////////

        size_t getNodeOutputElementCount(const std::string& outputName) override;
        void getParameterFromBuffer(const std::string& name, Int64 index, IWriteBuffer& value) override;
        void setParameterFromBuffer(const std::string& name, Int64 index, IReadBuffer& value) override;

        void initialize() override;
        void compute() override;
        std::string executeCommand(const std::vector<std::string>& args, Int64 index) override;

        size_t getParameterArrayCount(const std::string& name, Int64 index) override;

        virtual Byte getParameterByte(const std::string& name, Int64 index);
        virtual Int32 getParameterInt32(const std::string& name, Int64 index) override;
        virtual UInt32 getParameterUInt32(const std::string& name, Int64 index) override;
        virtual Int64 getParameterInt64(const std::string& name, Int64 index) override;
        virtual UInt64 getParameterUInt64(const std::string& name, Int64 index) override;
        virtual Real32 getParameterReal32(const std::string& name, Int64 index) override;
        virtual Real64 getParameterReal64(const std::string& name, Int64 index) override;
        virtual pybind11::object getParameterHandle(const std::string& name, Int64 index) override;
        virtual bool getParameterBool(const std::string& name, Int64 index) override;
        virtual std::string getParameterString(const std::string& name, Int64 index) override;

        virtual void setParameterByte(const std::string& name, Int64 index, Byte value);
        virtual void setParameterInt32(const std::string& name, Int64 index, Int32 value) override;
        virtual void setParameterUInt32(const std::string& name, Int64 index, UInt32 value) override;
        virtual void setParameterInt64(const std::string& name, Int64 index, Int64 value) override;
        virtual void setParameterUInt64(const std::string& name, Int64 index, UInt64 value) override;
        virtual void setParameterReal32(const std::string& name, Int64 index, Real32 value) override;
        virtual void setParameterReal64(const std::string& name, Int64 index, Real64 value) override;
        virtual void setParameterHandle(const std::string& name, Int64 index, Handle value) override;
        virtual void setParameterBool(const std::string& name, Int64 index, bool value) override;
        virtual void setParameterString(const std::string& name, Int64 index, const std::string& value) override;

        virtual void getParameterArray(const std::string& name, Int64 index, Array & array) override;
        virtual void setParameterArray(const std::string& name, Int64 index, const Array & array) override;

        // Helper methods
        template <typename T>
        T getParameterT(const std::string & name, Int64 index);

        template <typename T>
        void setParameterT(const std::string & name, Int64 index, T value);


    private:
        std::string module_;
        std::string className_;

        pybind11::object node_;
  		std::set<std::shared_ptr<PyArray<UInt64>>> splitterMaps_;
        
        // pointers rather than objects because Array doesnt
        // have a default constructor
        std::map<std::string, Array*> inputArrays_;

        static std::string last_error;
        
        Spec nodeSpec_;   // locally cached version of spec.
    };
    
    

} // namespace nupic

#endif //NTA_PYBIND_REGION_HPP