/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2015, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Definition of the RegisteredRegionImplPy
 * This provides the plugin interface for the Python implemented Regions.
 * This is a subclass of RegisteredRegionImpl; the base class of an object that can instantiate
 * a plugin (a subclass of RegionImpl) and get its spec.
 *
 * the subclasses of RegistedRegionImpl must perform the following:
 *    1) Be Registered with the CPP engine using
 *              Network::registerRegion( nodeType, module, classname);
 *       It only needs to be registed once even if multiple Regions will use
 *       an instance of the same plugin type. The 'nodeType' used in this registration
 *       is the 'nodeType' when calling Network::addRegion() to create a
 *       region. It is like declaring the type of the plugin.
 *       As a convention, the nodeType used by C++ plugins will be the class name.
 *       The name for Python plugins should start with 'py_'. Those for CSharp
 *       will start with 'cs_'.
 *
 *    2) Override the destructor if needed to cleanup your RegisteredRegionImpl subclass
 *
 *    3) Instantiate the plugin and return its pointer when createRegionImpl()
 *       is called.
 *
 *    4) Instantiate and deserialize the plugin when deserializeRegionImpl() is called,
 *       returning its pointer. This gets called when Network::Network(path) is called
 *       to initialize an entire network from a previous serialization file.
 *
 *    5) Get and return a pointer to the spec from the plugin when createSpec() is called.
 *       During this call the class should be loaded.
 *       The pointer to the spec returned from the plugin should be dynamically allocated
 *       and the caller takes ownership.
 *
 *    6)  Before doing anything with a python region, we must initialize the python interpreter.
 *
 *    7) After the last python region has been deleted, we must finalize the python interpreter.
 *
 * An instance of a RegisteredRegionImplPy class represents a Python Region implementation type registration.
 * An instance of a PyBindRegion class represents an instance of a Python Region implentation.
 *
 */

#ifndef NTA_REGISTERED_REGION_IMPL_CPP_HPP
#define NTA_REGISTERED_REGION_IMPL_CPP_HPP

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <plugin/PyBindRegion.hpp>

#include <htm/engine/RegisteredRegionImpl.hpp>
#include <htm/engine/RegionImplFactory.hpp>
#include <htm/ntypes/Value.hpp>
#include <string>

namespace py = pybind11;

// A global variable to hold the number of python classes currently registered.
// If this is 0 then python library has not been initialized.
static int python_node_count = 0;


namespace htm
{
  class Spec;
  class PyRegionImpl;
  class Region;


  class RegisteredRegionImplPy: public RegisteredRegionImpl {
  public:
	  RegisteredRegionImplPy(const std::string& classname, const std::string& module="")
			: RegisteredRegionImpl(classname, module) {
	    // NOTE: If this is called by .py code (and probably is) then the python interpreter is already running
		//       so we don't need to start it up.
	  }

      ~RegisteredRegionImplPy() override {
      }

      RegionImpl* createRegionImpl( ValueMap& params, Region *region) override
      {
	  	try {
          // use PyBindRegion class to instantiate the python class in the specified module.
          return new PyBindRegion(module_.c_str(), params, region, classname_.c_str());
        }
        catch (const py::error_already_set& e)
        {
            throw Exception(__FILE__, __LINE__, e.what());
        }
        catch (const py::cast_error& e)
        {
            throw Exception(__FILE__, __LINE__, e.what());
        }
        catch (htm::Exception & e)
        {
            throw htm::Exception(e);
        }
        catch (std::exception &e)
        {
            NTA_THROW << std::string(e.what());
        }
        catch (...)
        {
            NTA_THROW << "Something bad happed while creating a .py region";
        }
	  }

        // use PyBindRegion class to instantiate and deserialize the python class in the specified module.
      RegionImpl* deserializeRegionImpl(ArWrapper& wrapper, Region *region) override
      {
	  	try {
          return new PyBindRegion(module_.c_str(), wrapper, region, classname_.c_str());
        }
        catch (const py::error_already_set& e)
        {
            throw Exception(__FILE__, __LINE__, e.what());
        }
        catch (const py::cast_error& e)
        {
            throw Exception(__FILE__, __LINE__, e.what());
        }
        catch (htm::Exception & e)
        {
            throw htm::Exception(e);
        }
        catch (...)
        {
            NTA_THROW << "Something bad happed while deserializing a .py region";
        }
      }

      Spec* createSpec() override
      {
          Spec* sp = new Spec();
          try {
			PyBindRegion::createSpec(module_.c_str(), *sp, classname_.c_str());
		  }
          catch (htm::Exception & e) {
		    UNUSED(e);
            delete sp;
            throw;
          }
          catch (...) {
            delete sp;
			NTA_THROW << "PyBindRegion::createSpec failed: unknown exception.";
          }
          return sp;
      }

		/**
		* Registers a python region implementation class so that it can be instantiated
		* when its name is used in a Network::addRegion() call.
		*
		* @param className -- the name of the Python class that implements the region.
		* @param module    -- the module (full path and file) in which the class resides.
		*/
		inline static void registerPyRegion(const std::string& module, const std::string& className) {
		    std::string nodeType = "py." + className;
			RegisteredRegionImplPy *reg = new RegisteredRegionImplPy(className, module);
			RegionImplFactory::registerRegion(nodeType, reg);
		}

		/*
		  * Removes a region from RegionImplFactory's packages
		  */
		inline static void unregisterPyRegion(const std::string& className) {
		    std::string nodeType = "py." + className;
			RegionImplFactory::unregisterRegion(nodeType);
		}




  };



}

#endif // NTA_REGISTERED_REGION_IMPL_CPP_HPP
