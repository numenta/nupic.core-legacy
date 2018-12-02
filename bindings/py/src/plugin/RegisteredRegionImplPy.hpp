/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

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
 *       an instance of the same plugin. The 'nodeType' used in this registration
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
 *       The pointer to the spec returned from the plugin should be cached in this class.  The
 *       RegistedRegionImpl base class contains "std::shared_ptr<Spec> cachedSpec_;"
 *       that this subclass may use for this purpose.
 *
 *    6)  Before doing anything with a python region, we must initialize the python interpreter.
 *
 *    7) After the last python region has been deleted, we must finalize the python interpreter.
 */

#ifndef NTA_REGISTERED_REGION_IMPL_CPP_HPP
#define NTA_REGISTERED_REGION_IMPL_CPP_HPP

#include <string>
#include <nupic/engine/RegisteredRegionImpl.hpp>

namespace nupic
{
  class Spec;
  class BundleIO;
  class PyRegionImpl;
  class Region;
  class ValueMap;


  class RegisteredRegionImplPy: public RegisteredRegionImpl {
  public:
	  RegisteredRegionImplPy(const std::string& classname, const std::string& module="")
			: RegisteredRegionImpl(classname, module) {
		if (python_region_count == 0) {
			try {
				py::initialize_interpreter();
			}
			catch (const std::exception& e)
			{
				throw Exception(__FILE__, __LINE__, e.what());
			}
		}
		python_region_count++;
	  }

      ~RegisteredRegionImplPy() override {
		python_region_count--;
		if (python_region_count == 0) {
            py::finalize_interpreter();
		}
      }

      RegionImpl* createRegionImpl( ValueMap& params, Region *region) override
      {
        // use PyBindRegion class to instantiate the python class in the specified module.
        return new PyBindRegion(module_.c_str(), params, region, className_.c_str());
      }

        // use PyBindRegion class to instantiate and deserialize the python class in the specified module.
      RegionImpl* deserializeRegionImpl(BundleIO& bundle, Region *region) override
      {
        return new PyBindRegion(nodeType_.c_str(), bundle, region, className_.c_str());
      }

      Spec* createSpec() override
      {
        if (!cachedSpec_) {
          Spec* sp = new Spec();
          try {
			PyBindRegion::createSpec(module_.c_str(), *sp, className_.c_str());
		  }
          catch (nupic::Exception & e) {
            delete sp;
            throw;
          }
          catch (...) {
            delete sp;
			NTA_THROW << "PyBindRegion::createSpec failed: unknown exception.";
          }
          cachedSpec_.reset(sp);
        }
        return cachedSpec_.get();
      }

    private:
		static int python_region_count = 0;
		std::shared_ptr<Spec> cachedSpec_;

  };

}

#endif // NTA_REGISTERED_REGION_IMPL_CPP_HPP
