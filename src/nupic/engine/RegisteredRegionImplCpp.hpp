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
 * Definition of the RegisteredRegionImplCpp
 * Provides the plugin interface for the Cpp implemented Regions.
 * This is a subclass of RegisteredRegionImpl; the base class of an object that can instantiate
 * a plugin (a subclass of RegionImpl) and get its spec.
 *
 * the subclasses of RegistedRegionImpl must perform the following:
 *    1) Be Registered with the CPP engine using
 *              Network::registerCPPRegion(name, your_subclass);
 *       It only needs to be registed once even if multiple Regions will use
 *       an instance of the same plugin. The 'name' used in this registration
 *       is the 'nodeType' when calling Network::addRegion() to create a
 *       region. It is like declaring the type of the plugin.
 *       As a convention, the name used by C++ plugins will be the class name.
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
 *       The pointer returned from the plugin should be cached.  The
 *       RegistedRegionImpl base class contains "std::shared_ptr<Spec> cachedSpec_;"
 *       that the subclass may use for this purpose.
 *

 */

#ifndef NTA_REGISTERED_REGION_IMPL_CPP_HPP
#define NTA_REGISTERED_REGION_IMPL_CPP_HPP

#include <string>
#include <nupic/engine/RegisteredRegionImpl.hpp>

namespace nupic
{
  class Spec;
  class BundleIO;
  class RegionImpl;
  class Region;
  class ValueMap;


  template <class T>
  class RegisteredRegionImplCpp: public RegisteredRegionImpl {
    public:
      RegisteredRegionImplCpp(const std::string& classname="", const std::string& module="")
	  		: RegisteredRegionImpl(classname, module) {
	  }

      ~RegisteredRegionImplCpp() override {
      }

      RegionImpl* createRegionImpl( ValueMap& params, Region *region) override {
        return new T(params, region);
      }

      RegionImpl* deserializeRegionImpl( BundleIO& bundle, Region *region) override {
        return new T(bundle, region);
      }

      Spec* createSpec() override
      {
        if (!cachedSpec_) {
          Spec *sp = T::createSpec();
          cachedSpec_.reset(sp);
        }
        return cachedSpec_.get();
      }
	private:
		std::shared_ptr<Spec> cachedSpec_;

  };

}

#endif // NTA_REGISTERED_REGION_IMPL_CPP_HPP
