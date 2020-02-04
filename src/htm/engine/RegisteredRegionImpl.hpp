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
 * Definition of the RegisteredRegionImpl
 *
 * A RegisteredRegionImpl is a base class of an object that can instantiate
 * a plugin (a subclass of RegionImpl) and get its spec.
 *
 * Each Programming language interface to this library must create a subclass
 * of RegisteredRegionImpl which will handle the engine to plugin instantiation.
 * For example, here are the subclasses which implement plugin interfaces:
 *   - RegisteredRegionImplCpp for C++ builtin plugins (subclasses of RegionImpl).
 *   - RegisteredRegionImplPy  for Python implemented plugins (PyBindRegion)
 *   - RegisteredRegionImplCs  for CSharp implemented pubgins (CsBindRegion)
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
 *       to initialize an entire network from a previous serialization bundle.
 *
 *    5) Get and return a pointer to the spec from the plugin when createSpec() is called.
 *       The pointer returned from the plugin should be cached.  The
 *       RegistedRegionImpl base class contains "std::shared_ptr<Spec> cachedSpec_;"
 *       that the subclass may use for this purpose.
 */

#ifndef NTA_REGISTERED_REGION_IMPL_HPP
#define NTA_REGISTERED_REGION_IMPL_HPP

#include <string>
#include <htm/ntypes/Value.hpp>

namespace htm
{
  class Spec;
  class ArWrapper;
  class RegionImpl;
  class Region;

  class RegisteredRegionImpl {
  public:
    // 'classname' is name of the RegionImpl subclass that is to be instantiated for this plugin.
    // 'module' is the name (or path) to the shared library in which 'classname' resides.  Empty if staticly linked.
    RegisteredRegionImpl(const std::string& classname, const std::string& module=""){
		  classname_ = classname;
		  module_ = module;
    }

    virtual ~RegisteredRegionImpl() {}

    virtual RegionImpl* createRegionImpl( ValueMap& params, Region *region) = 0;

    virtual RegionImpl* deserializeRegionImpl( ArWrapper& wrapper, Region *region) = 0;

    virtual Spec* createSpec() = 0;

	  virtual std::string className() { return classname_; }
	  virtual std::string moduleName() { return module_; }

  protected:
	  std::string classname_;
	  std::string module_;

  };


}

#endif // NTA_REGISTERED_REGION_IMPL_HPP
