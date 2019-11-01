/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2015, Numenta, Inc.
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

#include <stdexcept>


#include <htm/engine/Region.hpp>
#include <htm/engine/RegionImpl.hpp>
#include <htm/engine/RegionImplFactory.hpp>
#include <htm/engine/RegisteredRegionImpl.hpp>
#include <htm/engine/RegisteredRegionImplCpp.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/os/Env.hpp>
#include <htm/os/Path.hpp>

// Built-in Region implementations
#include <htm/regions/TestNode.hpp>
#include <htm/regions/ScalarSensor.hpp>
#include <htm/regions/VectorFileEffector.hpp>
#include <htm/regions/VectorFileSensor.hpp>
#include <htm/regions/SPRegion.hpp>
#include <htm/regions/TMRegion.hpp>


#include <htm/utils/Log.hpp>

// from http://stackoverflow.com/a/9096509/1781435
#define stringify(x) #x
#define expand_and_stringify(x) stringify(x)

namespace htm {



void RegionImplFactory::registerRegion(const std::string& nodeType, RegisteredRegionImpl *wrapper) {
  RegionImplFactory& instance = getInstance();
  if (instance.regionTypeMap.find(nodeType) != instance.regionTypeMap.end()) {
	std::shared_ptr<RegisteredRegionImpl>& reg = instance.regionTypeMap[nodeType];
	if (reg->className() == wrapper->className() && reg->moduleName() == wrapper->moduleName()) {
		NTA_WARN << "A Region Type already exists with the name '" << nodeType
				 << "'. Overwriting it...";
		reg.reset(wrapper);  // replace this impl
	} else {
        NTA_THROW << "A region Type with name '" << nodeType
                  << "' already exists. Class name='" << reg->className()
                  << "'  Module='" << reg->moduleName() << "'. "
                  << "Unregister the existing region Type or register the new "
                     "region Type using a "
                  << "different name.";
	}
  } else {
    instance.addRegionType(nodeType, wrapper);
  }
}


void RegionImplFactory::unregisterRegion(const std::string nodeType) {
  RegionImplFactory& instance = getInstance();
  if (instance.regionTypeMap.find(nodeType) != instance.regionTypeMap.end()) {
    instance.regionTypeMap.erase(nodeType);
  }
}


RegionImplFactory &RegionImplFactory::getInstance() {
  static RegionImplFactory instance;

  // Initialize the Built-in Regions
  if (instance.regionTypeMap.empty()) {
    // Create internal C++ regions

	  instance.addRegionType("ScalarSensor",       new RegisteredRegionImplCpp<ScalarSensor>());
    instance.addRegionType("TestNode",           new RegisteredRegionImplCpp<TestNode>());
    instance.addRegionType("VectorFileEffector", new RegisteredRegionImplCpp<VectorFileEffector>());
    instance.addRegionType("VectorFileSensor",   new RegisteredRegionImplCpp<VectorFileSensor>());
    instance.addRegionType("SPRegion",           new RegisteredRegionImplCpp<SPRegion>());
    instance.addRegionType("TMRegion",            new RegisteredRegionImplCpp<TMRegion>());
  }

  return instance;
}

void RegionImplFactory::addRegionType(const std::string nodeType, RegisteredRegionImpl* wrapper) {
	std::shared_ptr<RegisteredRegionImpl> reg(wrapper);
  	regionTypeMap[nodeType] = reg;

	std::shared_ptr<Spec> ns(reg->createSpec());
	regionSpecMap[nodeType] = ns;
}


RegionImpl *RegionImplFactory::createRegionImpl(const std::string nodeType,
                                                const std::string nodeParams,
                                                Region *region) {

  RegionImpl *impl = nullptr;
  ValueMap vm;
  vm.parse(nodeParams);

  if (regionTypeMap.find(nodeType) != regionTypeMap.end()) {
    impl = regionTypeMap[nodeType]->createRegionImpl(vm, region);
  } else {
    NTA_THROW << "Unregistered node type '" << nodeType << "'";
  }

  // If the parameter 'dim' was defined, parse that out as a global parameter.
  if (vm.contains("dim")) {
    std::vector<UInt32> dim = vm["dim"].asVector<UInt32>();
    impl->setDimensions(dim);
  }

  return impl;
}

RegionImpl *RegionImplFactory::deserializeRegionImpl(const std::string nodeType,
                                                     ArWrapper &wrapper,
                                                     Region *region) {
std::cerr << "RegionImplFactory.deserializeRegionImpl called. " << nodeType << "\n";
  RegionImpl *impl = nullptr;
  if (regionTypeMap.find(nodeType) != regionTypeMap.end()) {
    impl = regionTypeMap[nodeType]->deserializeRegionImpl(wrapper, region);
  } else {
    NTA_THROW << "Unsupported node type '" << nodeType << "'";
  }
  return impl;
}



std::shared_ptr<Spec>& RegionImplFactory::getSpec(const std::string nodeType) {
  auto it = regionSpecMap.find(nodeType);
  if (it == regionSpecMap.end()) {
	NTA_THROW << "getSpec() -- unknown node type: '" << nodeType
		      << "'.  Custom node types must be registed before they can be used.";
  }
  return it->second;
}

void RegionImplFactory::cleanup() {
  RegionImplFactory& instance = getInstance();
  instance.regionTypeMap.clear();
  instance.regionSpecMap.clear();
}

} // namespace htm
