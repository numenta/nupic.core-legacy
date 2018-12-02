/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <stdexcept>


#include <nupic/regions/ScalarSensor.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/engine/RegisteredRegionImpl.hpp>
#include <nupic/engine/RegisteredRegionImplCpp.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/regions/TestNode.hpp>
#include <nupic/engine/YAMLUtils.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/regions/VectorFileEffector.hpp>
#include <nupic/regions/VectorFileSensor.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/StringUtils.hpp>

// from http://stackoverflow.com/a/9096509/1781435
#define stringify(x) #x
#define expand_and_stringify(x) stringify(x)

namespace nupic {

// Mappings for region nodeTypes that map to Class and module
static std::map<const std::string, std::shared_ptr<RegisteredRegionImpl> > regionTypeMap;

bool initializedRegions = false;



void RegionImplFactory::registerRegion(const std::string& nodeType, RegisteredRegionImpl *wrapper) {
  if (regionTypeMap.find(nodeType) != regionTypeMap.end()) {
	std::shared_ptr<RegisteredRegionImpl>& reg = regionTypeMap[nodeType];
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
    std::shared_ptr<RegisteredRegionImpl> reg(wrapper);

  	regionTypeMap[nodeType] = reg;
  }
}


void RegionImplFactory::unregisterRegion(const std::string nodeType) {
  if (regionTypeMap.find(nodeType) != regionTypeMap.end()) {
    regionTypeMap.erase(nodeType);
  }
}


RegionImplFactory &RegionImplFactory::getInstance() {
  static RegionImplFactory instance;

  // Initialize the Built-in Regions
  if (!initializedRegions) {
    // Create internal C++ regions
	instance.registerRegion("ScalarSensor",       new RegisteredRegionImplCpp<ScalarSensor>());
    instance.registerRegion("TestNode",           new RegisteredRegionImplCpp<TestNode>());
    instance.registerRegion("VectorFileEffector", new RegisteredRegionImplCpp<VectorFileEffector>());
    instance.registerRegion("VectorFileSensor",   new RegisteredRegionImplCpp<VectorFileSensor>());

    initializedRegions = true;
  }

  return instance;
}



RegionImpl *RegionImplFactory::createRegionImpl(const std::string nodeType,
                                                const std::string nodeParams,
                                                Region *region) {

  RegionImpl *impl = nullptr;
  Spec *ns = getSpec(nodeType);
  ValueMap vm = YAMLUtils::toValueMap(nodeParams.c_str(), ns->parameters,
                                      nodeType, region->getName());

  if (regionTypeMap.find(nodeType) != regionTypeMap.end()) {
    impl = regionTypeMap[nodeType]->createRegionImpl(vm, region);
  } else {
    NTA_THROW << "Unregistered node type '" << nodeType << "'";
  }

  return impl;
}

RegionImpl *RegionImplFactory::deserializeRegionImpl(const std::string nodeType,
                                                     BundleIO &bundle,
                                                     Region *region) {

  RegionImpl *impl = nullptr;

  if (regionTypeMap.find(nodeType) != regionTypeMap.end()) {
    impl = regionTypeMap[nodeType]->deserializeRegionImpl(bundle, region);
  } else {
    NTA_THROW << "Unsupported node type '" << nodeType << "'";
  }
  return impl;
}



Spec *RegionImplFactory::getSpec(const std::string nodeType) {
  std::map<std::string, std::shared_ptr<RegisteredRegionImpl>>::iterator it;
  it = regionTypeMap.find(nodeType);
  if (it == regionTypeMap.end()) {
	NTA_THROW << "getSpec() -- unknown node type: '" << nodeType
		      << "'.  Custom node types must be registed before they can be used.";
  }
  Spec *ns = it->second->createSpec();
  return ns;
}

void RegionImplFactory::cleanup() {

  regionTypeMap.clear();
  initializedRegions = false;
}

} // namespace nupic
