/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2018, Numenta, Inc.
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
 * Definition of the RegionImpl Factory API
 *
 * A RegionImplFactory creates RegionImpls upon request.
 * Pynode creation is delegated to pyBindRegion via RegisteredRegionImplPy
 * Because all C++ RegionImpls are compiled in to NuPIC,
 * the RegionImpl factory knows about them explicitly.
 *
 */

#ifndef NTA_REGION_IMPL_FACTORY_HPP
#define NTA_REGION_IMPL_FACTORY_HPP

#include <map>
#include <memory>
#include <string>
#include <htm/types/Serializable.hpp>




namespace htm {

class RegionImpl;
class Region;
class Spec;
class RegisteredRegionImpl;

class RegionImplFactory {
public:
  static RegionImplFactory &getInstance();

  // RegionImplFactory is a lightweight object
  ~RegionImplFactory(){};

  // Create a RegionImpl of a specific type; caller gets ownership.
  RegionImpl *createRegionImpl(const std::string nodeType,
                               const std::string nodeParams, Region *region);

  // Create a RegionImpl from serialized state; caller gets ownership.
  RegionImpl *deserializeRegionImpl(const std::string nodeType,
                                    ArWrapper &wrapper, Region *region);


  // Returns node spec for a specific node type as a shared pointer.
  std::shared_ptr<Spec>& getSpec(const std::string nodeType);

  // RegionImplFactory caches nodespecs and the dynamic library reference
  // This frees up the cached information.
  // Should be called only if there are no outstanding
  // nodespec references (e.g. in NuPIC shutdown) or pynodes.
  // Used in unit tests to Setup for next test.
  static void cleanup();


  // Allows the user to load custom region types
  static void registerRegion(const std::string& regionType,
                             RegisteredRegionImpl *wrapper);

  // Allows the user to unregister region types
  static void unregisterRegion(const std::string regionType);

private:
  RegionImplFactory(){};
  RegionImplFactory(const RegionImplFactory &);


  // Mappings for region nodeTypes that map to Class and module
  std::map<const std::string, std::shared_ptr<RegisteredRegionImpl> > regionTypeMap;
  std::map<const std::string, std::shared_ptr<Spec> > regionSpecMap;
  void addRegionType(const std::string nodeType, RegisteredRegionImpl* wrapper);

};
} // namespace htm

#endif // NTA_REGION_IMPL_FACTORY_HPP
