/*
 * Copyright 2013-2015 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Definition of the RegionImpl Factory API
 *
 * A RegionImplFactory creates RegionImpls upon request.
 * Pynode creation is delegated to another class (TBD).
 * Because all C++ RegionImpls are compiled in to NuPIC,
 * the RegionImpl factory knows about them explicitly.
 *
 */

#ifndef NTA_REGION_IMPL_FACTORY_HPP
#define NTA_REGION_IMPL_FACTORY_HPP

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

// Workaround windows.h collision:
// https://github.com/sandstorm-io/capnproto/issues/213
#undef VOID
#include <capnp/any.h>

namespace nupic {

class RegionImpl;
class Region;
class DynamicPythonLibrary;
struct Spec;
class BundleIO;
class ValueMap;
class GenericRegisteredRegionImpl;

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
                                    BundleIO &bundle, Region *region);

  // Create a RegionImpl from capnp proto; caller gets ownership.
  RegionImpl *deserializeRegionImpl(const std::string nodeType,
                                    capnp::AnyPointer::Reader &proto,
                                    Region *region);

  // Returns nodespec for a specific node type; Factory retains ownership.
  Spec *getSpec(const std::string nodeType);

  // RegionImplFactory caches nodespecs and the dynamic library reference
  // This frees up the cached information.
  // Should be called only if there are no outstanding
  // nodespec references (e.g. in NuPIC shutdown) or pynodes.
  void cleanup();

  static void registerPyRegionPackage(const char *path);

  // Allows the user to load custom Python regions
  static void registerPyRegion(const std::string module,
                               const std::string className);

  // Allows the user to load custom C++ regions
  static void registerCPPRegion(const std::string name,
                                GenericRegisteredRegionImpl *wrapper);

  // Allows the user to unregister Python regions
  static void unregisterPyRegion(const std::string className);

  // Allows the user to unregister C++ regions
  static void unregisterCPPRegion(const std::string name);

private:
  RegionImplFactory(){};
  RegionImplFactory(const RegionImplFactory &);

  // TODO: implement locking for thread safety for this global data structure
  // TODO: implement cleanup

  // getSpec returns references to nodespecs in this cache.
  // should not be cleaned up until those references have disappeared.
  std::map<std::string, Spec *> nodespecCache_;

  // Using shared_ptr here to ensure the dynamic python library object
  // is deleted when the factory goes away. Can't use scoped_ptr
  // because it is not initialized in the constructor.
  boost::shared_ptr<DynamicPythonLibrary> pyLib_;
};
} // namespace nupic

#endif // NTA_REGION_IMPL_FACTORY_HPP
