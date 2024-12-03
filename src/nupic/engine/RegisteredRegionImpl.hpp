/*
 * Copyright 2015 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Definition of the RegisteredRegionImpl
 *
 * A RegisteredRegionImpl is an object that can instantiate a subclass of
 * RegionImpl and get its spec.
 */

#ifndef NTA_REGISTERED_REGION_IMPL_HPP
#define NTA_REGISTERED_REGION_IMPL_HPP

#include <string>

namespace nupic {
struct Spec;
class BundleIO;
class RegionImpl;
class Region;
class ValueMap;

class GenericRegisteredRegionImpl {
public:
  GenericRegisteredRegionImpl() {}

  virtual ~GenericRegisteredRegionImpl() {}

  virtual RegionImpl *createRegionImpl(const ValueMap &params,
                                       Region *region) = 0;

  virtual RegionImpl *deserializeRegionImpl(BundleIO &params,
                                            Region *region) = 0;

  virtual RegionImpl *deserializeRegionImpl(capnp::AnyPointer::Reader &proto,
                                            Region *region) = 0;

  virtual Spec *createSpec() = 0;
};

template <class T>
class RegisteredRegionImpl : public GenericRegisteredRegionImpl {
public:
  RegisteredRegionImpl() {}

  ~RegisteredRegionImpl() {}

  virtual RegionImpl *createRegionImpl(const ValueMap &params,
                                       Region *region) override {
    return new T(params, region);
  }

  virtual RegionImpl *deserializeRegionImpl(BundleIO &params,
                                            Region *region) override {
    return new T(params, region);
  }

  virtual RegionImpl *deserializeRegionImpl(capnp::AnyPointer::Reader &proto,
                                            Region *region) override {
    return new T(proto, region);
  }

  virtual Spec *createSpec() override { return T::createSpec(); }
};

} // namespace nupic

#endif // NTA_REGISTERED_REGION_IMPL_HPP
