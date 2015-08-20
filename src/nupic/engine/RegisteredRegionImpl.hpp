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
 * Definition of the RegisteredRegionImpl
 *
 * A RegisteredRegionImpl is an object that can instantiate a subclass of
 * RegionImpl and get its spec.
 */

#ifndef NTA_REGISTERED_REGION_IMPL_HPP
#define NTA_REGISTERED_REGION_IMPL_HPP

#include <string>

namespace nupic
{
  struct Spec;
  class BundleIO;
  class RegionImpl;
  class Region;
  class ValueMap;

  class GenericRegisteredRegionImpl {
    public:
      GenericRegisteredRegionImpl() {}

      virtual ~GenericRegisteredRegionImpl() {}

      virtual RegionImpl* createRegionImpl(
          const ValueMap& params, Region *region) = 0;

      virtual RegionImpl* deserializeRegionImpl(
          BundleIO& params, Region *region) = 0;

      virtual RegionImpl* deserializeRegionImpl(
          capnp::AnyPointer::Reader& proto, Region *region) = 0;

      virtual Spec* createSpec() = 0;
  };

  template <class T>
  class RegisteredRegionImpl: public GenericRegisteredRegionImpl {
    public:
      RegisteredRegionImpl() {}

      ~RegisteredRegionImpl() {}

      virtual RegionImpl* createRegionImpl(
          const ValueMap& params, Region *region) override
      {
        return new T(params, region);
      }

      virtual RegionImpl* deserializeRegionImpl(
          BundleIO& params, Region *region) override
      {
        return new T(params, region);
      }

      virtual RegionImpl* deserializeRegionImpl(
          capnp::AnyPointer::Reader& proto, Region *region) override
      {
        return new T(proto, region);
      }

      virtual Spec* createSpec() override
      {
        return T::createSpec();
      }
  };

}

#endif // NTA_REGISTERED_REGION_IMPL_HPP
