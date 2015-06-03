/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
 * Definition of the RegionImplWapper
 *
 * A RegionImplWrapper is an object that can instantiate a subclass of
 * RegionImpl and get its spec.
 */

#ifndef NTA_REGION_IMPL_WRAPPER_HPP
#define NTA_REGION_IMPL_WRAPPER_HPP

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
      GenericRegisteredRegionImpl() {
      }
      virtual RegionImpl* createRegionImpl(const ValueMap& params, Region *region) {
      }
      virtual RegionImpl* deserializeRegionImpl(BundleIO& params, Region *region) {
      }
      virtual Spec* createSpec() {
      }
  };

  template <class T>
  class RegisteredRegionImpl: public GenericRegisteredRegionImpl {
    public:
      RegisteredRegionImpl() {
      }
      T* createRegionImpl(const ValueMap& params, Region *region) {
        return new T(params, region);
      }
      T* deserializeRegionImpl(BundleIO& params, Region *region) {
        return new T(params, region);
      }
      Spec* createSpec()
      {
        return T::createSpec();
      }
  };
}


#endif // NTA_REGION_IMPL_WRAPPER_HPP
