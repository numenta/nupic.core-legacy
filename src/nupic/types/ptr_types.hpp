/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
 * Basic C++ type definitions used throughout `nupic.core` and rely on `Types.h`
 */

#ifndef NTA_PTR_TYPES_HPP
#define NTA_PTR_TYPES_HPP

#include <memory>

namespace nupic {
  class Link;
  class Region;
  class Spec;

  typedef std::shared_ptr<Link> Link_Ptr_t;
  typedef std::shared_ptr<Region> Region_Ptr_t;
  typedef std::shared_ptr<Spec> Spec_Ptr_t;

} // namespace nupic

#endif // NTA_PTR_TYPES_HPP
