/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
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
 * Define the CategoryEncoder
 */

#ifndef NTA_ENCODERS_CATEGORY
#define NTA_ENCODERS_CATEGORY

#include <map>
#include <nupic/types/Types.hpp>
#include <nupic/types/Sdr.hpp>

namespace nupic {

// template<typename CategoryType>
  typedef UInt CategoryType;
class CategoryEncoder {
private:
  UInt size_;
  Real sparsity_;
  std::map<CategoryType, SDR> map_;

public:
  // TODO DOCUMENTATION
  CategoryEncoder(UInt size, Real sparsity);

  const Real &size                       = size_;
  const Real &sparsity                   = sparsity_;
  const std::map<CategoryType, SDR> &map = map_;

  /**
  TODO DOCUMENTATION
   */
  void encode(const CategoryType value, SDR &output);

};

#endif // NTA_ENCODERS_CATEGORY
