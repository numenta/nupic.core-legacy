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

#include <nupic/utils/Log.hpp> //NTA_CHECK
#include <nupic/encoders/CategoryEncoder.hpp>

namespace nupic {

CategoryEncoder::CategoryEncoder(UInt size, Real sparsity) {
  size_     = size;
  sparsity_ = sparsity;
}

void CategoryEncoder::encode(const CategoryType value, SDR &output) {
  NTA_CHECK( output.size == size );
  if( map_.count(value) == 0 ) {
    map_[value].randomize( sparsity )
  }
  output.setSDR( map_[value] );
}
} // end namespace nupic
