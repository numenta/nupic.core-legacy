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
 * Unit tests for the RandomDistributedScalarEncoder
 */

#include "gtest/gtest.h"
#include <nupic/types/Sdr.hpp>
#include <nupic/encoders/RandomDistributedScalarEncoder.hpp>
#include <string>
#include <vector>

using namespace nupic;
using namespace nupic::sdr;
using namespace nupic::encoders;

TEST(RDSE, testConstruct) {
  SDR  A({ 100u, 100u, 3 });
  RDSE_Parameters p;
  p.size       = A.size;
  p.sparsity   = 0.05f;
  p.resolution = 1.23f;
  RDSE R( p );
  R.encode( 3, A );
  // ASSERT_EQ( R.size,     A.size );
  // ASSERT_EQ( R.sparsity, 0.05f );
  // ASSERT_EQ( R.radius,   1.23f );
}

// TEST(RDSE, testErrorChecks) {
//   auto A = RDSE(100u, 0.02f, 20.0f);
//   SDR B({ 444u });
//   ASSERT_ANY_THROW( A.encode(0u, B) );
// }
