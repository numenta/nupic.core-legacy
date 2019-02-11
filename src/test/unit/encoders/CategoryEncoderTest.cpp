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
 * Unit tests for the CategoryEncoder
 */

#include "gtest/gtest.h"
#include <nupic/types/Sdr.hpp>
#include <nupic/encoders/CategoryEncoder.hpp>
#include <string>
#include <vector>

using namespace nupic;

TEST(CategoryEncoder, testErrorChecks) {
  auto A = nupic::CategoryEncoder<UInt>(100u, 0.02f);
  SDR B({ 444u });
  ASSERT_ANY_THROW( A.encode(0u, B) );
}

TEST(CategoryEncoder, testIntegers) {
  auto A = nupic::CategoryEncoder<UInt>(1003u, 0.02f);
  ASSERT(A.size == 1003u);
  ASSERT(A.sparsity == 0.02f);
  UNUSED(A.map);

  SDR B({ A.size });
  A.encode( 1234u, B );
  ASSERT_EQ( B.getSum(), 20u );
  SDR C({ A.size });
  for(auto i = 0u; i < 10; i++) {
    A.encode( i, C );
    ASSERT_LT( B.getOverlap( C ), 10u );
  }
}

TEST(CategoryEncoder, testStrings) {
  // nupic
}
