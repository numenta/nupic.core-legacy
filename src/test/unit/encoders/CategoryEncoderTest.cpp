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
  ASSERT_EQ(A.size, 1003u);
  ASSERT_EQ(A.sparsity, 0.02f);
  UNUSED(A.inputSeedMap);

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
  auto A = nupic::CategoryEncoder<string>(1003u, 0.02f);
  SDR B({ A.size });
  SDR C({ A.size });
  A.encode("foobar", B);
  A.encode("foobar", C);
  ASSERT_EQ( B, C );
  SDR D({ A.size });
  ASSERT_LT( D.getOverlap( C ), 10u );
}

TEST(CategoryEncoder, testMinimumOverlap) {
  // Make a lot of categories and make sure they don't conflict.
  UInt num_categories = 1000u;
  nupic::CategoryEncoder<UInt> enc(1000u, 0.05f);

  vector<SDR*> allResults;
  for(UInt category = 0; category < num_categories; category++) {
    SDR *output = new SDR({ enc.size });
    enc.encode( category, *output );
    for(SDR *other : allResults) {
      EXPECT_LT( output->getOverlap( *other ), 0.50f * output->getSum() );
    }
    allResults.push_back( output );
  }
  for(SDR *x : allResults)
    delete x;
}
