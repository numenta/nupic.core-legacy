/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
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
 * --------------------------------------------------------------------- */

/** @file
 * Unit tests for the RandomDistributedScalarEncoder
 */

#include "gtest/gtest.h"
#include <htm/types/Sdr.hpp>
#include <htm/encoders/RandomDistributedScalarEncoder.hpp>
#include <string>
#include <vector>

using namespace htm;

TEST(RDSE, testConstruct) {
  SDR  A({ 100u, 100u, 3u });
  RDSE_Parameters P;
  P.size       = A.size;
  P.sparsity   = 0.05f;
  P.resolution = 1.23f;
  RDSE R( P );
  R.encode( 3, A );
}

TEST(RDSE, testSerialize) {
  RDSE_Parameters P;
  P.size       = 1000;
  P.sparsity   = 0.05f;
  P.resolution = 1.23f;
  RDSE R1( P );

  std::stringstream buf;
  R1.save( buf );

  SDR A( R1.dimensions );
  R1.encode( 44.4f, A );

  RDSE R2;
  R2.load( buf );
  SDR B( R2.dimensions );
  R2.encode( 44.4f, B );

  ASSERT_EQ( A, B );
}
