/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
 *               2019, David McDougall
 *
 * Unless you have an agreement with Numenta, Inc., for a separate license for
 * this software code, the following terms and conditions apply:
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
 * Unit tests for the ScalarEncoder
 */

#include "gtest/gtest.h"
#include <htm/encoders/ScalarEncoder.hpp>
#include <vector>

namespace testing {
    
using namespace htm;


struct ScalarValueCase
{
  Real64 input;
  std::vector<UInt> expectedOutput; // Sparse indices of active bits.
};

void doScalarValueCases(ScalarEncoder& e, std::vector<ScalarValueCase> cases)
{
  for( auto c : cases )
  {
    SDR expectedOutput( e.dimensions );
    std::sort( c.expectedOutput.begin(), c.expectedOutput.end() );
    expectedOutput.setSparse( c.expectedOutput );

    SDR actualOutput( e.dimensions );
    e.encode( c.input, actualOutput );

    EXPECT_EQ( actualOutput, expectedOutput );
  }
}


TEST(ScalarEncoder, testClippingInputs) {
  ScalarEncoderParameters p;
  p.size       = 10;
  p.activeBits = 2;
  p.minimum    = 10;
  p.maximum    = 20;

  SDR output({ 10 });
  {
    p.clipInput = false;
    ScalarEncoder e( p );

    EXPECT_ANY_THROW(e.encode( 9.9f, output));
    EXPECT_NO_THROW( e.encode(10.0f, output));
    EXPECT_NO_THROW( e.encode(20.0f, output));
    EXPECT_ANY_THROW(e.encode(20.1f, output));
  }

  {
    p.clipInput = true;
    ScalarEncoder e( p );

    EXPECT_NO_THROW(e.encode( 9.9f, output));
    EXPECT_NO_THROW(e.encode(20.1f, output));
  }
}

TEST(ScalarEncoder, ValidScalarInputs) {
  ScalarEncoderParameters p;
  p.size       = 10;
  p.activeBits = 2;
  p.minimum    = 10;
  p.maximum    = 20;
  SDR output({ 10 });
  ScalarEncoder e( p );

  EXPECT_ANY_THROW(e.encode( 9.9f, output));
  EXPECT_NO_THROW( e.encode(10.0f, output));
  EXPECT_NO_THROW( e.encode(19.9f, output));
  EXPECT_ANY_THROW(e.encode(20.0001f, output));
}

TEST(ScalarEncoder, NonIntegerBucketWidth) {
  ScalarEncoderParameters p;
  p.size       = 7;
  p.activeBits = 3;
  p.minimum    = 10.0;
  p.maximum    = 20.0;
  ScalarEncoder encoder( p );

  std::vector<ScalarValueCase> cases = {{10.0, {0, 1, 2}},
                                        {20.0, {4, 5, 6}}};

  doScalarValueCases(encoder, cases);
}

TEST(ScalarEncoder, RoundToNearestMultipleOfResolution) {
  ScalarEncoderParameters p;
  p.activeBits = 3;
  p.minimum    = 10.0;
  p.maximum    = 20.0;
  p.resolution = 1;
  ScalarEncoder encoder( p );

  ASSERT_EQ(encoder.parameters.size, 13u);

  std::vector<ScalarValueCase> cases = {
      {10.00f, {0, 1, 2}},
      {10.49f, {0, 1, 2}},
      {10.50f, {1, 2, 3}},
      {11.49f, {1, 2, 3}},
      {11.50f, {2, 3, 4}},
      {14.49f, {4, 5, 6}},
      {14.50f, {5, 6, 7}},
      {15.49f, {5, 6, 7}},
      {15.50f, {6, 7, 8}},
      {19.00f, {9, 10, 11}},
      {19.49f, {9, 10, 11}},
      {19.50f, {10, 11, 12}},
      {20.00f, {10, 11, 12}}};

  doScalarValueCases(encoder, cases);
}

TEST(ScalarEncoder, PeriodicRoundNearestMultipleOfResolution) {
  ScalarEncoderParameters p;
  p.activeBits = 3;
  p.minimum    = 10.0;
  p.maximum    = 20.0;
  p.resolution = 1;
  p.periodic   = true;
  ScalarEncoder encoder( p );

  ASSERT_EQ(encoder.parameters.size, 10u);

  std::vector<ScalarValueCase> cases = {
      {10.00f, {0, 1, 2}},
      {10.49f, {0, 1, 2}},
      {10.50f, {1, 2, 3}},
      {11.49f, {1, 2, 3}},
      {11.50f, {2, 3, 4}},
      {14.49f, {4, 5, 6}},
      {14.50f, {5, 6, 7}},
      {15.49f, {5, 6, 7}},
      {15.50f, {6, 7, 8}},
      {19.49f, {9, 0, 1}},
      {19.50f, {0, 1, 2}},
      {20.00f, {0, 1, 2}}};

  doScalarValueCases(encoder, cases);
}

TEST(ScalarEncoder, Serialization) {
  std::vector<ScalarEncoder*> inputs;
  ScalarEncoderParameters p;
  p.minimum    = -1.234;
  p.maximum    = 12.34;
  p.activeBits = 34;
  p.radius     = .1337;
  inputs.push_back( new ScalarEncoder( p ) );
  p.clipInput = true;
  inputs.push_back( new ScalarEncoder( p ) );
  p.clipInput = false;
  p.periodic  = true;
  inputs.push_back( new ScalarEncoder( p ) );
  p.radius     = 0.0f;
  p.resolution = .1337;
  inputs.push_back( new ScalarEncoder( p ) );
  ScalarEncoderParameters q;
  q.minimum  = -1.0f;
  q.maximum  =  1.0003f;
  q.size     = 100u;
  q.sparsity = 0.15f;
  inputs.push_back( new ScalarEncoder( q ) );

  for( const auto enc1 : inputs ) {
    std::stringstream buf;
    enc1->save( buf, JSON );
  
    //std::cerr << "SERIALIZED:" << std::endl << buf.str() << std::endl;
    buf.seekg(0);

    ScalarEncoder enc2;
    enc2.load( buf, JSON );

    const auto &p1 = enc1->parameters;
    const auto &p2 = enc2.parameters;
    EXPECT_EQ(  p1.size,       p2.size);
    EXPECT_EQ(  p1.activeBits, p2.activeBits);
    EXPECT_EQ(  p1.periodic,   p2.periodic);
    EXPECT_EQ(  p1.clipInput,  p2.clipInput);
    EXPECT_NEAR(p1.minimum,    p2.minimum,       1.0f / 100000 );
    EXPECT_NEAR(p1.maximum,    p2.maximum,       1.0f / 100000 );
    EXPECT_NEAR(p1.resolution, p2.resolution,    1.0f / 100000 );
    EXPECT_NEAR(p1.sparsity,   p2.sparsity,      1.0f / 100000 );
    EXPECT_NEAR(p1.radius,     p2.radius,        1.0f / 100000 );
    delete enc1;
  }
}

}
