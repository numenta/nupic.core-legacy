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

#include <string>
#include <vector>
#include <nupic/ntypes/Array.hpp>
#include <nupic/encoders/Scalar.hpp>
#include "gtest/gtest.h"

using namespace nupic;

template <typename T>
std::string vec2str(std::vector<T> vec) {
  std::ostringstream oss("");
  for (size_t i = 0; i < vec.size(); i++)
    oss << vec[i];
  return oss.str();
}

std::vector<UInt> getEncoding(Encoder& e, Real64 input) {
  Array inputArray = Array(NTA_BasicType_Real64);
  inputArray.allocateBuffer(1);
  ((Real64*) inputArray.getBuffer())[0] = input;

  auto actualOutput = std::vector<UInt>(e.getWidth());
  e.encodeIntoArray(inputArray, &actualOutput[0], false);
  return actualOutput;
}

typedef struct _SCALAR_VALUE_CASE {
  Real64 input;
  std::vector<UInt> expectedOutput;
} SCALAR_VALUE_CASE;

typedef struct _SCALAR_COMPARE_CASE {
  Real64 a;
  Real64 b;
} SCALAR_COMPARE_CASE;

TEST(PeriodicScalarEncoderTest, BottomUpEncodingPeriodicEncoder)
{
  int n = 14;
  int w = 3;
  double minval = 1;
  double maxval = 8;
  double radius = 0;
  double resolution = 0;
  PeriodicScalarEncoder encoder(w, minval, maxval, n, radius, resolution);

  {
    std::vector<SCALAR_VALUE_CASE> cases =
      {{3,
        {0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
       {3.5,
        {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}},
       {4,
        {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0}},
       {1,
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
       {1.5,
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
       {7,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}},
       {7.5,
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}}};

    for (auto c = cases.begin(); c != cases.end(); c++) {
      auto actualOutput = getEncoding(encoder, c->input);
      for (int i = 0; i < n; i++) {
        ASSERT_EQ(c->expectedOutput[i], actualOutput[i])
          << "For input " << c->input << " and index " << i << std::endl
          << "EXPECTED:" << std::endl
          << vec2str(c->expectedOutput) << std::endl
          << "ACTUAL:" << std::endl
          << vec2str(actualOutput);
      }
    }
  }

  {
    std::vector<SCALAR_COMPARE_CASE> cases =
      {{3.1, 3},
       {3.6, 3.5},
       {3.7, 3.5}};

    for (auto c = cases.begin(); c != cases.end(); c++) {
      auto outputA = getEncoding(encoder, c->a);
      auto outputB = getEncoding(encoder, c->b);
      for (int i = 0; i < n; i++) {
        ASSERT_EQ(outputA[i], outputB[i])
          << "For inputs " << c->a << " and " << c->b << std::endl
          << "A:" << std::endl
          << vec2str(outputA) << std::endl
          << "B:" << std::endl
          << vec2str(outputB);
      }
    }
  }
}

TEST(ScalarEncoderTest, NonPeriodicBottomUp)
{
  int n = 14;
  int w = 5;
  double minval = 1;
  double maxval = 10;
  double radius = 0;
  double resolution = 0;
  bool clipInput = false;
  ScalarEncoder encoder(w, minval, maxval, n, radius, resolution, clipInput);

  {
    std::vector<SCALAR_VALUE_CASE> cases =
      {{1,
        {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
       {2,
        {0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
       {10,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}},};

    for (auto c = cases.begin(); c != cases.end(); c++) {
      auto actualOutput = getEncoding(encoder, c->input);
      for (int i = 0; i < n; i++) {
        ASSERT_EQ(c->expectedOutput[i], actualOutput[i])
          << "For input " << c->input << " and index " << i << std::endl
          << "EXPECTED:" << std::endl
          << vec2str(c->expectedOutput) << std::endl
          << "ACTUAL:" << std::endl
          << vec2str(actualOutput);
      }
    }
  }
}
