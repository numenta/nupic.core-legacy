/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
 * Unit tests for the ScalarEncoder and PeriodicScalarEncoder
 */

#include <string>
#include <vector>
#include <nupic/encoders/ScalarEncoder.hpp>
#include "gtest/gtest.h"

using namespace nupic;

template <typename T>
std::string vec2str(std::vector<T> vec)
{
  std::ostringstream oss("");
  for (size_t i = 0; i < vec.size(); i++)
    oss << vec[i];
  return oss.str();
}

std::vector<Real32> getEncoding(ScalarEncoderBase& e, Real64 input)
{
  auto actualOutput = std::vector<Real32>(e.getOutputWidth());
  e.encodeIntoArray(input, &actualOutput[0]);
  return actualOutput;
}

struct ScalarValueCase
{
  Real64 input;
  std::vector<Real32> expectedOutput;
};

std::vector<Real32> patternFromNZ(int n, std::vector<size_t> patternNZ)
{
  auto v = std::vector<Real32>(n, 0);
  for (auto it = patternNZ.begin(); it != patternNZ.end(); it++)
    {
      v[*it] = 1;
    }
  return v;
}

void doScalarValueCases(ScalarEncoderBase& e, std::vector<ScalarValueCase> cases)
{
  for (auto c = cases.begin(); c != cases.end(); c++)
    {
      auto actualOutput = getEncoding(e, c->input);
      for (int i = 0; i < e.getOutputWidth(); i++)
        {
          EXPECT_EQ(c->expectedOutput[i], actualOutput[i])
            << "For input " << c->input << " and index " << i << std::endl
            << "EXPECTED:" << std::endl
            << vec2str(c->expectedOutput) << std::endl
            << "ACTUAL:" << std::endl
            << vec2str(actualOutput);
        }
    }
}

TEST(ScalarEncoder, ValidScalarInputs)
{
  const int n = 10;
  const int w = 2;
  const double minValue = 10;
  const double maxValue = 20;
  const double radius = 0;
  const double resolution = 0;

  {
    const bool clipInput = false;
    ScalarEncoder encoder(w, minValue, maxValue, n, radius, resolution, clipInput);

    EXPECT_THROW(getEncoding(encoder, 9.9), std::exception);
    EXPECT_NO_THROW(getEncoding(encoder, 10.0));
    EXPECT_NO_THROW(getEncoding(encoder, 20.0));
    EXPECT_THROW(getEncoding(encoder, 20.1), std::exception);
  }

  {
    const bool clipInput = true;
    ScalarEncoder encoder(w, minValue, maxValue, n, radius, resolution, clipInput);

    EXPECT_NO_THROW(getEncoding(encoder, 9.9));
    EXPECT_NO_THROW(getEncoding(encoder, 20.1));
  }
}

TEST(PeriodicScalarEncoder, ValidScalarInputs)
{
  const int n = 10;
  const int w = 2;
  const double minValue = 10;
  const double maxValue = 20;
  const double radius = 0;
  const double resolution = 0;
  PeriodicScalarEncoder encoder(w, minValue, maxValue, n, radius, resolution);

  EXPECT_THROW(getEncoding(encoder, 9.9), std::exception);
  EXPECT_NO_THROW(getEncoding(encoder, 10.0));
  EXPECT_NO_THROW(getEncoding(encoder, 19.9));
  EXPECT_THROW(getEncoding(encoder, 20.0), std::exception);
}

TEST(ScalarEncoder, NonIntegerBucketWidth)
{
  const int n = 7;
  const int w = 3;
  const double minValue = 10;
  const double maxValue = 20;
  const double radius = 0;
  const double resolution = 0;
  const bool clipInput = false;
  ScalarEncoder encoder(w, minValue, maxValue, n, radius, resolution, clipInput);

  std::vector<ScalarValueCase> cases =
    {{10.0, patternFromNZ(n, {0, 1, 2})},
     {20.0, patternFromNZ(n, {4, 5, 6})}};

  doScalarValueCases(encoder, cases);
}

TEST(PeriodicScalarEncoder, NonIntegerBucketWidth)
{
  const int n = 7;
  const int w = 3;
  const double minValue = 10;
  const double maxValue = 20;
  const double radius = 0;
  const double resolution = 0;
  PeriodicScalarEncoder encoder(w, minValue, maxValue, n, radius, resolution);

  std::vector<ScalarValueCase> cases =
    {{10.0, patternFromNZ(n, {6, 0, 1})},
     {19.9, patternFromNZ(n, {5, 6, 0})}};

  doScalarValueCases(encoder, cases);
}

TEST(ScalarEncoder, RoundToNearestMultipleOfResolution)
{
  const int n_in = 0;
  const int w = 3;
  const double minValue = 10;
  const double maxValue = 20;
  const double radius = 0;
  const double resolution = 1;
  const bool clipInput = false;
  ScalarEncoder encoder(w, minValue, maxValue, n_in, radius, resolution, clipInput);

  const int n = 13;
  ASSERT_EQ(n, encoder.getOutputWidth());

  std::vector<ScalarValueCase> cases =
    {{10.00, patternFromNZ(n, {0, 1, 2})},
     {10.49, patternFromNZ(n, {0, 1, 2})},
     {10.50, patternFromNZ(n, {1, 2, 3})},
     {11.49, patternFromNZ(n, {1, 2, 3})},
     {11.50, patternFromNZ(n, {2, 3, 4})},
     {14.49, patternFromNZ(n, {4, 5, 6})},
     {14.50, patternFromNZ(n, {5, 6, 7})},
     {15.49, patternFromNZ(n, {5, 6, 7})},
     {15.50, patternFromNZ(n, {6, 7, 8})},
     {19.49, patternFromNZ(n, {9, 10, 11})},
     {19.50, patternFromNZ(n, {10, 11, 12})},
     {20.00, patternFromNZ(n, {10, 11, 12})}};

  doScalarValueCases(encoder, cases);
}

TEST(PeriodicScalarEncoder, FloorToNearestMultipleOfResolution)
{
  const int n_in = 0;
  const int w = 3;
  const double minValue = 10;
  const double maxValue = 20;
  const double radius = 0;
  const double resolution = 1;
  PeriodicScalarEncoder encoder(w, minValue, maxValue, n_in, radius, resolution);

  const int n = 10;
  ASSERT_EQ(n, encoder.getOutputWidth());

  std::vector<ScalarValueCase> cases =
    {{10.00, patternFromNZ(n, {9, 0, 1})},
     {10.99, patternFromNZ(n, {9, 0, 1})},
     {11.00, patternFromNZ(n, {0, 1, 2})},
     {11.99, patternFromNZ(n, {0, 1, 2})},
     {12.00, patternFromNZ(n, {1, 2, 3})},
     {14.00, patternFromNZ(n, {3, 4, 5})},
     {14.99, patternFromNZ(n, {3, 4, 5})},
     {15.00, patternFromNZ(n, {4, 5, 6})},
     {15.99, patternFromNZ(n, {4, 5, 6})},
     {19.00, patternFromNZ(n, {8, 9, 0})},
     {19.99, patternFromNZ(n, {8, 9, 0})}};

  doScalarValueCases(encoder, cases);
}
