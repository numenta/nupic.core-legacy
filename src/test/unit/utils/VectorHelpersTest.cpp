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



#include <vector>
#include <algorithm>
#include "nupic/utils/VectorHelpers.hpp"
#include "gtest/gtest.h"
#include "nupic/types/Types.hpp"

using namespace nupic::utils;
using namespace nupic;
using namespace std;

TEST(VectorHelpers, print_vector) 
{
  std::vector<float> v{1.2, 0.2, 1, 2.2, 0.1};
  VectorHelpers::print_vector<float>(v);
  VectorHelpers::print_vector<float>(v," , ", "Hi there:\n");
  ASSERT_FLOAT_EQ(0.0, 0.0);
  vector<string> str = {"a", "b", "c"};
  VectorHelpers::print_vector<string>(str);
  ASSERT_FLOAT_EQ(0.0, 0.0);
};


TEST(VectorHelpers, castVectorType)
{
  std::vector<float> v{1.2, 0.2, 1, 2.2, 0.1};
  vector<UInt> expected {1, 0, 1, 2, 0};
  vector<UInt> res = VectorHelpers::castVectorType<float, UInt>(v);
  for(UInt i=0; i< res.size(); i++) { //FIXME how compare vectors?
    ASSERT_EQ(res[i], expected[i]);
  }
};


TEST(VectorHelpers, stringToFloatVector)
{
  vector<string> s{"1.2", "0.2", "1", "2.2", "0.1"};
  vector<Real> expected2 {1.2, 0.2, 1.0, 2.2, 0.1};
  vector<Real> res2 = VectorHelpers::stringToFloatVector(s);
  for(UInt i=0; i< res2.size(); i++) { //FIXME how compare vectors?
    ASSERT_EQ(res2[i], expected2[i]);
  }
};


TEST(VectorHelpers, binaryToSparse)
{
  vector<Real> v{0.0,0.0,1.0,1.0,0.0};
  vector<UInt> expected {2, 3};
  vector<UInt> res = VectorHelpers::binaryToSparse<Real>(v);
  for(UInt i=0; i< res.size(); i++) {
    ASSERT_EQ(res[i], expected[i]);
  }
};


TEST(VectorHelpers, sparseToBinary)
{
  vector<Real> expected{0.0,0.0,1.0,1.0,0.0};
  vector<UInt> v {2, 3};
  vector<Real> res = VectorHelpers::sparseToBinary<Real>(v, 5);
  for(UInt i=0; i< res.size(); i++) {
    ASSERT_EQ(res[i], expected[i]);
  }
};


TEST(VectorHelpers, cellsToColumns)
{ // using binary vector 3x3 (3 cols with 3 cells per column)
  vector<UInt> v{0,0,0, 0,1,1, 0,0,1};
  vector<UInt> expected {0, 1, 1};
  vector<UInt> res = VectorHelpers::cellsToColumns(v, 3);
  for(UInt i=0; i< res.size(); i++) {
    ASSERT_EQ(res[i], expected[i]);
  }
};

