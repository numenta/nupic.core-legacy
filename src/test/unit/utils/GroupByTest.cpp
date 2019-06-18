/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * ---------------------------------------------------------------------- */

/** @file
 * Implementation of unit tests for groupBy
 */

#include "gtest/gtest.h"
#include <htm/utils/GroupBy.hpp>

namespace testing {

using std::tie;
using std::tuple;
using std::vector;

using htm::groupBy;

struct ReturnValue3 {
  int key;
  vector<int> results0;
  vector<int> results1;
  vector<int> results2;
};

TEST(GroupByTest, ThreeSequences) {
  const vector<int> sequence0 = {7, 12, 16};
  const vector<int> sequence1 = {3, 4, 5};
  const vector<int> sequence2 = {3, 3, 4, 5};

  auto identity = [](int a) { return a; };
  auto times3 = [](int a) { return a * 3; };
  auto times4 = [](int a) { return a * 4; };

  const vector<ReturnValue3> expectedValues = {
      {7, {7}, { }, { }},  
      {9, { }, {3}, { }},    
      {12,{12},{4}, {3, 3}},
      {15,{ }, {5}, { }}, 
      {16,{16},{ }, {4}}, 
      {20,{ }, { }, {5}}
  };

  //
  // groupBy
  //
  size_t i = 0;
  for (auto data : groupBy(sequence0, identity, 
			   sequence1, times3, 
			   sequence2, times4)) {
    int key;
    vector<int>::const_iterator begin0, end0, begin1, end1, begin2, end2;

    tie(key, begin0, end0, begin1, end1, begin2, end2) = data;

    const ReturnValue3 actualValue = {
        key, {begin0, end0}, {begin1, end1}, {begin2, end2}};

    EXPECT_EQ(expectedValues[i].key,      actualValue.key);
    EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
    EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
    EXPECT_EQ(expectedValues[i].results2, actualValue.results2);

    i++;
  }
  EXPECT_EQ(expectedValues.size(), i);
}

} // namespace
