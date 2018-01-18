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
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for groupBy
 */

#include <nupic/utils/GroupBy.hpp>
#include "gtest/gtest.h"

using std::tuple;
using std::tie;
using std::vector;

using nupic::groupBy;
using nupic::iterGroupBy;

namespace {

  struct ReturnValue1 {
    int key;
    vector<int> results0;
  };

  TEST(GroupByTest, OneSequence)
  {
    const vector<int> sequence0 = {7, 12, 12, 16};

    auto identity = [](int a) { return a; };

    const vector<ReturnValue1> expectedValues = {
      {7, {7}},
      {12, {12, 12}},
      {16, {16}}
    };

    //
    // groupBy
    //
    size_t i = 0;
    for (auto data : groupBy(sequence0, identity))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0;

      tie(key,
          begin0, end0) = data;

      const ReturnValue1 actualValue =
        {key, {begin0, end0}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);

      i++;
    }

    //
    // iterGroupBy
    //
    i = 0;
    for (auto data : iterGroupBy(
           sequence0.begin(), sequence0.end(), identity))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0;

      tie(key,
          begin0, end0) = data;

      const ReturnValue1 actualValue =
        {key, {begin0, end0}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);

      i++;
    }

    EXPECT_EQ(expectedValues.size(), i);
  }



  struct ReturnValue2 {
    int key;
    vector<int> results0;
    vector<int> results1;
  };

  TEST(GroupByTest, TwoSequences)
  {
    const vector<int> sequence0 = {7, 12, 16};
    const vector<int> sequence1 = {3, 4, 5};

    auto identity = [](int a) { return a; };
    auto times3 = [](int a) { return a*3; };

    const vector<ReturnValue2> expectedValues = {
      {7, {7}, {}},
      {9, {}, {3}},
      {12, {12}, {4}},
      {15, {}, {5}},
      {16, {16}, {}}
    };

    //
    // groupBy
    //
    size_t i = 0;
    for (auto data : groupBy(sequence0, identity,
                             sequence1, times3))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1;

      tie(key,
          begin0, end0,
          begin1, end1) = data;

      const ReturnValue2 actualValue =
        {key, {begin0, end0}, {begin1, end1}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);

      i++;
    }

    //
    // iterGroupBy
    //
    i = 0;
    for (auto data : iterGroupBy(
           sequence0.begin(), sequence0.end(), identity,
           sequence1.begin(), sequence1.end(), times3))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1;

      tie(key,
          begin0, end0,
          begin1, end1) = data;

      const ReturnValue2 actualValue =
        {key, {begin0, end0}, {begin1, end1}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);

      i++;
    }

    EXPECT_EQ(expectedValues.size(), i);
  }



  struct ReturnValue3 {
    int key;
    vector<int> results0;
    vector<int> results1;
    vector<int> results2;
  };

  TEST(GroupByTest, ThreeSequences)
  {
    const vector<int> sequence0 = {7, 12, 16};
    const vector<int> sequence1 = {3, 4, 5};
    const vector<int> sequence2 = {3, 3, 4, 5};

    auto identity = [](int a) { return a; };
    auto times3 = [](int a) { return a*3; };
    auto times4 = [](int a) { return a*4; };

    const vector<ReturnValue3> expectedValues = {
      {7, {7}, {}, {}},
      {9, {}, {3}, {}},
      {12, {12}, {4}, {3, 3}},
      {15, {}, {5}, {}},
      {16, {16}, {}, {4}},
      {20, {}, {}, {5}}
    };

    //
    // groupBy
    //
    size_t i = 0;
    for (auto data : groupBy(sequence0, identity,
                             sequence1, times3,
                             sequence2, times4))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1,
        begin2, end2;

      tie(key,
          begin0, end0,
          begin1, end1,
          begin2, end2) = data;

      const ReturnValue3 actualValue =
        {key, {begin0, end0}, {begin1, end1}, {begin2, end2}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
      EXPECT_EQ(expectedValues[i].results2, actualValue.results2);

      i++;
    }

    //
    // iterGroupBy
    //
    i = 0;
    for (auto data : iterGroupBy(
           sequence0.begin(), sequence0.end(), identity,
           sequence1.begin(), sequence1.end(), times3,
           sequence2.begin(), sequence2.end(), times4))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1,
        begin2, end2;

      tie(key,
          begin0, end0,
          begin1, end1,
          begin2, end2) = data;

      const ReturnValue3 actualValue =
        {key, {begin0, end0}, {begin1, end1}, {begin2, end2}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
      EXPECT_EQ(expectedValues[i].results2, actualValue.results2);

      i++;
    }

    EXPECT_EQ(expectedValues.size(), i);
  }


  struct ReturnValue4 {
    int key;
    vector<int> results0;
    vector<int> results1;
    vector<int> results2;
    vector<int> results3;
  };

  TEST(GroupByTest, FourSequences)
  {
    const vector<int> sequence0 = {7, 12, 16};
    const vector<int> sequence1 = {3, 4, 5};
    const vector<int> sequence2 = {3, 3, 4, 5};
    const vector<int> sequence3 = {3, 3, 4, 5};

    auto identity = [](int a) { return a; };
    auto times3 = [](int a) { return a*3; };
    auto times4 = [](int a) { return a*4; };
    auto times5 = [](int a) { return a*5; };

    const vector<ReturnValue4> expectedValues = {
      {7, {7}, {}, {}, {}},
      {9, {}, {3}, {}, {}},
      {12, {12}, {4}, {3, 3}, {}},
      {15, {}, {5}, {}, {3, 3}},
      {16, {16}, {}, {4}, {}},
      {20, {}, {}, {5}, {4}},
      {25, {}, {}, {}, {5}}
    };

    //
    // groupBy
    //
    size_t i = 0;
    for (auto data : groupBy(sequence0, identity,
                             sequence1, times3,
                             sequence2, times4,
                             sequence3, times5))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1,
        begin2, end2,
        begin3, end3;

      tie(key,
          begin0, end0,
          begin1, end1,
          begin2, end2,
          begin3, end3) = data;

      const ReturnValue4 actualValue =
        {key, {begin0, end0}, {begin1, end1}, {begin2, end2}, {begin3, end3}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
      EXPECT_EQ(expectedValues[i].results2, actualValue.results2);
      EXPECT_EQ(expectedValues[i].results3, actualValue.results3);

      i++;
    }

    //
    // iterGroupBy
    //
    i = 0;
    for (auto data : iterGroupBy(
           sequence0.begin(), sequence0.end(), identity,
           sequence1.begin(), sequence1.end(), times3,
           sequence2.begin(), sequence2.end(), times4,
           sequence3.begin(), sequence3.end(), times5))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1,
        begin2, end2,
        begin3, end3;

      tie(key,
          begin0, end0,
          begin1, end1,
          begin2, end2,
          begin3, end3) = data;

      const ReturnValue4 actualValue =
        {key, {begin0, end0}, {begin1, end1}, {begin2, end2}, {begin3, end3}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
      EXPECT_EQ(expectedValues[i].results2, actualValue.results2);
      EXPECT_EQ(expectedValues[i].results3, actualValue.results3);

      i++;
    }

    EXPECT_EQ(expectedValues.size(), i);
  }



  struct ReturnValue5 {
    int key;
    vector<int> results0;
    vector<int> results1;
    vector<int> results2;
    vector<int> results3;
    vector<int> results4;
  };

  TEST(GroupByTest, FiveSequences)
  {
    const vector<int> sequence0 = {7, 12, 16};
    const vector<int> sequence1 = {3, 4, 5};
    const vector<int> sequence2 = {3, 3, 4, 5};
    const vector<int> sequence3 = {3, 3, 4, 5};
    const vector<int> sequence4 = {2, 2, 3};

    auto identity = [](int a) { return a; };
    auto times3 = [](int a) { return a*3; };
    auto times4 = [](int a) { return a*4; };
    auto times5 = [](int a) { return a*5; };
    auto times6 = [](int a) { return a*6; };

    const vector<ReturnValue5> expectedValues = {
      {7, {7}, {}, {}, {}, {}},
      {9, {}, {3}, {}, {}, {}},
      {12, {12}, {4}, {3, 3}, {}, {2, 2}},
      {15, {}, {5}, {}, {3, 3}, {}},
      {16, {16}, {}, {4}, {}, {}},
      {18, {}, {}, {}, {}, {3}},
      {20, {}, {}, {5}, {4}, {}},
      {25, {}, {}, {}, {5}, {}}
    };

    //
    // groupBy
    //
    size_t i = 0;
    for (auto data : groupBy(sequence0, identity,
                             sequence1, times3,
                             sequence2, times4,
                             sequence3, times5,
                             sequence4, times6))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1,
        begin2, end2,
        begin3, end3,
        begin4, end4;

      tie(key,
          begin0, end0,
          begin1, end1,
          begin2, end2,
          begin3, end3,
          begin4, end4) = data;

      const ReturnValue5 actualValue =
        {key,
         {begin0, end0}, {begin1, end1},
         {begin2, end2}, {begin3, end3},
         {begin4, end4}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
      EXPECT_EQ(expectedValues[i].results2, actualValue.results2);
      EXPECT_EQ(expectedValues[i].results3, actualValue.results3);
      EXPECT_EQ(expectedValues[i].results4, actualValue.results4);

      i++;
    }

    //
    // iterGroupBy
    //
    i = 0;
    for (auto data : iterGroupBy(
           sequence0.begin(), sequence0.end(), identity,
           sequence1.begin(), sequence1.end(), times3,
           sequence2.begin(), sequence2.end(), times4,
           sequence3.begin(), sequence3.end(), times5,
           sequence4.begin(), sequence4.end(), times6))
    {
      int key;
      vector<int>::const_iterator
        begin0, end0,
        begin1, end1,
        begin2, end2,
        begin3, end3,
        begin4, end4;

      tie(key,
          begin0, end0,
          begin1, end1,
          begin2, end2,
          begin3, end3,
          begin4, end4) = data;

      const ReturnValue5 actualValue =
        {key,
         {begin0, end0}, {begin1, end1},
         {begin2, end2}, {begin3, end3},
         {begin4, end4}};

      EXPECT_EQ(expectedValues[i].key, actualValue.key);
      EXPECT_EQ(expectedValues[i].results0, actualValue.results0);
      EXPECT_EQ(expectedValues[i].results1, actualValue.results1);
      EXPECT_EQ(expectedValues[i].results2, actualValue.results2);
      EXPECT_EQ(expectedValues[i].results3, actualValue.results3);
      EXPECT_EQ(expectedValues[i].results4, actualValue.results4);

      i++;
    }

    EXPECT_EQ(expectedValues.size(), i);
  }
}
