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

/** @file
 * Unit tests for Topology.hpp
 */

#include <nupic/math/Topology.hpp>
#include "gtest/gtest.h"

using std::vector;
using namespace nupic;
using namespace nupic::math::topology;

namespace {
  TEST(TopologyTest, IndexFromCoordinates)
  {
    EXPECT_EQ(0, indexFromCoordinates({0}, {100}));
    EXPECT_EQ(50, indexFromCoordinates({50}, {100}));
    EXPECT_EQ(99, indexFromCoordinates({99}, {100}));

    EXPECT_EQ(0, indexFromCoordinates({0, 0}, {100, 80}));
    EXPECT_EQ(10, indexFromCoordinates({0, 10}, {100, 80}));
    EXPECT_EQ(80, indexFromCoordinates({1, 0}, {100, 80}));
    EXPECT_EQ(90, indexFromCoordinates({1, 10}, {100, 80}));

    EXPECT_EQ(0, indexFromCoordinates({0, 0, 0}, {100, 10, 8}));
    EXPECT_EQ(7, indexFromCoordinates({0, 0, 7}, {100, 10, 8}));
    EXPECT_EQ(8, indexFromCoordinates({0, 1, 0}, {100, 10, 8}));
    EXPECT_EQ(80, indexFromCoordinates({1, 0, 0}, {100, 10, 8}));
    EXPECT_EQ(88, indexFromCoordinates({1, 1, 0}, {100, 10, 8}));
    EXPECT_EQ(89, indexFromCoordinates({1, 1, 1}, {100, 10, 8}));
  }

  TEST(TopologyTest, CoordinatesFromIndex)
  {
    EXPECT_EQ(vector<UInt>({0}), coordinatesFromIndex(0, {100}));
    EXPECT_EQ(vector<UInt>({50}), coordinatesFromIndex(50, {100}));
    EXPECT_EQ(vector<UInt>({99}), coordinatesFromIndex(99, {100}));

    EXPECT_EQ(vector<UInt>({0, 0}), coordinatesFromIndex(0, {100, 80}));
    EXPECT_EQ(vector<UInt>({0, 10}), coordinatesFromIndex(10, {100, 80}));
    EXPECT_EQ(vector<UInt>({1, 0}), coordinatesFromIndex(80, {100, 80}));
    EXPECT_EQ(vector<UInt>({1, 10}), coordinatesFromIndex(90, {100, 80}));

    EXPECT_EQ(vector<UInt>({0, 0, 0}), coordinatesFromIndex(0, {100, 10, 8}));
    EXPECT_EQ(vector<UInt>({0, 0, 7}), coordinatesFromIndex(7, {100, 10, 8}));
    EXPECT_EQ(vector<UInt>({0, 1, 0}), coordinatesFromIndex(8, {100, 10, 8}));
    EXPECT_EQ(vector<UInt>({1, 0, 0}), coordinatesFromIndex(80, {100, 10, 8}));
    EXPECT_EQ(vector<UInt>({1, 1, 0}), coordinatesFromIndex(88, {100, 10, 8}));
    EXPECT_EQ(vector<UInt>({1, 1, 1}), coordinatesFromIndex(89, {100, 10, 8}));
  }

  // ==========================================================================
  // NEIGHBORHOOD
  // ==========================================================================

  void expectNeighborhoodIndices(
    const vector<UInt>& centerCoords,
    const vector<UInt>& dimensions,
    UInt radius,
    const vector<UInt>& expected)
  {
    const UInt centerIndex = indexFromCoordinates(centerCoords, dimensions);

    int i = 0;
    for (UInt index : Neighborhood(centerIndex, radius, dimensions))
    {
      EXPECT_EQ(expected[i], index);
      i++;
    }

    EXPECT_EQ(expected.size(), i);
  }

  void expectNeighborhoodCoords(
    const vector<UInt>& centerCoords,
    const vector<UInt>& dimensions,
    UInt radius,
    const vector<vector<UInt> >& expected)
  {
    const UInt centerIndex = indexFromCoordinates(centerCoords, dimensions);

    int i = 0;
    for (UInt index : Neighborhood(centerIndex, radius, dimensions))
    {
      EXPECT_EQ(indexFromCoordinates(expected[i], dimensions), index);
      i++;
    }

    EXPECT_EQ(expected.size(), i);
  }

  TEST(TopologyTest, NeighborhoodOfOrigin1D)
  {
    expectNeighborhoodIndices(
      /*centerCoords*/ {0},
      /*dimensions*/ {100},
      /*radius*/ 2,
      /*expected*/ {0, 1, 2});
  }

  TEST(TopologyTest, NeighborhoodOfOrigin2D)
  {
    expectNeighborhoodCoords(
      /*centerCoords*/ {0, 0},
      /*dimensions*/ {100, 80},
      /*radius*/ 2,
      /*expected*/ {{0, 0}, {0, 1}, {0, 2},
                    {1, 0}, {1, 1}, {1, 2},
                    {2, 0}, {2, 1}, {2, 2}});
  }

  TEST(TopologyTest, NeighborhoodOfOrigin3D)
  {
    expectNeighborhoodCoords(
      /*centerCoords*/ {0, 0, 0},
      /*dimensions*/ {100, 80, 60},
      /*radius*/ 1,
      /*expected*/ {{0, 0, 0}, {0, 0, 1},
                    {0, 1, 0}, {0, 1, 1},
                    {1, 0, 0}, {1, 0, 1},
                    {1, 1, 0}, {1, 1, 1}});
  }

  TEST(TopologyTest, NeighborhoodOfMiddle1D)
  {
    expectNeighborhoodIndices(
      /*centerCoords*/ {50},
      /*dimensions*/ {100},
      /*radius*/ 1,
      /*expected*/ {49, 50, 51});
  }

  TEST(TopologyTest, NeighborhoodOfMiddle2D)
  {
    expectNeighborhoodCoords(
      /*centerCoords*/ {50, 50},
      /*dimensions*/ {100, 80},
      /*radius*/ 1,
      /*expected*/ {{49, 49}, {49, 50}, {49, 51},
                    {50, 49}, {50, 50}, {50, 51},
                    {51, 49}, {51, 50}, {51, 51}});
  }

  TEST(TopologyTest, NeighborhoodOfEnd2D)
  {
    expectNeighborhoodCoords(
      /*centerCoords*/ {99, 79},
      /*dimensions*/ {100, 80},
      /*radius*/ 2,
      /*expected*/ {{97, 77}, {97, 78}, {97, 79},
                    {98, 77}, {98, 78}, {98, 79},
                    {99, 77}, {99, 78}, {99, 79}});
  }

  TEST(TopologyTest, NeighborhoodWiderThanWorld)
  {
    expectNeighborhoodCoords(
      /*centerCoords*/ {0, 0},
      /*dimensions*/ {3, 2},
      /*radius*/ 3,
      /*expected*/ {{0, 0}, {0, 1},
                    {1, 0}, {1, 1},
                    {2, 0}, {2, 1}});
  }

  TEST(TopologyTest, NeighborhoodRadiusZero)
  {
    expectNeighborhoodIndices(
      /*centerCoords*/ {0},
      /*dimensions*/ {100},
      /*radius*/ 0,
      /*expected*/ {0});

    expectNeighborhoodCoords(
      /*centerCoords*/ {0, 0},
      /*dimensions*/ {100, 80},
      /*radius*/ 0,
      /*expected*/ {{0, 0}});

    expectNeighborhoodCoords(
      /*centerCoords*/ {0, 0, 0},
      /*dimensions*/ {100, 80, 60},
      /*radius*/ 0,
      /*expected*/ {{0, 0, 0}});
  }

  TEST(TopologyTest, NeighborhoodDimensionOne)
  {
    expectNeighborhoodCoords(
      /*centerCoords*/ {5, 0},
      /*dimensions*/ {10, 1},
      /*radius*/ 1,
      /*expected*/ {{4, 0}, {5, 0}, {6, 0}});

    expectNeighborhoodCoords(
      /*centerCoords*/ {5, 0, 0},
      /*dimensions*/ {10, 1, 1},
      /*radius*/ 1,
      /*expected*/ {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}});
  }


  // ==========================================================================
  // WRAPPING NEIGHBORHOOD
  // ==========================================================================

  void expectWrappingNeighborhoodIndices(
    const vector<UInt>& centerCoords,
    const vector<UInt>& dimensions,
    UInt radius,
    const vector<UInt>& expected)
  {
    const UInt centerIndex = indexFromCoordinates(centerCoords, dimensions);

    int i = 0;
    for (UInt index : WrappingNeighborhood(centerIndex, radius, dimensions))
    {
      EXPECT_EQ(expected[i], index);
      i++;
    }

    EXPECT_EQ(expected.size(), i);
  }

  void expectWrappingNeighborhoodCoords(
    const vector<UInt>& centerCoords,
    const vector<UInt>& dimensions,
    UInt radius,
    const vector<vector<UInt> >& expected)
  {
    const UInt centerIndex = indexFromCoordinates(centerCoords, dimensions);

    int i = 0;
    for (UInt index : WrappingNeighborhood(centerIndex, radius, dimensions))
    {
      EXPECT_EQ(indexFromCoordinates(expected[i], dimensions), index);
      i++;
    }

    EXPECT_EQ(expected.size(), i);
  }

  TEST(TopologyTest, WrappingNeighborhoodOfOrigin1D)
  {
    expectWrappingNeighborhoodIndices(
      /*centerCoords*/ {0},
      /*dimensions*/ {100},
      /*radius*/ 1,
      /*expected*/ {99, 0, 1});
  }

  TEST(TopologyTest, WrappingNeighborhoodOfOrigin2D)
  {
    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {0, 0},
      /*dimensions*/ {100, 80},
      /*radius*/ 1,
      /*expected*/ {{99, 79}, {99, 0}, {99, 1},
                    {0, 79}, {0, 0}, {0, 1},
                    {1, 79}, {1, 0}, {1, 1}});
  }

  TEST(TopologyTest, WrappingNeighborhoodOfOrigin3D)
  {
    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {0, 0, 0},
      /*dimensions*/ {100, 80, 60},
      /*radius*/ 1,
      /*expected*/ {{99, 79, 59}, {99, 79, 0}, {99, 79, 1},
                    {99, 0, 59}, {99, 0, 0}, {99, 0, 1},
                    {99, 1, 59}, {99, 1, 0}, {99, 1, 1},
                    {0, 79, 59}, {0, 79, 0}, {0, 79, 1},
                    {0, 0, 59}, {0, 0, 0}, {0, 0, 1},
                    {0, 1, 59}, {0, 1, 0}, {0, 1, 1},
                    {1, 79, 59}, {1, 79, 0}, {1, 79, 1},
                    {1, 0, 59}, {1, 0, 0}, {1, 0, 1},
                    {1, 1, 59}, {1, 1, 0}, {1, 1, 1}});
  }

  TEST(TopologyTest, WrappingNeighborhoodOfMiddle1D)
  {
    expectWrappingNeighborhoodIndices(
      /*centerCoords*/ {50},
      /*dimensions*/ {100},
      /*radius*/ 1,
      /*expected*/ {49, 50, 51});
  }

  TEST(TopologyTest, WrappingNeighborhoodOfMiddle2D)
  {
    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {50, 50},
      /*dimensions*/ {100, 80},
      /*radius*/ 1,
      /*expected*/{{49, 49}, {49, 50}, {49, 51},
                   {50, 49}, {50, 50}, {50, 51},
                   {51, 49}, {51, 50}, {51, 51}});
  }

  TEST(TopologyTest, WrappingNeighborhoodOfEnd2D)
  {
    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {99, 79},
      /*dimensions*/ {100, 80},
      /*radius*/ 1,
      /*expected*/{{98, 78}, {98, 79}, {98, 0},
                   {99, 78}, {99, 79}, {99, 0},
                   {0, 78}, {0, 79}, {0, 0}});
  }

  TEST(TopologyTest, WrappingNeighborhoodWiderThanWorld)
  {
    // The order is weird because it starts walking from {-3, -3} and avoids
    // walking the same point twice.
    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {0, 0},
      /*dimensions*/ {3, 2},
      /*radius*/ 3,
      /*expected*/{{0, 1}, {0, 0},
                   {1, 1}, {1, 0},
                   {2, 1}, {2, 0}});
  }

  TEST(TopologyTest, WrappingNeighborhoodRadiusZero)
  {
    expectWrappingNeighborhoodIndices(
      /*centerCoords*/ {0},
      /*dimensions*/ {100},
      /*radius*/ 0,
      /*expected*/ {0});

    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {0, 0},
      /*dimensions*/ {100, 80},
      /*radius*/ 0,
      /*expected*/ {{0, 0}});

    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {0, 0, 0},
      /*dimensions*/ {100, 80, 60},
      /*radius*/ 0,
      /*expected*/ {{0, 0, 0}});
  }

  TEST(TopologyTest, WrappingNeighborhoodDimensionOne)
  {
    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {5, 0},
      /*dimensions*/ {10, 1},
      /*radius*/ 1,
      /*expected*/ {{4, 0}, {5, 0}, {6, 0}});

    expectWrappingNeighborhoodCoords(
      /*centerCoords*/ {5, 0, 0},
      /*dimensions*/ {10, 1, 1},
      /*radius*/ 1,
      /*expected*/ {{4, 0, 0}, {5, 0, 0}, {6, 0, 0}});
  }
}
