/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Implementation of UniformLinkPolicy test
 */

#include "gtest/gtest.h"

#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/types/Fraction.hpp>
#include <nupic/utils/Log.hpp>
#include <utility>

#include <nupic/engine/UniformLinkPolicy.hpp>

using namespace nupic;

enum LinkSide { srcLinkSide, destLinkSide };

struct CoordBounds {
  Coordinate coord;
  size_t dimension;
  std::pair<Fraction, Fraction> bounds;

  CoordBounds(Coordinate c, size_t dim, std::pair<Fraction, Fraction> b)
      : coord(std::move(c)), dimension(dim), bounds(std::move(b)) {}
};

// ---
// We need to access private methods in UniformLinkPolicy, and so
// we create this class as a singleton and declare it a friend
// class in UniformLinkPolicy.
// ---
namespace nupic {
class UniformLinkPolicyInspector {
public:
  bool setAndCheckDimensions(LinkSide setLinkSide, Dimensions setDimensions,
                             Dimensions checkDimensions, std::string linkParams,
                             size_t elementCount = 1) {
    Link dummyLink("UnitTestLink", "", "", "");
    UniformLinkPolicy test(linkParams, &dummyLink);

    // ---
    // Since we're a unit test working in isolation, the infrastructure won't
    // invoke setNodeOutputElementCount() for us; consequently we'll do that
    // directly here.
    // ---
    test.setNodeOutputElementCount(elementCount);

    setLinkSide == srcLinkSide ? test.setSrcDimensions(setDimensions)
                               : test.setDestDimensions(setDimensions);

    Dimensions destDims = test.getDestDimensions();
    Dimensions srcDims = test.getSrcDimensions();

    bool wasExpectedDimensions;

    setLinkSide == srcLinkSide
        ? (wasExpectedDimensions =
               (srcDims == setDimensions && destDims == checkDimensions))
        : (wasExpectedDimensions =
               (srcDims == checkDimensions && destDims == setDimensions));

    return (wasExpectedDimensions);
  }

  bool setDimensionsAndCheckBounds(LinkSide setLinkSide,
                                   Dimensions setDimensions,
                                   std::vector<CoordBounds> checkBoundsVec,
                                   std::string linkParams,
                                   size_t elementCount = 1) {
    Link dummyLink("UnitTestLink", "", "", "");
    UniformLinkPolicy test(linkParams, &dummyLink);

    // ---
    // Since we're a unit test working in isolation, the infrastructure won't
    // invoke setNodeOutputElementCount() for us; consequently we'll do that
    // directly here.
    // ---
    test.setNodeOutputElementCount(elementCount);

    setLinkSide == srcLinkSide ? test.setSrcDimensions(setDimensions)
                               : test.setDestDimensions(setDimensions);

    // ---
    // Since we're a unit test working in isolation, the infrastructure won't
    // invoke initialize() for us; consequently we'll do that directly here.
    // ---
    test.initialize();

    bool allBoundsEqual = true;

    for (auto &elem : checkBoundsVec) {
      std::pair<Fraction, Fraction> testBounds;

      testBounds = test.getInputBoundsForNode(elem.coord, elem.dimension);

      if (testBounds != elem.bounds) {
        allBoundsEqual = false;
      }
    }

    return (allBoundsEqual);
  }
};
} // end namespace nupic

UniformLinkPolicyInspector inspector;

Coordinate makeCoordinate(size_t x, size_t y) {
  Coordinate coord;

  coord.push_back(x);
  coord.push_back(y);

  return (coord);
}

TEST(UniformLinkPolicyTest, StrictMappingOddSource) {
  // ---
  // Check that a strict mapping with an rfSize of 2 fails on odd source
  // dimensions
  // ---
  EXPECT_THROW(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(9, 6),
                                               Dimensions(0, 0),
                                               "{mapping: in, "
                                               "rfSize: [2]}"),
               std::exception);
}

TEST(UniformLinkPolicyTest, StrictMappingDimensions) {
  // ---
  // Check that a strict mapping with an rfSize of 2 calculates proper
  // dimensions when setting the source
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(8, 6),
                                              Dimensions(4, 3),
                                              "{mapping: in, "
                                              "rfSize: [2]}"));
}

TEST(UniformLinkPolicyTest, SpanNoImpactSource) {
  // ---
  // Check that adding in a span with size equal to the source dimensions has
  // no impact on the calculated destination dimensions when setting the source
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(8, 6),
                                              Dimensions(4, 3),
                                              "{mapping: in, "
                                              "rfSize: [2], "
                                              "span: [8,6]}"));
}

TEST(UniformLinkPolicyTest, StrictMappingDestination) {
  // ---
  // Check that a strict mapping with an rfSize of 2 calculates proper
  // dimensions when setting the destination
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(destLinkSide, Dimensions(4, 3),
                                              Dimensions(8, 6),
                                              "{mapping: in, "
                                              "rfSize: [2]}"));
}

TEST(UniformLinkPolicyTest, SpanNoImpactDestination) {
  // ---
  // Check that adding in a span with size equal to the source dimensions has
  // no impact on the calculated destination dimensions when setting the
  // destination
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(destLinkSide, Dimensions(4, 3),
                                              Dimensions(8, 6),
                                              "{mapping: in, "
                                              "rfSize: [2], "
                                              "span: [8,6]}"));
}

TEST(UniformLinkPolicyTest, StrictMappingGranularityDestFails) {
  // ---
  // Check that using a fractional rfSize with a granularity of elements fails
  // when the number of elements is inconsistent with a strict mapping
  // ---
  EXPECT_THROW(inspector.setAndCheckDimensions(destLinkSide, Dimensions(7),
                                               Dimensions(10),
                                               "{mapping: in, "
                                               "rfSize: [1.42857], "
                                               "rfGranularity: elements}",
                                               1),
               std::exception);
}

TEST(UniformLinkPolicyTest, StrictMappingGranularityDestPasses) {
  // ---
  // Check that when using a compatible number of elements, the above test
  // passes
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(destLinkSide, Dimensions(7),
                                              Dimensions(10),
                                              "{mapping: in, "
                                              "rfSize: [1.42857], "
                                              "rfGranularity: elements}",
                                              7));
}

TEST(UniformLinkPolicyTest, StrictMappingGranularitySourceFails) {
  // ---
  // Repeat the above two tests setting the source instead of the destination
  // ---
  EXPECT_THROW(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(10),
                                               Dimensions(7),
                                               "{mapping: in, "
                                               "rfSize: [1.42857], "
                                               "rfGranularity: elements}",
                                               1),
               std::exception);
}

TEST(UniformLinkPolicyTest, StrictMappingGranularitySourcePasses) {
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(10),
                                              Dimensions(7),
                                              "{mapping: in, "
                                              "rfSize: [1.42857], "
                                              "rfGranularity: elements}",
                                              7));
}

TEST(UniformLinkPolicyTest, NonStrictMappingSourcePasses) {
  // ---
  // Check that a non-strict mapping with an rfSize of 2 succeeds on odd source
  // dimensions and returns the expected values.  Specifically, when working in
  // non-strict mode, UniformLinkPolicy should favor a mapping that provides
  // more source nodes in a given destination node than fewer; consequently,
  // for source dimensions of [9, 6] and a rfSize of [2] we would expect
  // dimensions of [4, 3] instead of [5, 3].
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(9, 6),
                                              Dimensions(4, 3),
                                              "{mapping: in, "
                                              "rfSize: [2], "
                                              "strict: false}"));
}

TEST(UniformLinkPolicyTest, NonStrictMappingExpectedDimensions) {
  // ---
  // Check that a non-strict mapping with overlap and a span has the expected
  // dimensions.
  //
  // In the following test, our second dimension is a valid strict mapping with
  // no overlap or span, so we expect it to be 2 given the parameters.  Our
  // first dimension is more complicated.  Given a receptive field of 3 nodes
  // with an overlap of 2 and a span of 4, each set of four source nodes is
  // going to correspond to two destination nodes.  The remaining lone 9th node
  // should, due to non-strict favoring of mappings that provide more source
  // nodes in a given destination node than fewer, be packed into one of the
  // two spans.  Therefore we expect the first dimension to be of size 4.
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(9, 6),
                                              Dimensions(4, 2),
                                              "{mapping: in, "
                                              "rfSize: [3], "
                                              "rfOverlap: [2, 0], "
                                              "span: [4, 0], "
                                              "strict: false}"));
}

TEST(UniformLinkPolicyTest, NonStrictMappingExpectedDimensions2) {
  // ---
  // Repeat the above test using source dimensions of [10, 6].  In this case
  // The remaining 9th and 10th node should, be packed into one each of the
  // two spans.  Therefore we expect the first dimension to be of size 4.
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(10, 6),
                                              Dimensions(4, 2),
                                              "{mapping: in, "
                                              "rfSize: [3], "
                                              "rfOverlap: [2, 0], "
                                              "span: [4, 0], "
                                              "strict: false}"));
}

TEST(UniformLinkPolicyTest, NonStrictMappingExpectedDimensionsEdge) {
  // ---
  // Check the same condition as above, but setting the destination and
  // inducing the source dimensions.  We will test using destination dimensions
  // of [5, 2] since this is an edge case which can not possibly be mapped.
  //
  // We expect source dimensions of [10,2], however we further expect a
  // warning that our specified destination dimensions will result in one
  // of the destination nodes in the first dimension receiving no input.
  // This is because with 10 source nodes, given the parameters, you'd have:
  //
  //  * * * *   * * * * * *
  // | RF  |   | RF  |
  //   | RF  |   | RF  |
  // | SPAN  | | SPAN  |
  //
  // i.e. two spans with two receptive fields and two extra nodes.
  //
  // The mapping of these two extra source nodes to a fifth destination node
  // is implied by the formulas; however given the specified span parameter,
  // it makes no sense to have a 5th destination node in the absence of a
  // sixth, and as is the case, the two extra nodes should be distributed
  // across the two valid spans.  This is what is done, and a warning is
  // issued to indicate that the destination dimensions being set, while
  // being honored due to strict=false, will result in the 5th destination
  // node receiving no input.
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(destLinkSide, Dimensions(5, 2),
                                              Dimensions(10, 6),
                                              "{mapping: in, "
                                              "rfSize: [3], "
                                              "rfOverlap: [2, 0], "
                                              "span: [4, 0], "
                                              "strict: false}"));
}

TEST(UniformLinkPolicyTest, NonStrictMappingSourceDimensions) {
  // ---
  // Test basic non-strict mapping when setting source dimensions.
  //
  // When working in a non-strict mode, UniformLinkPolicy should favor a
  // mapping that provides more source nodes in a given destination node than
  // fewer; consequently we expect dimensions of [4, 3] instead of [5, 4] for
  // the following settings.
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(8, 6),
                                              Dimensions(4, 3),
                                              "{mapping: in, "
                                              "rfSize: [1.7], "
                                              "strict: false}"));
}

TEST(UniformLinkPolicyTest, NonStrictMappingDestinationDimensions) {
  // ---
  // Test basic non-strict mapping when setting destination dimensions.
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(destLinkSide, Dimensions(4, 3),
                                              Dimensions(7, 6),
                                              "{mapping: in, "
                                              "rfSize: [1.7], "
                                              "strict: false}"));
}

TEST(UniformLinkPolicyTest, OverlapOverhangRealisticDimensions) {
  // ---
  // Test overhang and overlap while using realistic image size dimensions.
  // ---
  EXPECT_TRUE(inspector.setAndCheckDimensions(srcLinkSide, Dimensions(320, 240),
                                              Dimensions(41, 31),
                                              "{mapping: in, "
                                              "rfSize: [16], "
                                              "rfOverlap: [8], "
                                              "overhang: [8]}"));
}

TEST(UniformLinkPolicyTest, StrictMappingSplitOverReceptiveFields) {
  // ---
  // Test a strict mapping to make sure the elements are split across
  // receptive fields as expected
  // ---
  std::vector<CoordBounds> expectedBoundVec;

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(0, 0), 0, std::pair<size_t, size_t>(0, 1)));

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(1, 0), 0, std::pair<size_t, size_t>(2, 3)));

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(2, 0), 0, std::pair<size_t, size_t>(4, 5)));

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(3, 0), 0, std::pair<size_t, size_t>(6, 7)));

  EXPECT_TRUE(inspector.setDimensionsAndCheckBounds(
      srcLinkSide, Dimensions(8, 6), expectedBoundVec,
      "{mapping: in, "
      "rfSize: [2], "
      "strict: false}"));
}

TEST(UniformLinkPolicyTest, NonStrictMappingSplitOverReceptiveFields) {
  // ---
  // Test a non-strict mapping to make sure the elements are split across
  // receptive fields as expected
  // ---
  std::vector<CoordBounds> expectedBoundVec;

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(0, 0), 0, std::pair<size_t, size_t>(0, 1)));

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(1, 0), 0, std::pair<size_t, size_t>(2, 3)));

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(2, 0), 0, std::pair<size_t, size_t>(4, 5)));

  expectedBoundVec.push_back(
      CoordBounds(makeCoordinate(3, 0), 0, std::pair<size_t, size_t>(6, 8)));

  EXPECT_TRUE(inspector.setDimensionsAndCheckBounds(
      srcLinkSide, Dimensions(9, 6), expectedBoundVec,
      "{mapping: in, "
      "rfSize: [2], "
      "strict: false}"));
}
