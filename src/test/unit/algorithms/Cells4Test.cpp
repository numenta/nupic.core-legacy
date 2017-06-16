/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of unit tests for Segment
 */

#include <set>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>
#include <gtest/gtest.h>

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/Segment.hpp>
#include <nupic/math/ArrayAlgo.hpp> // is_in
#include <nupic/proto/Cells4.capnp.h>


using namespace nupic::algorithms::Cells4;



template <class InputIterator>
std::vector<UInt> _getOrderedSrcCellIndexesForSrcCells(const Segment& segment,
                                                       InputIterator first,
                                                       InputIterator last)
{
  std::vector<UInt> result;

  const std::set<UInt> srcCellsSet(first, last);

  for (UInt i = 0; i < segment.size(); ++i) {
    UInt srcCellIdx = segment[i].srcCellIdx();
    if (is_in(srcCellIdx, srcCellsSet)) {
      result.push_back(srcCellIdx);
    }
  }

  return result;
}


template <class InputIterator>
std::vector<UInt> _getOrderedSynapseIndexesForSrcCells(const Segment& segment,
                                                       InputIterator first,
                                                       InputIterator last)
{
  std::vector<UInt> result;

  const std::set<UInt> srcCellsSet(first, last);

  for (UInt i = 0; i < segment.size(); ++i) {
    UInt srcCellIdx = segment[i].srcCellIdx();
    if (is_in(srcCellIdx, srcCellsSet)) {
      result.push_back(i);
    }
  }

  return result;
}


/**
 * Simple comparison function that does the easy checks. It can be expanded to
 * cover more of the attributes of Cells4 in the future.
 */
bool checkCells4Attributes(const Cells4& c1, const Cells4& c2)
{
  if (c1.nSegments() != c2.nSegments() ||
      c1.nCells() != c2.nCells() ||
      c1.nColumns() != c2.nColumns() ||
      c1.nCellsPerCol() != c2.nCellsPerCol() ||
      c1.getMinThreshold() != c2.getMinThreshold() ||
      c1.getPermConnected() != c2.getPermConnected() ||
      c1.getVerbosity() != c2.getVerbosity() ||
      c1.getMaxAge() != c2.getMaxAge() ||
      c1.getPamLength() != c2.getPamLength() ||
      c1.getMaxInfBacktrack() != c2.getMaxInfBacktrack() ||
      c1.getMaxLrnBacktrack() != c2.getMaxLrnBacktrack() ||

      c1.getPamCounter() != c2.getPamCounter() ||
      c1.getMaxSeqLength() != c2.getMaxSeqLength() ||
      c1.getAvgLearnedSeqLength() != c2.getAvgLearnedSeqLength() ||
      c1.getNLrnIterations() != c2.getNLrnIterations() ||

      c1.getMaxSegmentsPerCell() != c2.getMaxSegmentsPerCell() ||
      c1.getMaxSynapsesPerSegment() != c2.getMaxSynapsesPerSegment() ||
      c1.getCheckSynapseConsistency() != c2.getCheckSynapseConsistency())
  {
    return false;
  }
  return true;
}


TEST(Cells4Test, capnpSerialization)
{
  Cells4 cells(
      10, 2, 1, 1, 1, 1, 0.5, 0.8, 1, 0.1, 0.1, 0, false, -1, true, false);
  std::vector<Real> input1(10, 0.0);
  input1[1] = 1.0;
  input1[4] = 1.0;
  input1[5] = 1.0;
  input1[9] = 1.0;
  std::vector<Real> input2(10, 0.0);
  input2[0] = 1.0;
  input2[2] = 1.0;
  input2[5] = 1.0;
  input2[6] = 1.0;
  std::vector<Real> input3(10, 0.0);
  input3[1] = 1.0;
  input3[3] = 1.0;
  input3[6] = 1.0;
  input3[7] = 1.0;
  std::vector<Real> input4(10, 0.0);
  input4[2] = 1.0;
  input4[4] = 1.0;
  input4[7] = 1.0;
  input4[8] = 1.0;
  std::vector<Real> output(10*2);
  for (UInt i = 0; i < 10; ++i)
  {
    cells.compute(&input1.front(), &output.front(), true, true);
    cells.compute(&input2.front(), &output.front(), true, true);
    cells.compute(&input3.front(), &output.front(), true, true);
    cells.compute(&input4.front(), &output.front(), true, true);
    cells.reset();
  }

  Cells4 secondCells;
  {
    capnp::MallocMessageBuilder message1;
    Cells4Proto::Builder cells4Builder = message1.initRoot<Cells4Proto>();
    cells.write(cells4Builder);
    std::stringstream ss;
    kj::std::StdOutputStream out(ss);
    capnp::writeMessage(out, message1);

    kj::std::StdInputStream in(ss);
    capnp::InputStreamMessageReader message2(in);
    Cells4Proto::Reader cells4Reader = message2.getRoot<Cells4Proto>();
    secondCells.read(cells4Reader);
  }

  NTA_CHECK(checkCells4Attributes(cells, secondCells));

  std::vector<Real> secondOutput(10*2);
  cells.compute(&input1.front(), &output.front(), true, true);
  secondCells.compute(&input1.front(), &secondOutput.front(), true, true);
  for (UInt i = 0; i < 10; ++i)
  {
    ASSERT_EQ(output[i], secondOutput[i]) << "Outputs differ at index " << i;
  }
  NTA_CHECK(checkCells4Attributes(cells, secondCells));
}



/*
 * Test Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment.
 */
TEST(Cells4Test, generateListsOfSynapsesToAdjustForAdaptSegment)
{
  Segment segment;

  const std::set<UInt> srcCells {99, 88, 77, 66, 55, 44, 33, 22, 11, 0};

  segment.addSynapses(srcCells,
                      0.8/*initStrength*/,
                      0.5/*permConnected*/);

  std::set<UInt> synapsesSet {222, 111, 77, 55, 22, 0};

  std::vector<UInt> inactiveSrcCellIdxs {(UInt)-1};
  std::vector<UInt> inactiveSynapseIdxs {(UInt)-1};
  std::vector<UInt> activeSrcCellIdxs {(UInt)-1};
  std::vector<UInt> activeSynapseIdxs {(UInt)-1};

  const std::set<UInt> expectedInactiveSrcCellSet {99, 88, 66, 44, 33, 11};
  const std::set<UInt> expectedActiveSrcCellSet {77, 55, 22, 0};
  const std::set<UInt> expectedSynapsesSet {222, 111};

  const std::vector<UInt> expectedInactiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedInactiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment(
    segment,
    synapsesSet,
    inactiveSrcCellIdxs,
    inactiveSynapseIdxs,
    activeSrcCellIdxs,
    activeSynapseIdxs);

  ASSERT_EQ(expectedSynapsesSet, synapsesSet);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSrcCellIdxs, activeSrcCellIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}



/*
 * Test Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment with new
 * synapes, but no active synapses.
 */
TEST(Cells4Test, generateListsOfSynapsesToAdjustForAdaptSegmentWithOnlyNewSynapses)
{
  Segment segment;

  const std::set<UInt> srcCells {99, 88, 77, 66, 55};

  segment.addSynapses(srcCells,
                      0.8/*initStrength*/,
                      0.5/*permConnected*/);

  std::set<UInt> synapsesSet {222, 111};

  std::vector<UInt> inactiveSrcCellIdxs {(UInt)-1};
  std::vector<UInt> inactiveSynapseIdxs {(UInt)-1};
  std::vector<UInt> activeSrcCellIdxs {(UInt)-1};
  std::vector<UInt> activeSynapseIdxs {(UInt)-1};

  const std::set<UInt> expectedInactiveSrcCellSet {99, 88, 77, 66, 55};
  const std::set<UInt> expectedActiveSrcCellSet {};
  const std::set<UInt> expectedSynapsesSet {222, 111};

  const std::vector<UInt> expectedInactiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedInactiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment(
    segment,
    synapsesSet,
    inactiveSrcCellIdxs,
    inactiveSynapseIdxs,
    activeSrcCellIdxs,
    activeSynapseIdxs);

  ASSERT_EQ(expectedSynapsesSet, synapsesSet);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSrcCellIdxs, activeSrcCellIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}



/*
 * Test Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment with active
 * synapses, but no new synapses.
 */
TEST(Cells4Test, generateListsOfSynapsesToAdjustForAdaptSegmentWithoutNewSynapses)
{
  Segment segment;

  const std::set<UInt> srcCells {99, 88, 77, 66, 55};

  segment.addSynapses(srcCells,
                      0.8/*initStrength*/,
                      0.5/*permConnected*/);

  std::set<UInt> synapsesSet {88, 66};

  std::vector<UInt> inactiveSrcCellIdxs {(UInt)-1};
  std::vector<UInt> inactiveSynapseIdxs {(UInt)-1};
  std::vector<UInt> activeSrcCellIdxs {(UInt)-1};
  std::vector<UInt> activeSynapseIdxs {(UInt)-1};

  const std::set<UInt> expectedInactiveSrcCellSet {99, 77, 55};
  const std::set<UInt> expectedActiveSrcCellSet {88, 66};
  const std::set<UInt> expectedSynapsesSet {};

  const std::vector<UInt> expectedInactiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedInactiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment(
    segment,
    synapsesSet,
    inactiveSrcCellIdxs,
    inactiveSynapseIdxs,
    activeSrcCellIdxs,
    activeSynapseIdxs);

  ASSERT_EQ(expectedSynapsesSet, synapsesSet);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSrcCellIdxs, activeSrcCellIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}



/*
 * Test Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment with active and
 * new synapses, but no inactive synapses.
 */
TEST(Cells4Test, generateListsOfSynapsesToAdjustForAdaptSegmentWithoutInactiveSynapses)
{
  Segment segment;

  const std::set<UInt> srcCells {88, 77, 66};

  segment.addSynapses(srcCells,
                      0.8/*initStrength*/,
                      0.5/*permConnected*/);

  std::set<UInt> synapsesSet {222, 111, 88, 77, 66};

  std::vector<UInt> inactiveSrcCellIdxs {(UInt)-1};
  std::vector<UInt> inactiveSynapseIdxs {(UInt)-1};
  std::vector<UInt> activeSrcCellIdxs {(UInt)-1};
  std::vector<UInt> activeSynapseIdxs {(UInt)-1};

  const std::set<UInt> expectedInactiveSrcCellSet {};
  const std::set<UInt> expectedActiveSrcCellSet {88, 77, 66};
  const std::set<UInt> expectedSynapsesSet {222, 111};

  const std::vector<UInt> expectedInactiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedInactiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment(
    segment,
    synapsesSet,
    inactiveSrcCellIdxs,
    inactiveSynapseIdxs,
    activeSrcCellIdxs,
    activeSynapseIdxs);

  ASSERT_EQ(expectedSynapsesSet, synapsesSet);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSrcCellIdxs, activeSrcCellIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}



/*
 * Test Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment without initial
 * synapses, and only new synapses.
 */
TEST(Cells4Test, generateListsOfSynapsesToAdjustForAdaptSegmentWithoutInitialSynapses)
{
  Segment segment;

  const std::set<UInt> srcCells {};

  segment.addSynapses(srcCells,
                      0.8/*initStrength*/,
                      0.5/*permConnected*/);

  std::set<UInt> synapsesSet {222, 111};

  std::vector<UInt> inactiveSrcCellIdxs {(UInt)-1};
  std::vector<UInt> inactiveSynapseIdxs {(UInt)-1};
  std::vector<UInt> activeSrcCellIdxs {(UInt)-1};
  std::vector<UInt> activeSynapseIdxs {(UInt)-1};

  const std::set<UInt> expectedInactiveSrcCellSet {};
  const std::set<UInt> expectedActiveSrcCellSet {};
  const std::set<UInt> expectedSynapsesSet {222, 111};

  const std::vector<UInt> expectedInactiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedInactiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment(
    segment,
    synapsesSet,
    inactiveSrcCellIdxs,
    inactiveSynapseIdxs,
    activeSrcCellIdxs,
    activeSynapseIdxs);

  ASSERT_EQ(expectedSynapsesSet, synapsesSet);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSrcCellIdxs, activeSrcCellIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}



/*
 * Test Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment with empty
 * update set.
 */
TEST(Cells4Test, generateListsOfSynapsesToAdjustForAdaptSegmentWithEmptySynapseSet)
{
  Segment segment;

  const std::set<UInt> srcCells {88, 77, 66};

  segment.addSynapses(srcCells,
                      0.8/*initStrength*/,
                      0.5/*permConnected*/);

  std::set<UInt> synapsesSet {};

  std::vector<UInt> inactiveSrcCellIdxs {(UInt)-1};
  std::vector<UInt> inactiveSynapseIdxs {(UInt)-1};
  std::vector<UInt> activeSrcCellIdxs {(UInt)-1};
  std::vector<UInt> activeSynapseIdxs {(UInt)-1};

  const std::set<UInt> expectedInactiveSrcCellSet {88, 77, 66};
  const std::set<UInt> expectedActiveSrcCellSet {};
  const std::set<UInt> expectedSynapsesSet {};

  const std::vector<UInt> expectedInactiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedInactiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedInactiveSrcCellSet.begin(),
                                         expectedInactiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSrcCellIdxs =
    _getOrderedSrcCellIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  const std::vector<UInt> expectedActiveSynapseIdxs =
    _getOrderedSynapseIndexesForSrcCells(segment,
                                         expectedActiveSrcCellSet.begin(),
                                         expectedActiveSrcCellSet.end());

  Cells4::_generateListsOfSynapsesToAdjustForAdaptSegment(
    segment,
    synapsesSet,
    inactiveSrcCellIdxs,
    inactiveSynapseIdxs,
    activeSrcCellIdxs,
    activeSynapseIdxs);

  ASSERT_EQ(expectedSynapsesSet, synapsesSet);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSrcCellIdxs, activeSrcCellIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}
