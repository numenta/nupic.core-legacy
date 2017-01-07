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
 * ----------------------------------------------------------------------
 */

/** @file
 * Definitions for SparseMatrixConnections class
 */

#include <nupic/math/SparseMatrixConnections.hpp>

using namespace nupic;

SparseMatrixConnections::SparseMatrixConnections(
  UInt32 numCells, UInt32 numAxons)
  : SegmentMatrixAdapter<SparseMatrix<UInt32, Real32, Int32, Real64>>(numCells,
                                                                      numAxons)
{}


void SparseMatrixConnections::computeActivity(
  const UInt32* activeAxons_begin, const UInt32* activeAxons_end,
  UInt32* overlaps_begin) const
{
  matrix.rightVecSumAtNZSparse(
    activeAxons_begin, activeAxons_end,
    overlaps_begin);
}

void SparseMatrixConnections::computeActivity(
  const UInt32* activeAxons_begin, const UInt32* activeAxons_end,
  Real32 permanenceThreshold, UInt32* overlaps_begin) const
{
  matrix.rightVecSumAtNZGteThresholdSparse(
    activeAxons_begin, activeAxons_end,
    overlaps_begin, permanenceThreshold);
}

void SparseMatrixConnections::adjustSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* activeAxons_begin, const UInt32* activeAxons_end,
  Real32 activePermanenceDelta, Real32 inactivePermanenceDelta)
{
  matrix.incrementNonZerosOnOuter(
    segments_begin, segments_end,
    activeAxons_begin, activeAxons_end,
    activePermanenceDelta);

  matrix.incrementNonZerosOnRowsExcludingCols(
    segments_begin, segments_end,
    activeAxons_begin, activeAxons_end,
    inactivePermanenceDelta);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::adjustActiveSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* activeAxons_begin, const UInt32* activeAxons_end,
  Real32 permanenceDelta)
{
  matrix.incrementNonZerosOnOuter(
    segments_begin, segments_end,
    activeAxons_begin, activeAxons_end,
    permanenceDelta);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::adjustInactiveSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* activeAxons_begin, const UInt32* activeAxons_end,
  Real32 permanenceDelta)
{
  matrix.incrementNonZerosOnRowsExcludingCols(
    segments_begin, segments_end,
    activeAxons_begin, activeAxons_end,
    permanenceDelta);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::growSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* axons_begin, const UInt32* axons_end,
  Real32 initialPermanence)
{
  matrix.setZerosOnOuter(
    segments_begin, segments_end,
    axons_begin, axons_end,
    initialPermanence);
}

void SparseMatrixConnections::growSynapsesToSample(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* axons_begin, const UInt32* axons_end,
  UInt32 sampleSize, Real32 initialPermanence, nupic::Random& rng)
{
  matrix.setRandomZerosOnOuter(
    segments_begin, segments_end,
    axons_begin, axons_end,
    sampleSize, initialPermanence, rng);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::growSynapsesToSample(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* axons_begin, const UInt32* axons_end,
  const UInt32* sampleSizes_begin, const UInt32* sampleSizes_end,
  Real32 initialPermanence, nupic::Random& rng)
{
  NTA_ASSERT(std::distance(sampleSizes_begin, sampleSizes_end) ==
             std::distance(segments_begin, segments_end));

  matrix.setRandomZerosOnOuter(
    segments_begin, segments_end,
    axons_begin, axons_end,
    sampleSizes_begin, sampleSizes_end,
    initialPermanence, rng);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::clipPermanences(
  const UInt32* segments_begin, const UInt32* segments_end)
{
  matrix.clipRowsBelowAndAbove(segments_begin, segments_end, 0.0, 1.0);
}

void SparseMatrixConnections::mapSegmentsToSynapseCounts(
  const UInt32* segments_begin, const UInt32* segments_end,
  UInt32* out_begin) const
{
  matrix.nNonZerosPerRow(segments_begin, segments_end, out_begin);
}
