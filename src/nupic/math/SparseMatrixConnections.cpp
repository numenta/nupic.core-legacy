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
  UInt32 numCells, UInt32 numInputs)
  : SegmentMatrixAdapter<SparseMatrix<UInt32, Real32, Int32, Real64>>(numCells,
                                                                      numInputs)
{}


void SparseMatrixConnections::computeActivity(
  const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
  Int32* overlaps_begin) const
{
  matrix.rightVecSumAtNZSparse(
    activeInputs_begin, activeInputs_end,
    overlaps_begin);
}

void SparseMatrixConnections::computeActivity(
  const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
  Real32 permanenceThreshold, Int32* overlaps_begin) const
{
  matrix.rightVecSumAtNZGteThresholdSparse(
    activeInputs_begin, activeInputs_end,
    overlaps_begin, permanenceThreshold);
}

void SparseMatrixConnections::adjustSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
  Real32 activePermanenceDelta, Real32 inactivePermanenceDelta)
{
  matrix.incrementNonZerosOnOuter(
    segments_begin, segments_end,
    activeInputs_begin, activeInputs_end,
    activePermanenceDelta);

  matrix.incrementNonZerosOnRowsExcludingCols(
    segments_begin, segments_end,
    activeInputs_begin, activeInputs_end,
    inactivePermanenceDelta);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::adjustActiveSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
  Real32 permanenceDelta)
{
  matrix.incrementNonZerosOnOuter(
    segments_begin, segments_end,
    activeInputs_begin, activeInputs_end,
    permanenceDelta);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::adjustInactiveSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
  Real32 permanenceDelta)
{
  matrix.incrementNonZerosOnRowsExcludingCols(
    segments_begin, segments_end,
    activeInputs_begin, activeInputs_end,
    permanenceDelta);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::growSynapses(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* inputs_begin, const UInt32* inputs_end,
  Real32 initialPermanence)
{
  matrix.setZerosOnOuter(
    segments_begin, segments_end,
    inputs_begin, inputs_end,
    initialPermanence);
}

void SparseMatrixConnections::growSynapsesToSample(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* inputs_begin, const UInt32* inputs_end,
  Int32 sampleSize, Real32 initialPermanence, nupic::Random& rng)
{
  matrix.setRandomZerosOnOuter(
    segments_begin, segments_end,
    inputs_begin, inputs_end,
    sampleSize, initialPermanence, rng);

  clipPermanences(segments_begin, segments_end);
}

void SparseMatrixConnections::growSynapsesToSample(
  const UInt32* segments_begin, const UInt32* segments_end,
  const UInt32* inputs_begin, const UInt32* inputs_end,
  const Int32* sampleSizes_begin, const Int32* sampleSizes_end,
  Real32 initialPermanence, nupic::Random& rng)
{
  NTA_ASSERT(std::distance(sampleSizes_begin, sampleSizes_end) ==
             std::distance(segments_begin, segments_end));

  matrix.setRandomZerosOnOuter(
    segments_begin, segments_end,
    inputs_begin, inputs_end,
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
  Int32* out_begin) const
{
  matrix.nNonZerosPerRow(segments_begin, segments_end, out_begin);
}
