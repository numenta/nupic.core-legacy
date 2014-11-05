/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of Connections
 */

#include <iostream>
#include <nta/algorithms/Connections.hpp>

using namespace std;
using namespace nta;
using namespace nta::algorithms::connections;

Connections::Connections(CellIdx numCells) : cells_(numCells)
{
}

Segment Connections::createSegment(const Cell& cell)
{
  vector<SegmentData>& segments = cells_[cell.idx].segments;
  Segment segment = {segments.size(), cell.idx};

  SegmentData segmentData;
  segments.push_back(segmentData);

  return segment;
}

Synapse Connections::createSynapse(const Segment& segment,
                                   const Cell& presynapticCell,
                                   Permanence permanence)
{
  vector<SynapseData>& synapses = cells_[segment.cellIdx].segments[segment.idx].synapses;
  Synapse synapse = {synapses.size(), segment.idx, segment.cellIdx};

  SynapseData synapseData;
  synapses.push_back(synapseData);

  return synapse;
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
}

vector<Segment> Connections::getSegmentsForCell(const Cell& cell)
{
  vector<Segment> segments;
  Segment segment;

  for(SegmentIdx i = 0; i < cells_[cell.idx].segments.size(); i++) {
    segment.idx = i;
    segment.cellIdx = cell.idx;
    segments.push_back(segment);
  }

  return segments;
}

vector<Synapse> Connections::getSynapsesForSegment(const Segment& segment)
{
  vector<Synapse> synapses;
  Synapse synapse;

  for(SynapseIdx i = 0; i < cells_[segment.cellIdx].segments[segment.idx].synapses.size(); i++) {
    synapse.idx = i;
    synapse.segmentIdx = segment.idx;
    synapse.cellIdx = segment.cellIdx;
    synapses.push_back(synapse);
  }

  return synapses;
}

bool Connections::getMostActiveSegmentForCells(const std::vector<Cell>& cells,
                                               const std::vector<Cell>& input,
                                               UInt synapseThreshold,
                                               Segment& segment) const
{
  return false;
}

Activity Connections::computeActivity(const std::vector<Cell>& input,
                                      Permanence permanenceThreshold,
                                      UInt synapseThreshold) const
{
  Activity activity;
  return activity;
}
