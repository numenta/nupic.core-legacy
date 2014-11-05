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
  Segment segment = {segments.size(), cell};

  SegmentData segmentData;
  segments.push_back(segmentData);

  return segment;
}

Synapse Connections::createSynapse(const Segment& segment,
                                   const Cell& presynapticCell,
                                   Permanence permanence)
{
  const Cell& cell = segment.cell;

  vector<SynapseData>& synapses = cells_[cell.idx].segments[segment.idx].synapses;
  Synapse synapse = {synapses.size(), segment};

  SynapseData synapseData = {presynapticCell, permanence};
  synapses.push_back(synapseData);

  return synapse;
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  cells_[cell.idx].segments[segment.idx].synapses[synapse.idx].permanence = permanence;
}

vector<Segment> Connections::getSegmentsForCell(const Cell& cell)
{
  vector<Segment> segments;
  Segment segment;

  for(SegmentIdx i = 0; i < cells_[cell.idx].segments.size(); i++) {
    segment.idx = i;
    segment.cell = cell;
    segments.push_back(segment);
  }

  return segments;
}

vector<Synapse> Connections::getSynapsesForSegment(const Segment& segment)
{
  const Cell& cell = segment.cell;
  vector<Synapse> synapses;
  Synapse synapse;

  for(SynapseIdx i = 0; i < cells_[cell.idx].segments[segment.idx].synapses.size(); i++) {
    synapse.idx = i;
    synapse.segment = segment;
    synapses.push_back(synapse);
  }

  return synapses;
}

SynapseData Connections::getDataForSynapse(const Synapse& synapse) const
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  return cells_[cell.idx].segments[segment.idx].synapses[synapse.idx];
}

bool Connections::getMostActiveSegmentForCells(const std::vector<Cell>& cells,
                                               const std::vector<Cell>& input,
                                               UInt synapseThreshold,
                                               Segment& segment) const
{
  // UInt numSynapses, maxSynapses = synapseThreshold;
  // vector<SegmentData> segments;
  // vector<SynapseData> synapses;
  bool found = false;

  // for (vector<Cell>::const_iterator c = cells.begin(); c != cells.end(); c++) {
  //   segments = cells_[c->idx].segments;

  //   for (vector<SegmentData>::const_iterator s = segments.begin(); s != segments.end(); s++) {
  //     synapses = s->synapses;
  //     numSynapses = 0;

  //     for (vector<SynapseData>::const_iterator syn = synapses.begin(); syn != synapses.end(); syn++) {
  //       if (find(input.begin(), input.end(), syn->presynapticCell) != input.end()) {  // TODO: Optimize this
  //         numSynapses++;
  //       }
  //     }

  //     if (numSynapses >= maxSynapses) {
  //       maxSynapses = numSynapses;
  //       segment.idx = s - segments.begin();
  //       segment.cell = *c;
  //       found = true;
  //     }
  //   }
  // }

  return found;
}

Activity Connections::computeActivity(const std::vector<Cell>& input,
                                      Permanence permanenceThreshold,
                                      UInt synapseThreshold) const
{
  Activity activity;
  // vector<Synapse> synapses;
  // SynapseData synapseData;
  // Synapse synapse;

  // for (vector<Cell>::const_iterator c = input.begin(); c != input.end(); c++) {
  //   synapses = synapsesForPresynapticCell_[*c];

  //   for (vector<Synapse>::const_iterator s = synapses.begin(); s != synapses.end(); s++) {
  //     synapseData = getDataForSynapse(*s);

  //     if (synapseData.permanence >= permanenceThreshold) {
  //       activity.numActiveSynapsesForSegment[synapse.segment] += 1;
  //     }

  //     if (activity.numActiveSynapsesForSegment[synapse.segment]) {
  //       activity.numActiveSegmentsForCell[*c] += 1;
  //     }
  //   }
  // }

  return activity;
}
