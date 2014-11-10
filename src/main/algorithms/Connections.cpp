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

Connections::Connections(CellIdx numCells) : cells_(numCells) {}

Segment Connections::createSegment(const Cell& cell)
{
  vector<SegmentData>& segments = cells_[cell.idx].segments;
  Segment segment(segments.size(), cell);

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
  Synapse synapse(synapses.size(), segment);

  SynapseData synapseData = {presynapticCell, permanence};
  synapses.push_back(synapseData);

  synapsesForPresynapticCell_[presynapticCell].push_back(synapse);

  return synapse;
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  cells_[cell.idx].segments[segment.idx].synapses[synapse.idx].permanence = permanence;
}

vector<Segment> Connections::segmentsForCell(const Cell& cell)
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

vector<Synapse> Connections::synapsesForSegment(const Segment& segment)
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

SynapseData Connections::dataForSynapse(const Synapse& synapse) const
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  return cells_[cell.idx].segments[segment.idx].synapses[synapse.idx];
}

bool Connections::mostActiveSegmentForCells(const vector<Cell>& cells,
                                            vector<Cell> input,
                                            UInt synapseThreshold,
                                            Segment& retSegment) const
{
  UInt numSynapses, maxSynapses = synapseThreshold;
  vector<SegmentData> segments;
  vector<SynapseData> synapses;
  bool found = false;

  sort(input.begin(), input.end());  // for binary search

  for (vector<Cell>::const_iterator cell = cells.begin(); cell != cells.end(); cell++) {
    segments = cells_[cell->idx].segments;

    for (vector<SegmentData>::const_iterator segment = segments.begin(); segment != segments.end(); segment++) {
      synapses = segment->synapses;
      numSynapses = 0;

      for (vector<SynapseData>::const_iterator synapse = synapses.begin(); synapse != synapses.end(); synapse++) {
        if (binary_search(input.begin(), input.end(), synapse->presynapticCell)) {
          numSynapses++;
        }
      }

      if (numSynapses >= maxSynapses) {
        maxSynapses = numSynapses;
        retSegment.idx = segment - segments.begin();
        retSegment.cell = *cell;
        found = true;
      }
    }
  }

  return found;
}

Activity Connections::computeActivity(const vector<Cell>& input,
                                      Permanence permanenceThreshold,
                                      UInt synapseThreshold) const
{
  Activity activity;
  vector<Synapse> synapses;
  SynapseData synapseData;

  for (vector<Cell>::const_iterator cell = input.begin(); cell != input.end(); cell++) {
    if (!synapsesForPresynapticCell_.count(*cell)) continue;
    synapses = synapsesForPresynapticCell_.at(*cell);

    for (vector<Synapse>::const_iterator synapse = synapses.begin(); synapse != synapses.end(); synapse++) {
      synapseData = dataForSynapse(*synapse);

      if (synapseData.permanence >= permanenceThreshold) {
        activity.numActiveSynapsesForSegment[synapse->segment] += 1;

        if (activity.numActiveSynapsesForSegment[synapse->segment] == synapseThreshold) {
          activity.activeSegmentsForCell[synapse->segment.cell].push_back(synapse->segment);
        }
      }
    }
  }

  return activity;
}

vector<Segment> Connections::activeSegments(const Activity& activity)
{
  vector<Segment> segments;

  for (map< Cell, std::vector<Segment> >::const_iterator i = activity.activeSegmentsForCell.begin();
       i != activity.activeSegmentsForCell.end();
       i++) {
    segments.insert(segments.end(), i->second.begin(), i->second.end());
  }

  return segments;
}

vector<Cell> Connections::activeCells(const Activity& activity)
{
  vector<Cell> cells;

  for (map< Cell, std::vector<Segment> >::const_iterator i = activity.activeSegmentsForCell.begin();
       i != activity.activeSegmentsForCell.end();
       i++) {
    cells.push_back(i->first);
  }

  return cells;
}

bool Cell::operator==(const Cell &other) const {
  return idx == other.idx;
}

bool Cell::operator<=(const Cell &other) const
{
  return idx <= other.idx;
}

bool Cell::operator<(const Cell &other) const
{
  return idx < other.idx;
}

bool Cell::operator>=(const Cell &other) const
{
  return idx >= other.idx;
}

bool Cell::operator>(const Cell &other) const
{
  return idx > other.idx;
}

bool Segment::operator==(const Segment &other) const
{
  return idx == other.idx && cell == other.cell;
}

bool Segment::operator<=(const Segment &other) const
{
  return idx == other.idx ? cell <= other.cell : idx <= other.idx;
}

bool Segment::operator<(const Segment &other) const
{
  return idx == other.idx ? cell < other.cell : idx < other.idx;
}

bool Segment::operator>=(const Segment &other) const
{
  return idx == other.idx ? cell >= other.cell : idx >= other.idx;
}

bool Segment::operator>(const Segment &other) const
{
  return idx == other.idx ? cell > other.cell : idx > other.idx;
}

bool Synapse::operator==(const Synapse &other) const {
  return idx == other.idx && segment == other.segment;
}
