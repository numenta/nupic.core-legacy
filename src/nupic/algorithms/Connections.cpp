/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of Connections
 */

#include <climits>
#include <iostream>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/Connections.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;

Connections::Connections(CellIdx numCells,
                         SegmentIdx maxSegmentsPerCell,
                         SynapseIdx maxSynapsesPerSegment)
{
  initialize(numCells, maxSegmentsPerCell, maxSynapsesPerSegment);
}

void Connections::initialize(CellIdx numCells,
                             SegmentIdx maxSegmentsPerCell,
                             SynapseIdx maxSynapsesPerSegment)
{
  cells_ = vector<CellData>(numCells);
  numSegments_ = 0;
  numSynapses_ = 0;
  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;
  iteration_ = 0;
}

Segment Connections::createSegment(const Cell& cell)
{
  vector<SegmentData>& segments = cells_[cell.idx].segments;
  SegmentData segmentData = {vector<SynapseData>(), false, iteration_};
  Segment segment(segments.size(), cell);

  if ((SegmentIdx)segments.size() == maxSegmentsPerCell_)
  {
    bool found = leastRecentlyUsedSegment(cell, segment);
    if (!found) { NTA_THROW << "Unable to find segment to reuse."; }
    destroySegment(segment);
    segments[segment.idx] = segmentData;
  }
  else
  {
    segments.push_back(segmentData);
  }

  numSegments_++;
  return segment;
}

Synapse Connections::createSynapse(const Segment& segment,
                                   const Cell& presynapticCell,
                                   Permanence permanence)
{
  vector<SynapseData>& synapses = cells_[segment.cell.idx].segments[segment.idx].synapses;
  if (synapses.size() == maxSynapsesPerSegment_)
  {
    Synapse minSynapse;
    minPermanenceSynapse(segment, minSynapse);
    destroySynapse(minSynapse);
  }
  Synapse synapse(synapses.size(), segment);

  SynapseData synapseData = {presynapticCell, permanence, false};
  synapses.push_back(synapseData);
  numSynapses_++;

  synapsesForPresynapticCell_[presynapticCell].push_back(synapse);

  return synapse;
}

void Connections::destroySegment(const Segment& segment)
{
  const Cell& cell = segment.cell;
  SegmentData& segmentData = cells_[cell.idx].segments[segment.idx];

  for (auto synapse : synapsesForSegment(segment))
  {
    destroySynapse(synapse);
  }

  segmentData.destroyed = true;
  numSegments_--;
}

void Connections::destroySynapse(const Synapse& synapse)
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;
  SynapseData& synapseData = cells_[cell.idx].segments[segment.idx].synapses[synapse.idx];

  synapseData.destroyed = true;
  numSynapses_--;

  vector<Synapse>& synapses = synapsesForPresynapticCell_[synapseData.presynapticCell];

  for (auto s = synapses.begin(); s != synapses.end(); s++)
  {
    if (*s == synapse)
    {
      synapses.erase(s);
      break;
    }
  }

  if (synapses.size() == 0)
  {
    synapsesForPresynapticCell_.erase(synapseData.presynapticCell);
  }
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  cells_[cell.idx].segments[segment.idx].synapses[synapse.idx].permanence = permanence;
}

vector<Segment> Connections::segmentsForCell(const Cell& cell) const
{
  vector<Segment> segments;
  Segment segment;

  for (SegmentIdx i = 0; i < (SegmentIdx)cells_[cell.idx].segments.size(); i++)
  {
    segment.idx = i;
    segment.cell = cell;
    segments.push_back(segment);
  }

  return segments;
}

vector<Synapse> Connections::synapsesForSegment(const Segment& segment)
{
  const Cell& cell = segment.cell;
  SegmentData segmentData = cells_[cell.idx].segments[segment.idx];
  vector<Synapse> synapses;
  Synapse synapse;
  SynapseData synapseData;

  if (segmentData.destroyed)
  {
    throw runtime_error("Attempting to access destroyed segment's synapses.");
  }

  for (SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
  {
    synapse.idx = i;
    synapse.segment = segment;
    synapseData = dataForSynapse(synapse);

    if (!synapseData.destroyed)
    {
      synapses.push_back(synapse);
    }
  }

  return synapses;
}

SegmentData Connections::dataForSegment(const Segment& segment) const
{
  const Cell& cell = segment.cell;

  return cells_[cell.idx].segments[segment.idx];
}

SynapseData Connections::dataForSynapse(const Synapse& synapse) const
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  return cells_[cell.idx].segments[segment.idx].synapses[synapse.idx];
}

std::vector<Synapse> Connections::synapsesForPresynapticCell(const Cell& presynapticCell) const
{
  if (synapsesForPresynapticCell_.find(presynapticCell) == 
      synapsesForPresynapticCell_.end())
    return vector<Synapse>{};

  return synapsesForPresynapticCell_.at(presynapticCell.idx);
}

bool Connections::mostActiveSegmentForCells(const vector<Cell>& cells,
                                            vector<Cell> input,
                                            SynapseIdx synapseThreshold,
                                            Segment& retSegment) const
{
  SynapseIdx numSynapses, maxSynapses = synapseThreshold;
  vector<SegmentData> segments;
  vector<SynapseData> synapses;
  SegmentIdx segmentIdx = 0;
  bool found = false;

  sort(input.begin(), input.end());  // for binary search

  for (auto cell : cells)
  {
    segments = cells_[cell.idx].segments;
    segmentIdx = 0;

    for (auto segment : segments)
    {
      synapses = segment.synapses;
      numSynapses = 0;

      for (auto synapse : synapses)
      {
        if (!synapse.destroyed && synapse.permanence > 0 &&
            binary_search(input.begin(), input.end(), synapse.presynapticCell))
        {
          numSynapses++;
        }
      }

      if (numSynapses >= maxSynapses)
      {
        maxSynapses = numSynapses;
        retSegment.idx = segmentIdx;
        retSegment.cell = cell;
        found = true;
      }

      segmentIdx++;
    }
  }

  return found;
}

bool Connections::leastRecentlyUsedSegment(const Cell& cell,
                                           Segment& retSegment) const
{
  bool found = false;
  Iteration minIteration = ULLONG_MAX;
  SegmentData segmentData;

  for (auto segment : segmentsForCell(cell))
  {
    // TODO: Possible optimization - define constant variable here?
    segmentData = dataForSegment(segment);

    if (segmentData.lastUsedIteration < minIteration && !segmentData.destroyed)
    {
      retSegment = segment;
      found = true;
      minIteration = segmentData.lastUsedIteration;
    }
  }

  return found;
}

bool Connections::minPermanenceSynapse(const Segment& segment,
                                       Synapse& retSynapse) const
{
  const vector<SynapseData>& synapses =
    cells_[segment.cell.idx].segments[segment.idx].synapses;

  if (synapses.size() == 0)
  {
    return false;
  }

  vector<SynapseData>::const_iterator minSynapseIterator = min_element(
    synapses.begin(),
    synapses.end(),
    [](const SynapseData &a, const SynapseData &b) -> bool
    {
      return a.permanence < b.permanence;
    });

  SynapseIdx minSynapseIdx = distance(synapses.begin(), minSynapseIterator);
  retSynapse = Synapse(minSynapseIdx, segment);

  return true;
}

Activity Connections::computeActivity(const vector<Cell>& input,
                                      Permanence permanenceThreshold,
                                      SynapseIdx synapseThreshold,
                                      bool recordIteration)
{
  Activity activity;
  vector<Synapse> synapses;
  SynapseData synapseData;

  for (auto cell : input)
  {
    if (!synapsesForPresynapticCell_.count(cell)) continue;
    synapses = synapsesForPresynapticCell_.at(cell);

    for (auto synapse : synapses)
    {
      // TODO: Possible optimization - define constant variable here?
      synapseData = dataForSynapse(synapse);

      // Ignore any synapses with permanence 0
      if (synapseData.permanence >= permanenceThreshold &&
          synapseData.permanence > 0)
      {
        activity.numActiveSynapsesForSegment[synapse.segment] += 1;

        if (activity.numActiveSynapsesForSegment[synapse.segment] == synapseThreshold)
        {
          activity.activeSegmentsForCell[synapse.segment.cell].push_back(synapse.segment);

          if (recordIteration)
          {
            cells_[synapse.segment.cell.idx].segments[synapse.segment.idx].lastUsedIteration++;
          }
        }
      }
    }
  }

  if (recordIteration)
  {
    iteration_++;
  }

  return activity;
}

vector<Segment> Connections::activeSegments(const Activity& activity)
{
  vector<Segment> segments;

  for (auto i : activity.activeSegmentsForCell)
  {
    segments.insert(segments.end(), i.second.begin(), i.second.end());
  }

  return segments;
}

vector<Cell> Connections::activeCells(const Activity& activity)
{
  vector<Cell> cells;

  for (auto i : activity.activeSegmentsForCell)
  {
    cells.push_back(i.first);
  }

  return cells;
}

void Connections::save(ostream& outStream) const
{
  // Write a starting marker.
  outStream << "Connections" << endl;
  outStream << Connections::VERSION << endl;

  outStream << cells_.size() << " "
            << maxSegmentsPerCell_ << " "
            << maxSynapsesPerSegment_ << " "
            << endl;

  for (CellData cellData : cells_) {
    auto segments = cellData.segments;
    outStream << segments.size() << " ";
    
    for (SegmentData segment : segments) {
      outStream << segment.destroyed << " ";
      outStream << segment.lastUsedIteration << " ";

      auto synapses = segment.synapses;
      outStream << synapses.size() << " ";

      for (SynapseData synapse : synapses) {
        outStream << synapse.presynapticCell.idx << " ";
        outStream << synapse.permanence << " ";
        outStream << synapse.destroyed << " ";
      }
      outStream << endl;
    }
    outStream << endl;
  }
  outStream << endl;

  outStream << iteration_ << " " << endl;

  outStream << "~Connections" << endl;
}

void Connections::write(ConnectionsProto::Builder& proto) const
{
  proto.setVersion(Connections::VERSION);

  auto protoCells = proto.initCells(cells_.size());

  for (CellIdx i = 0; i < cells_.size(); ++i) {
    auto segments = cells_[i].segments;
    auto protoSegments = protoCells[i].initSegments(segments.size());

    for (SegmentIdx j = 0; j < (SegmentIdx)segments.size(); ++j) {
      auto synapses = segments[j].synapses;
      auto protoSynapses = protoSegments[j].initSynapses(synapses.size());
      protoSegments[j].setDestroyed(segments[j].destroyed);
      protoSegments[j].setLastUsedIteration(segments[j].lastUsedIteration);

      for (SynapseIdx k = 0; k < synapses.size(); ++k) {
        protoSynapses[k].setPresynapticCell(synapses[k].presynapticCell.idx);
        protoSynapses[k].setPermanence(synapses[k].permanence);
        protoSynapses[k].setDestroyed(synapses[k].destroyed);
      }
    }
  }

  proto.setMaxSegmentsPerCell(maxSegmentsPerCell_);
  proto.setMaxSynapsesPerSegment(maxSynapsesPerSegment_);
  proto.setIteration(iteration_);
}

void Connections::load(istream& inStream)
{
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "Connections");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version <= Connections::VERSION);

  // Retrieve simple variables
  UInt numCells;
  inStream >> numCells
           >> maxSegmentsPerCell_
           >> maxSynapsesPerSegment_;

  initialize(numCells, maxSegmentsPerCell_, maxSynapsesPerSegment_);

  cells_.resize(numCells);
  for (UInt i = 0; i < numCells; i++) {
    CellData& cellData = cells_[i];

    UInt numSegments;
    inStream >> numSegments;

    cellData.segments.resize(numSegments);
    for (UInt j = 0; j < numSegments; j++) {
      inStream >> cellData.segments[j].destroyed;
      inStream >> cellData.segments[j].lastUsedIteration;

      UInt numSynapses;
      inStream >> numSynapses;

      auto& synapses = cellData.segments[j].synapses;
      synapses.resize(numSynapses);
      for (UInt k = 0; k < numSynapses; k++) {
        inStream >> synapses[k].presynapticCell.idx;
        inStream >> synapses[k].permanence;
        inStream >> synapses[k].destroyed;

        if (!synapses[k].destroyed) {
          numSynapses_++;

          Synapse synapse = Synapse(k, Segment(j, Cell(i)));
          synapsesForPresynapticCell_[synapses[k].presynapticCell].push_back(synapse);
        }
      }

      if (!cellData.segments[j].destroyed) {
        numSegments_++;
      }
    }
  }

  inStream >> iteration_;

  inStream >> marker;
  NTA_CHECK(marker == "~Connections");
}

void Connections::read(ConnectionsProto::Reader& proto)
{
  // Check the saved version.
  UInt version = proto.getVersion();
  NTA_CHECK(version <= Connections::VERSION);

  auto protoCells = proto.getCells();

  initialize(protoCells.size(),
             proto.getMaxSegmentsPerCell(),
             proto.getMaxSynapsesPerSegment());

  for (CellIdx i = 0; i < protoCells.size(); ++i) {
    auto protoSegments = protoCells[i].getSegments();
    vector<SegmentData>& segments = cells_[i].segments;

    for (SegmentIdx j = 0; j < (SegmentIdx)protoSegments.size(); ++j) {
      SegmentData segmentData = {vector<SynapseData>(),
                                 protoSegments[j].getDestroyed(),
                                 protoSegments[j].getLastUsedIteration()};
      segments.push_back(segmentData);

      auto protoSynapses = protoSegments[j].getSynapses();
      vector<SynapseData>& synapses = segments[j].synapses;

      for (SynapseIdx k = 0; k < protoSynapses.size(); ++k) {
        Cell presynapticCell = Cell(protoSynapses[k].getPresynapticCell());
        SynapseData synapseData = {presynapticCell,
                                   protoSynapses[k].getPermanence(),
                                   protoSynapses[k].getDestroyed()};
        synapses.push_back(synapseData);

        if (!synapseData.destroyed) {
          numSynapses_++;

          Synapse synapse = Synapse(k, Segment(j, Cell(i)));
          synapsesForPresynapticCell_[presynapticCell].push_back(synapse);
        }
      }

      if (!segmentData.destroyed) {
        numSegments_++;
      }
    }
  }

  iteration_ = proto.getIteration();
}

UInt Connections::numSegments() const
{
  return numSegments_;
}

UInt Connections::numSynapses() const
{
  return numSynapses_;
}

bool Connections::operator==(const Connections &other) const
{
  if (maxSegmentsPerCell_ != other.maxSegmentsPerCell_) return false;
  if (maxSynapsesPerSegment_ != other.maxSynapsesPerSegment_) return false;

  if (cells_.size() != other.cells_.size()) return false;

  for (CellIdx i = 0; i < cells_.size(); ++i) {
    auto segments = cells_[i].segments;
    auto otherSegments = other.cells_[i].segments;

    if (segments.size() != otherSegments.size()) return false;

    for (SegmentIdx j = 0; j < (SegmentIdx)segments.size(); ++j) {
      auto segment = segments[j];
      auto otherSegment = otherSegments[j];
      auto synapses = segment.synapses;
      auto otherSynapses = otherSegment.synapses;

      if (segment.destroyed != otherSegment.destroyed) return false;
      if (segment.lastUsedIteration != otherSegment.lastUsedIteration) return false;
      if (synapses.size() != otherSynapses.size()) return false;

      for (SynapseIdx k = 0; k < synapses.size(); ++k) {
        auto synapse = synapses[k];
        auto otherSynapse = synapses[k];

        if (synapse.presynapticCell.idx != otherSynapse.presynapticCell.idx) return false;
        if (synapse.permanence != otherSynapse.permanence) return false;
        if (synapse.destroyed != otherSynapse.destroyed) return false;
      }
    }
  }

  if (synapsesForPresynapticCell_.size() != other.synapsesForPresynapticCell_.size()) return false;

  for (auto i = synapsesForPresynapticCell_.begin(); i != synapsesForPresynapticCell_.end(); ++i) {
    auto synapses = i->second;
    auto otherSynapses = other.synapsesForPresynapticCell_.at(i->first);

    if (synapses.size() != otherSynapses.size()) return false;

    for (SynapseIdx j = 0; j < synapses.size(); ++j) {
      auto synapse = synapses[j];
      auto otherSynapse = otherSynapses[j];
      auto segment = synapse.segment;
      auto otherSegment = otherSynapse.segment;
      auto cell = segment.cell;
      auto otherCell = otherSegment.cell;

      if (synapse.idx != otherSynapse.idx) return false;
      if (segment.idx != otherSegment.idx) return false;
      if (cell.idx != otherCell.idx) return false;
    }
  }

  if (numSegments_ != other.numSegments_) return false;
  if (numSynapses_ != other.numSynapses_) return false;
  if (iteration_ != other.iteration_) return false;

  return true;
}

bool Cell::operator==(const Cell &other) const
{
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

bool Synapse::operator==(const Synapse &other) const
{
  return idx == other.idx && segment == other.segment;
}
