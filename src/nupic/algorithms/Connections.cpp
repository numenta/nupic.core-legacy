/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014-2016, Numenta, Inc.  Unless you have an agreement
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

static const Permanence EPSILON = 0.00001;

Connections::Connections(UInt32 numCells,
                         SegmentIdx maxSegmentsPerCell,
                         SynapseIdx maxSynapsesPerSegment)
{
  initialize(numCells, maxSegmentsPerCell, maxSynapsesPerSegment);
}

void Connections::initialize(UInt32 numCells,
                             SegmentIdx maxSegmentsPerCell,
                             SynapseIdx maxSynapsesPerSegment)
{
  cells_ = vector<CellData>(numCells);
  numSegments_ = 0;
  numSynapses_ = 0;
  nextFlatIdx_ = 0;
  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;
  iteration_ = 0;
  nextEventToken_ = 0;
}

UInt32 Connections::subscribe(ConnectionsEventHandler* handler)
{
  UInt32 token = nextEventToken_++;
  eventHandlers_[token] = handler;
  return token;
}

void Connections::unsubscribe(UInt32 token)
{
  delete eventHandlers_.at(token);
  eventHandlers_.erase(token);
}

Segment Connections::createSegment(UInt32 cell)
{
  NTA_CHECK(maxSegmentsPerCell_ > 0);
  while (numSegments(cell) >= maxSegmentsPerCell_)
  {
    destroySegment(leastRecentlyUsedSegment_(cell));
  }

  CellData& cellData = cells_[cell];
  Segment segment = {(SegmentIdx)-1, cell};
  if (cellData.numDestroyedSegments > 0)
  {
    bool found = false;
    for (SegmentIdx i = 0; i < cellData.segments.size(); i++)
    {
      if (cellData.segments[i].destroyed)
      {
        segment.idx = i;
        found = true;
      }
    }

    NTA_CHECK(found);

    cellData.segments[segment.idx].destroyed = false;
    cellData.numDestroyedSegments--;
  }
  else
  {
    segment.idx = cellData.segments.size();
    cellData.segments.push_back(SegmentData());
    cellData.segments[segment.idx].flatIdx = nextFlatIdx_++;
    segmentForFlatIdx_.push_back(segment);
  }

  cellData.segments[segment.idx].lastUsedIteration = iteration_;
  numSegments_++;

  for (auto h : eventHandlers_)
  {
    h.second->onCreateSegment(segment);
  }

  return segment;
}

Synapse Connections::createSynapse(const Segment& segment,
                                   UInt32 presynapticCell,
                                   Permanence permanence)
{
  NTA_CHECK(maxSynapsesPerSegment_ > 0);
  NTA_CHECK(permanence > 0);
  while (numSynapses(segment) >= maxSynapsesPerSegment_)
  {
    destroySynapse(minPermanenceSynapse_(segment));
  }

  SegmentData& segmentData = dataForSegment_(segment);
  SynapseIdx synapseIdx = (SynapseIdx)-1;
  if (segmentData.numDestroyedSynapses > 0)
  {
    bool found = false;
    for(SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
    {
      if (segmentData.synapses[i].destroyed)
      {
        synapseIdx = i;
        found = true;
        break;
      }
    }

    NTA_CHECK(found);

    segmentData.synapses[synapseIdx].destroyed = false;
    segmentData.numDestroyedSynapses--;
  }
  else
  {
    synapseIdx = segmentData.synapses.size();
    segmentData.synapses.push_back(SynapseData());
  }

  SynapseData& synapseData = segmentData.synapses[synapseIdx];
  synapseData.presynapticCell = presynapticCell;
  synapseData.permanence = permanence;

  Synapse synapse = {synapseIdx, segment};
  synapsesForPresynapticCell_[presynapticCell].push_back(synapse);
  numSynapses_++;

  for (auto h : eventHandlers_)
  {
    h.second->onCreateSynapse(synapse);
  }

  return synapse;
}

void Connections::destroySegment(const Segment& segment)
{
  for (auto h : eventHandlers_)
  {
    h.second->onDestroySegment(segment);
  }

  SegmentData& segmentData = dataForSegment_(segment);

  NTA_CHECK(!segmentData.destroyed) << "Segment already destroyed.";

  for (SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
  {
    Synapse synapse = {i, segment};
    const SynapseData& synapseData = dataForSynapse_(synapse);

    if (!synapseData.destroyed)
    {
      vector<Synapse>& presynapticSynapses =
        synapsesForPresynapticCell_.at(synapseData.presynapticCell);

      auto it = std::find(presynapticSynapses.begin(), presynapticSynapses.end(),
                          synapse);
      NTA_CHECK(it != presynapticSynapses.end());
      presynapticSynapses.erase(it);

      if (presynapticSynapses.size() == 0)
      {
        synapsesForPresynapticCell_.erase(synapseData.presynapticCell);
      }
      numSynapses_--;
    }
  }
  segmentData.synapses.clear();
  segmentData.numDestroyedSynapses = 0;

  segmentData.destroyed = true;
  cells_[segment.cell].numDestroyedSegments++;
  numSegments_--;
}

void Connections::destroySynapse(const Synapse& synapse)
{
  for (auto h : eventHandlers_)
  {
    h.second->onDestroySynapse(synapse);
  }

  SynapseData& synapseData = dataForSynapse_(synapse);

  NTA_CHECK(!synapseData.destroyed) << "Synapse already destroyed.";

  vector<Synapse>& presynapticSynapses =
    synapsesForPresynapticCell_.at(synapseData.presynapticCell);

  auto it = std::find(presynapticSynapses.begin(), presynapticSynapses.end(),
                      synapse);
  NTA_CHECK(it != presynapticSynapses.end());
  presynapticSynapses.erase(it);

  if (presynapticSynapses.size() == 0)
  {
    synapsesForPresynapticCell_.erase(synapseData.presynapticCell);
  }

  synapseData.destroyed = true;
  dataForSegment_(synapse.segment).numDestroyedSynapses++;
  numSynapses_--;
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
  for (auto h : eventHandlers_)
  {
    h.second->onUpdateSynapsePermanence(synapse, permanence);
  }

  dataForSynapse_(synapse).permanence = permanence;
}

vector<Segment> Connections::segmentsForCell(UInt32 cell) const
{
  vector<Segment> segments;
  segments.reserve(numSegments(cell));

  const vector<SegmentData>& allSegments = cells_[cell].segments;

  for (SegmentIdx i = 0; i < (SegmentIdx)allSegments.size(); i++)
  {
    if (!allSegments[i].destroyed)
    {
      segments.push_back({i, cell});
    }
  }

  return segments;
}

vector<Synapse> Connections::synapsesForSegment(const Segment& segment)
{
  vector<Synapse> synapses;
  synapses.reserve(numSynapses(segment));

  const SegmentData& segmentData = dataForSegment_(segment);
  if (segmentData.destroyed)
  {
    NTA_THROW << "Attempting to access destroyed segment's synapses.";
  }

  for (SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
  {
    if (!segmentData.synapses[i].destroyed)
    {
      synapses.push_back({i, segment});
    }
  }

  return synapses;
}

SegmentData Connections::dataForSegment(const Segment& segment) const
{
  return cells_[segment.cell].segments[segment.idx];
}

SegmentData& Connections::dataForSegment_(const Segment& segment)
{
  return cells_[segment.cell].segments[segment.idx];
}

const SegmentData& Connections::dataForSegment_(const Segment& segment) const
{
  return cells_[segment.cell].segments[segment.idx];
}

SynapseData Connections::dataForSynapse(const Synapse& synapse) const
{
  return cells_[synapse.segment.cell]
    .segments[synapse.segment.idx]
    .synapses[synapse.idx];
}

SynapseData& Connections::dataForSynapse_(const Synapse& synapse)
{
  SegmentData& segmentData = dataForSegment_(synapse.segment);
  return segmentData.synapses[synapse.idx];
}

const SynapseData& Connections::dataForSynapse_(const Synapse& synapse) const
{
  const SegmentData& segmentData = dataForSegment_(synapse.segment);
  return segmentData.synapses[synapse.idx];
}

Segment Connections::segmentForFlatIdx(UInt32 flatIdx) const
{
  return segmentForFlatIdx_[flatIdx];
}

std::vector<Synapse> Connections::synapsesForPresynapticCell(UInt32 presynapticCell) const
{
  if (synapsesForPresynapticCell_.find(presynapticCell) ==
      synapsesForPresynapticCell_.end())
    return vector<Synapse>{};

  return synapsesForPresynapticCell_.at(presynapticCell);
}

Segment Connections::leastRecentlyUsedSegment_(UInt32 cell) const
{
  const vector<SegmentData>& segments = cells_[cell].segments;
  bool found = false;
  SegmentIdx minIdx;
  Iteration minIteration;
  for (SegmentIdx i = 0; i < segments.size(); i++)
  {
    if (!segments[i].destroyed && (!found ||
                                   segments[i].lastUsedIteration < minIteration))
    {
      minIdx = i;
      minIteration = segments[i].lastUsedIteration;
      found = true;
    }
  }

  NTA_CHECK(found);

  return Segment(minIdx, cell);
}

Synapse Connections::minPermanenceSynapse_(const Segment& segment) const
{
  const vector<SynapseData>& synapses =
    cells_[segment.cell].segments[segment.idx].synapses;
  bool found = false;
  SynapseIdx minIdx;
  Permanence minPermanence;
  for (SynapseIdx i = 0; i < synapses.size(); i++)
  {
    if(!synapses[i].destroyed &&
       (!found || synapses[i].permanence < minPermanence - EPSILON))
    {
      minIdx = i;
      minPermanence = synapses[i].permanence;
      found = true;
    }
  }

  NTA_CHECK(found);

  return Synapse(minIdx, segment);
}

bool segmentIncreasingOrder(const SegmentOverlap& a, const SegmentOverlap& b)
{
  return a.segment < b.segment;
}

void Connections::computeActivity(const vector<CellIdx>& input,
                                  Permanence activePermanenceThreshold,
                                  SynapseIdx activeSynapseThreshold,
                                  Permanence matchingPermanenceThreshold,
                                  SynapseIdx matchingSynapseThreshold,
                                  vector<SegmentOverlap>& activeSegmentsOut,
                                  vector<SegmentOverlap>& matchingSegmentsOut)
  const
{
  SegmentExcitationTally excitations(*this, activePermanenceThreshold,
                                     matchingPermanenceThreshold);

  for (CellIdx cell : input)
  {
    excitations.addActivePresynapticCell(cell);
  }

  excitations.getResults(activeSynapseThreshold, matchingSynapseThreshold,
                         activeSegmentsOut, matchingSegmentsOut);
}

void Connections::recordSegmentActivity(Segment segment)
{
  dataForSegment_(segment).lastUsedIteration = iteration_;
}

void Connections::startNewIteration()
{
  iteration_++;
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
        outStream << synapse.presynapticCell << " ";
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

  for (UInt32 i = 0; i < cells_.size(); ++i) {
    auto segments = cells_[i].segments;
    auto protoSegments = protoCells[i].initSegments(segments.size());

    for (SegmentIdx j = 0; j < (SegmentIdx)segments.size(); ++j) {
      auto synapses = segments[j].synapses;
      auto protoSynapses = protoSegments[j].initSynapses(synapses.size());
      protoSegments[j].setDestroyed(segments[j].destroyed);
      protoSegments[j].setLastUsedIteration(segments[j].lastUsedIteration);

      for (SynapseIdx k = 0; k < synapses.size(); ++k) {
        protoSynapses[k].setPresynapticCell(synapses[k].presynapticCell);
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
  for (UInt cell = 0; cell < numCells; cell++)
  {
    CellData& cellData = cells_[cell];

    UInt numSegments;
    inStream >> numSegments;

    cellData.segments.resize(numSegments);
    for (UInt j = 0; j < numSegments; j++)
    {
      inStream >> cellData.segments[j].destroyed;
      inStream >> cellData.segments[j].lastUsedIteration;
      cellData.segments[j].flatIdx = nextFlatIdx_++;
      segmentForFlatIdx_.push_back(Segment(j, cell));

      UInt numSynapses;
      inStream >> numSynapses;

      auto& synapses = cellData.segments[j].synapses;
      synapses.resize(numSynapses);
      for (UInt k = 0; k < numSynapses; k++) {
        inStream >> synapses[k].presynapticCell;
        inStream >> synapses[k].permanence;
        inStream >> synapses[k].destroyed;

        if (synapses[k].destroyed)
        {
          cellData.segments[j].numDestroyedSynapses++;
        }
        else
        {
          numSynapses_++;

          Synapse synapse = Synapse(k, Segment(j, cell));
          synapsesForPresynapticCell_[synapses[k].presynapticCell].push_back(synapse);
        }
      }

      if (cellData.segments[j].destroyed)
      {
        cellData.numDestroyedSegments++;
      }
      else
      {
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

  for (UInt32 cell = 0; cell < protoCells.size(); ++cell)
  {
    auto protoSegments = protoCells[cell].getSegments();
    vector<SegmentData>& segments = cells_[cell].segments;

    for (SegmentIdx j = 0; j < (SegmentIdx)protoSegments.size(); ++j)
    {
      SegmentData segmentData = {vector<SynapseData>(),
                                 0,
                                 protoSegments[j].getDestroyed(),
                                 protoSegments[j].getLastUsedIteration(),
                                 nextFlatIdx_++};
      segments.push_back(segmentData);
      segmentForFlatIdx_.push_back(Segment(j, cell));

      auto protoSynapses = protoSegments[j].getSynapses();
      vector<SynapseData>& synapses = segments[j].synapses;

      for (SynapseIdx k = 0; k < protoSynapses.size(); ++k) {
        UInt32 presynapticCell = protoSynapses[k].getPresynapticCell();
        SynapseData synapseData = {presynapticCell,
                                   protoSynapses[k].getPermanence(),
                                   protoSynapses[k].getDestroyed()};
        synapses.push_back(synapseData);

        if (synapseData.destroyed)
        {
          segments[j].numDestroyedSynapses++;
        }
        else
        {
          numSynapses_++;

          Synapse synapse = Synapse(k, Segment(j, cell));
          synapsesForPresynapticCell_[presynapticCell].push_back(synapse);
        }
      }

      if (segmentData.destroyed)
      {
        cells_[cell].numDestroyedSegments++;
      }
      else
      {
        numSegments_++;
      }
    }
  }

  iteration_ = proto.getIteration();
}

CellIdx Connections::numCells() const
{
  return cells_.size();
}

UInt Connections::numSegments() const
{
  return numSegments_;
}

UInt Connections::numSegments(UInt32 cell) const
{
  const CellData& cellData = cells_[cell];
  return cellData.segments.size() - cellData.numDestroyedSegments;
}

UInt Connections::numSynapses() const
{
  return numSynapses_;
}

UInt Connections::numSynapses(const Segment& segment) const
{
  const SegmentData& segmentData = cells_[segment.cell].segments[segment.idx];
  return segmentData.synapses.size() - segmentData.numDestroyedSynapses;
}

bool Connections::operator==(const Connections &other) const
{
  if (maxSegmentsPerCell_ != other.maxSegmentsPerCell_) return false;
  if (maxSynapsesPerSegment_ != other.maxSynapsesPerSegment_) return false;

  if (cells_.size() != other.cells_.size()) return false;

  for (UInt32 i = 0; i < cells_.size(); ++i) {
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

        if (synapse.presynapticCell != otherSynapse.presynapticCell) return false;
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
      if (cell != otherCell) return false;
    }
  }

  if (numSegments_ != other.numSegments_) return false;
  if (numSynapses_ != other.numSynapses_) return false;
  if (iteration_ != other.iteration_) return false;

  return true;
}

SegmentExcitationTally::SegmentExcitationTally(
  const Connections& connections,
  Permanence activePermanenceThreshold,
  Permanence matchingPermanenceThreshold)
  :connections_(connections),
   activePermanenceThreshold_(activePermanenceThreshold),
   matchingPermanenceThreshold_(matchingPermanenceThreshold),
   numActiveSynapsesForSegment_(connections.nextFlatIdx_, 0),
   numMatchingSynapsesForSegment_(connections.nextFlatIdx_, 0)
{
  NTA_ASSERT(matchingPermanenceThreshold <= activePermanenceThreshold);
}

void SegmentExcitationTally::addActivePresynapticCell(CellIdx cell)
{
  if (connections_.synapsesForPresynapticCell_.count(cell))
  {
    for (const Synapse& synapse :
           connections_.synapsesForPresynapticCell_.at(cell))
    {
      const SynapseData& synapseData = connections_.dataForSynapse_(synapse);

      NTA_ASSERT(synapseData.permanence > 0);

      if (synapseData.permanence >= matchingPermanenceThreshold_ - EPSILON)
      {
        const SegmentData& segmentData =
          connections_.dataForSegment_(synapse.segment);

        ++numMatchingSynapsesForSegment_[segmentData.flatIdx];

        if (synapseData.permanence >= activePermanenceThreshold_ - EPSILON)
        {
          ++numActiveSynapsesForSegment_[segmentData.flatIdx];
        }
      }
    }
  }
}

void SegmentExcitationTally::getResults(
  SynapseIdx activeSynapseThreshold,
  SynapseIdx matchingSynapseThreshold,
  vector<SegmentOverlap>& activeSegmentsOut,
  vector<SegmentOverlap>& matchingSegmentsOut) const
{
  for (size_t i = 0; i < numActiveSynapsesForSegment_.size(); i++)
  {
    if (numActiveSynapsesForSegment_[i] >= activeSynapseThreshold)
    {
      const SegmentOverlap segmentOverlap =
        {connections_.segmentForFlatIdx_[i], numActiveSynapsesForSegment_[i]};
      activeSegmentsOut.push_back(segmentOverlap);
    }
  }

  for (size_t i = 0; i < numMatchingSynapsesForSegment_.size(); i++)
  {
    if (numMatchingSynapsesForSegment_[i] >= matchingSynapseThreshold)
    {
      const SegmentOverlap segmentOverlap =
        {connections_.segmentForFlatIdx_[i], numMatchingSynapsesForSegment_[i]};
      matchingSegmentsOut.push_back(segmentOverlap);
    }
  }

  std::sort(activeSegmentsOut.begin(), activeSegmentsOut.end(),
            segmentIncreasingOrder);
  std::sort(matchingSegmentsOut.begin(), matchingSegmentsOut.end(),
            segmentIncreasingOrder);
}

bool Segment::operator==(const Segment &other) const
{
  return idx == other.idx && cell == other.cell;
}

bool Segment::operator<=(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell <= other.cell;
  }
  else
  {
    return idx <= other.idx;
  }
}

bool Segment::operator<(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell < other.cell;
  }
  else
  {
    return idx < other.idx;
  }
}

bool Segment::operator>=(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell >= other.cell;
  }
  else
  {
    return idx >= other.idx;
  }
}

bool Segment::operator>(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell > other.cell;
  }
  else
  {
    return idx > other.idx;
  }
}

bool Synapse::operator==(const Synapse &other) const
{
  return idx == other.idx && segment == other.segment;
}
