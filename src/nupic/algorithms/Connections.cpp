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

#include <algorithm> // nth_element
#include <climits>
#include <iomanip>
#include <iostream>

#include <nupic/algorithms/Connections.hpp>

#include <nupic/math/Math.hpp> // nupic::Epsilon

using std::endl;
using std::string;
using std::vector;
using namespace nupic;
using namespace nupic::algorithms::connections;

Connections::Connections(CellIdx numCells, Permanence connectedThreshold) {
  initialize(numCells, connectedThreshold);
}

void Connections::initialize(CellIdx numCells, Permanence connectedThreshold) {
  cells_ = vector<CellData>(numCells);
  segments_.clear();
  destroyedSegments_.clear();
  synapses_.clear();
  destroyedSynapses_.clear();
  potentialSynapsesForPresynapticCell_.clear();
  connectedSynapsesForPresynapticCell_.clear();
  potentialSegmentsForPresynapticCell_.clear();
  connectedSegmentsForPresynapticCell_.clear();
  segmentOrdinals_.clear();
  synapseOrdinals_.clear();
  eventHandlers_.clear();
  NTA_CHECK(connectedThreshold >= minPermanence);
  NTA_CHECK(connectedThreshold <= maxPermanence);
  connectedThreshold_ = connectedThreshold - nupic::Epsilon;

  // Every time a segment or synapse is created, we assign it an ordinal and
  // increment the nextOrdinal. Ordinals are never recycled, so they can be used
  // to order segments or synapses by age.
  nextSegmentOrdinal_ = 0;
  nextSynapseOrdinal_ = 0;

  nextEventToken_ = 0;
}

UInt32 Connections::subscribe(ConnectionsEventHandler *handler) {
  UInt32 token = nextEventToken_++;
  eventHandlers_[token] = handler;
  return token;
}

void Connections::unsubscribe(UInt32 token) {
  delete eventHandlers_.at(token);
  eventHandlers_.erase(token);
}

Segment Connections::createSegment(CellIdx cell) {
  Segment segment;
  if (destroyedSegments_.size() > 0) {
    segment = destroyedSegments_.back();
    destroyedSegments_.pop_back();
  } else {
    segment = (Segment)segments_.size();
    segments_.push_back(SegmentData());
    segmentOrdinals_.push_back(0);
  }

  SegmentData &segmentData = segments_[segment];
  segmentData.numConnected = 0;
  segmentData.cell = cell;

  CellData &cellData = cells_[cell];
  segmentOrdinals_[segment] = nextSegmentOrdinal_++;
  cellData.segments.push_back(segment);

  for (auto h : eventHandlers_) {
    h.second->onCreateSegment(segment);
  }

  return segment;
}

Synapse Connections::createSynapse(Segment segment,
                                   CellIdx presynapticCell,
                                   Permanence permanence) {
  // Get an index into the synapses_ list, for the new synapse to reside at.
  Synapse synapse;
  if (destroyedSynapses_.size() > 0) {
    synapse = destroyedSynapses_.back();
    destroyedSynapses_.pop_back();
  } else {
    synapse = (UInt)synapses_.size();
    synapses_.push_back(SynapseData());
    synapseOrdinals_.push_back(0);
  }

  // Fill in the new synapse's data
  SynapseData &synapseData    = synapses_[synapse];
  synapseData.presynapticCell = presynapticCell;
  synapseData.segment         = segment;
  synapseOrdinals_[synapse]   = nextSynapseOrdinal_++;
  // Start in disconnected state.
  synapseData.permanence           = connectedThreshold_ - 1.0f;
  synapseData.presynapticMapIndex_ = 
    (Synapse)potentialSynapsesForPresynapticCell_[presynapticCell].size();
  potentialSynapsesForPresynapticCell_[presynapticCell].push_back(synapse);
  potentialSegmentsForPresynapticCell_[presynapticCell].push_back(segment);

  SegmentData &segmentData = segments_[segment];
  segmentData.synapses.push_back(synapse);


  for (auto h : eventHandlers_) {
    h.second->onCreateSynapse(synapse);
  }

  updateSynapsePermanence(synapse, permanence);

  return synapse;
}

bool Connections::segmentExists_(Segment segment) const {
  const SegmentData &segmentData = segments_[segment];
  const vector<Segment> &segmentsOnCell = cells_[segmentData.cell].segments;
  return (std::find(segmentsOnCell.begin(), segmentsOnCell.end(), segment) !=
          segmentsOnCell.end());
}

bool Connections::synapseExists_(Synapse synapse) const {
  const SynapseData &synapseData = synapses_[synapse];
  const vector<Synapse> &synapsesOnSegment =
      segments_[synapseData.segment].synapses;
  return (std::find(synapsesOnSegment.begin(), synapsesOnSegment.end(),
                    synapse) != synapsesOnSegment.end());
}

/**
 * Helper method to remove a synapse from a presynaptic map, by moving the
 * last synapse in the list over this synapse.
 */
void Connections::removeSynapseFromPresynapticMap_(
    const Synapse index,
    vector<Synapse> &preSynapses,
    vector<Segment> &preSegments)
{
  NTA_ASSERT( !preSynapses.empty() );
  NTA_ASSERT( index < preSynapses.size() );
  NTA_ASSERT( preSynapses.size() == preSegments.size() );

  const auto move = preSynapses.back();
  synapses_[move].presynapticMapIndex_ = index;
  preSynapses[index] = move;
  preSynapses.pop_back();

  preSegments[index] = preSegments.back();
  preSegments.pop_back();
}

void Connections::destroySegment(Segment segment) {
  NTA_ASSERT(segmentExists_(segment));
  for (auto h : eventHandlers_) {
    h.second->onDestroySegment(segment);
  }

  SegmentData &segmentData = segments_[segment];

  // Destroy synapses from the end of the list, so that the index-shifting is
  // easier to do.
  while( !segmentData.synapses.empty() )
    destroySynapse(segmentData.synapses.back());

  CellData &cellData = cells_[segmentData.cell];

  const auto segmentOnCell =
      std::lower_bound(cellData.segments.begin(), cellData.segments.end(),
                       segment, [&](Segment a, Segment b) {
                         return segmentOrdinals_[a] < segmentOrdinals_[b];
                       });

  NTA_ASSERT(segmentOnCell != cellData.segments.end());
  NTA_ASSERT(*segmentOnCell == segment);

  cellData.segments.erase(segmentOnCell);

  destroyedSegments_.push_back(segment);
}

void Connections::destroySynapse(Synapse synapse) {
  NTA_ASSERT(synapseExists_(synapse));
  for (auto h : eventHandlers_) {
    h.second->onDestroySynapse(synapse);
  }

  const SynapseData &synapseData = synapses_[synapse];
        SegmentData &segmentData = segments_[synapseData.segment];
  const auto         presynCell  = synapseData.presynapticCell;

  if( synapseData.permanence >= connectedThreshold_ ) {
    segmentData.numConnected--;

    removeSynapseFromPresynapticMap_(
      synapseData.presynapticMapIndex_,
      connectedSynapsesForPresynapticCell_.at( presynCell ),
      connectedSegmentsForPresynapticCell_.at( presynCell ));

    if( connectedSynapsesForPresynapticCell_.at( presynCell ).empty() ){
      connectedSynapsesForPresynapticCell_.erase( presynCell );
      connectedSegmentsForPresynapticCell_.erase( presynCell );
    }
  }
  else {
    removeSynapseFromPresynapticMap_(
      synapseData.presynapticMapIndex_,
      potentialSynapsesForPresynapticCell_.at( presynCell ),
      potentialSegmentsForPresynapticCell_.at( presynCell ));

    if( potentialSynapsesForPresynapticCell_.at( presynCell ).empty() ){
      potentialSynapsesForPresynapticCell_.erase( presynCell );
      potentialSegmentsForPresynapticCell_.erase( presynCell );
    }
  }

  const auto synapseOnSegment =
      std::lower_bound(segmentData.synapses.begin(), segmentData.synapses.end(),
                       synapse, [&](Synapse a, Synapse b) {
                         return synapseOrdinals_[a] < synapseOrdinals_[b];
                       });

  NTA_ASSERT(synapseOnSegment != segmentData.synapses.end());
  NTA_ASSERT(*synapseOnSegment == synapse);

  segmentData.synapses.erase(synapseOnSegment);

  destroyedSynapses_.push_back(synapse);
}

void Connections::updateSynapsePermanence(Synapse synapse,
                                          Permanence permanence) {
  permanence = std::min(permanence, maxPermanence );
  permanence = std::max(permanence, minPermanence );

  auto &synData = synapses_[synapse];
  
  const bool before = synData.permanence >= connectedThreshold_;
  const bool after  = permanence         >= connectedThreshold_;
  synData.permanence = permanence;

  if( before == after ) { //no change
      return;
  }
    const auto &presyn    = synData.presynapticCell;
    auto &potentialPresyn = potentialSynapsesForPresynapticCell_[presyn];
    auto &potentialPreseg = potentialSegmentsForPresynapticCell_[presyn];
    auto &connectedPresyn = connectedSynapsesForPresynapticCell_[presyn];
    auto &connectedPreseg = connectedSegmentsForPresynapticCell_[presyn];
    const auto &segment   = synData.segment;
    auto &segmentData     = segments_[segment];
    if( after ) {
      segmentData.numConnected++;

      // Remove this synapse from presynaptic potential synapses.
      removeSynapseFromPresynapticMap_( synData.presynapticMapIndex_,
                                        potentialPresyn, potentialPreseg );

      // Add this synapse to the presynaptic connected synapses.
      synData.presynapticMapIndex_ = (Synapse)connectedPresyn.size();
      connectedPresyn.push_back( synapse );
      connectedPreseg.push_back( segment );
    }
    else {
      segmentData.numConnected--;

      // Remove this synapse from presynaptic connected synapses.
      removeSynapseFromPresynapticMap_( synData.presynapticMapIndex_,
                                        connectedPresyn, connectedPreseg );

      // Add this synapse to the presynaptic connected synapses.
      synData.presynapticMapIndex_ = (Synapse)potentialPresyn.size();
      potentialPresyn.push_back( synapse );
      potentialPreseg.push_back( segment );
    }

    for (auto h : eventHandlers_) { //TODO handle callbacks in performance-critical method only in Debug?
      h.second->onUpdateSynapsePermanence(synapse, permanence);
    }
}

const vector<Segment> &Connections::segmentsForCell(CellIdx cell) const {
  return cells_[cell].segments;
}

Segment Connections::getSegment(CellIdx cell, SegmentIdx idx) const {
  return cells_[cell].segments[idx];
}

const vector<Synapse> &Connections::synapsesForSegment(Segment segment) const {
  NTA_ASSERT(segment < segments_.size()) << "Segment out of bounds! " << segment;
  return segments_[segment].synapses;
}

CellIdx Connections::cellForSegment(Segment segment) const {
  return segments_[segment].cell;
}

SegmentIdx Connections::idxOnCellForSegment(Segment segment) const {
  const vector<Segment> &segments = segmentsForCell(cellForSegment(segment));
  const auto it = std::find(segments.begin(), segments.end(), segment);
  NTA_ASSERT(it != segments.end());
  return (SegmentIdx)std::distance(segments.begin(), it);
}

void Connections::mapSegmentsToCells(const Segment *segments_begin,
                                     const Segment *segments_end,
                                     CellIdx *cells_begin) const {
  CellIdx *out = cells_begin;

  for (auto segment = segments_begin; segment != segments_end;
       ++segment, ++out) {
    NTA_ASSERT(segmentExists_(*segment));
    *out = segments_[*segment].cell;
  }
}

Segment Connections::segmentForSynapse(Synapse synapse) const {
  return synapses_[synapse].segment;
}

const SegmentData &Connections::dataForSegment(Segment segment) const {
  return segments_[segment];
}

const SynapseData &Connections::dataForSynapse(Synapse synapse) const {
  return synapses_[synapse];
}

Segment Connections::segmentFlatListLength() const { return (Segment)segments_.size(); }

bool Connections::compareSegments(const Segment a, const Segment b) const {
  const SegmentData &aData = segments_[a];
  const SegmentData &bData = segments_[b];
  if (aData.cell < bData.cell) {
    return true;
  } else if (bData.cell < aData.cell) {
    return false;
  } else {
    return segmentOrdinals_[a] < segmentOrdinals_[b];
  }
}

vector<Synapse>
Connections::synapsesForPresynapticCell(CellIdx presynapticCell) const {
  vector<Synapse> all(
      potentialSynapsesForPresynapticCell_.at(presynapticCell).begin(),
      potentialSynapsesForPresynapticCell_.at(presynapticCell).end());
  all.insert( all.end(),
      connectedSynapsesForPresynapticCell_.at(presynapticCell).begin(),
      connectedSynapsesForPresynapticCell_.at(presynapticCell).end());
  return all;
}

void Connections::computeActivity(
    vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
    vector<SynapseIdx> &numActivePotentialSynapsesForSegment,
    const CellIdx activePresynapticCell, 
    const Permanence connectedPermanence) const {
  std::vector<CellIdx> activePresynapticCells({activePresynapticCell});
  computeActivity(
    numActiveConnectedSynapsesForSegment,
    numActivePotentialSynapsesForSegment,
    activePresynapticCells, connectedPermanence);
}


void Connections::computeActivity(
    vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
    const vector<CellIdx> &activePresynapticCells) const
{
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == segments_.size());

  // Iterate through all connected synapses.
  for (const auto& cell : activePresynapticCells) {
    if (connectedSegmentsForPresynapticCell_.count(cell)) {
      for(const auto& segment : connectedSegmentsForPresynapticCell_.at(cell)) {
        ++numActiveConnectedSynapsesForSegment[segment];
      }
    }
  }
}

void Connections::computeActivity(
    vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
    vector<SynapseIdx> &numActivePotentialSynapsesForSegment,
    const vector<CellIdx> &activePresynapticCells,
    Permanence connectedPermanence) const {
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == segments_.size());
  NTA_ASSERT(numActivePotentialSynapsesForSegment.size() == segments_.size());
  NTA_CHECK( abs(connectedPermanence - nupic::Epsilon - connectedThreshold_) <= nupic::Epsilon );

  // Iterate through all connected synapses.
  computeActivity(
      numActiveConnectedSynapsesForSegment,
      activePresynapticCells );

  // Iterate through all potential synapses.
  std::copy( numActiveConnectedSynapsesForSegment.begin(),
             numActiveConnectedSynapsesForSegment.end(),
             numActivePotentialSynapsesForSegment.begin());
  for (const auto& cell : activePresynapticCells) {
    if (potentialSegmentsForPresynapticCell_.count(cell)) {
      for(const auto& segment : potentialSegmentsForPresynapticCell_.at(cell)) {
        ++numActivePotentialSynapsesForSegment[segment];
      }
    }
  }
}


void Connections::adaptSegment(const Segment segment, 
                               const SDR &inputs,
                               const Permanence increment,
                               const Permanence decrement)
{
  const vector<Synapse> &synapses = synapsesForSegment(segment);

  const auto &inputArray = inputs.getDense();

  for (SynapseIdx i = 0; i < synapses.size(); i++) {
    const SynapseData &synapseData = dataForSynapse(synapses[i]);

    Permanence permanence = synapseData.permanence;
    if( inputArray[synapseData.presynapticCell] ) {
      permanence += increment;
    } else {
      permanence -= decrement;
    }

    updateSynapsePermanence(synapses[i], permanence);
  }
}

/** called for under-performing Segments. (can have synapses pruned, etc.)
 * After the call, Segment will have at least 
 * segmentThreshold synapses connected (>= permanenceThreshold).
 * So the Segment could likely be active next time.
 */
void Connections::raisePermanencesToThreshold(
                  const Segment    segment,
                  const Permanence permanenceThreshold,
                  const UInt       segmentThreshold)
{
  if( segmentThreshold == 0 ) //no synapses requested to be connected, done.
    return;

  NTA_ASSERT(segment < segments_.size()) << "Accessing segment out of bounds.";
  auto &segData = segments_[segment];
  if( segData.numConnected >= segmentThreshold ) //the segment already satisfies the requirement, done.
    return;

  vector<Synapse> &synapses = segData.synapses;
  if( synapses.empty()) return; //no synapses to raise permanences to, no work
  // Prune empty segment? No. 
  // The SP calls this method, but the SP does not do any pruning. 
  // The TM already has code to do pruning, but it doesn't ever call this method.

  // There can be situation when synapses are pruned so the segment has too few synapses to ever activate. 
  // (so we cannot satisfy the >= segmentThreshold connected). 
  // In this case the method should do the next best thing and connect as many synapses as it can.
  //
  //keep segmentThreshold within synapses range
  const auto threshold = std::min((size_t)segmentThreshold, synapses.size());


  // Sort the potential pool by permanence values, and look for the synapse with
  // the N'th greatest permanence, where N is the desired minimum number of
  // connected synapses.  Then calculate how much to increase the N'th synapses
  // permance by such that it becomes a connected synapse.
  // After that there will be at least N synapses connected.

  auto minPermSynPtr = synapses.begin() + threshold - 1; //threshold is ensured to be >=1 by condition at very beginning if(thresh == 0)... 
  // Do a partial sort, it's faster than a full sort. Only minPermSynPtr is in
  // its final sorted position.
  const auto permanencesGreater = [&](const Synapse &A, const Synapse &B)
    { return synapses_[A].permanence > synapses_[B].permanence; };
  std::nth_element(synapses.begin(), minPermSynPtr, synapses.end(), permanencesGreater);

  const Real increment = permanenceThreshold - synapses_[ *minPermSynPtr ].permanence;
  if( increment <= 0 ) // if( minPermSynPtr is already connected ) then ...
    return;            // Enough synapses are already connected.

  // Raise the permance of all synapses in the potential pool uniformly.
  for( const auto &syn : synapses ) //TODO vectorize: vector + const to all members
    updateSynapsePermanence(syn, synapses_[syn].permanence + increment);
}


void Connections::bumpSegment(const Segment segment, const Permanence delta) {
  const vector<Synapse> &synapses = synapsesForSegment(segment);
  for( const auto &syn : synapses ) {
    updateSynapsePermanence(syn, synapses_[syn].permanence + delta);
  }
}


void Connections::save(std::ostream &outStream) const {
  outStream << std::setprecision(std::numeric_limits<Real32>::max_digits10);
  outStream << std::setprecision(std::numeric_limits<Real64>::max_digits10);

  // Write a starting marker.
  outStream << "Connections" << endl;
  outStream << VERSION << endl;

  outStream << cells_.size() << " " << endl;
  // Save the original permanence threshold, not the private copy which is used
  // only for floating point comparisons.
  outStream << connectedThreshold_ + nupic::Epsilon << " " << endl;

  for (CellData cellData : cells_) {
    const vector<Segment> &segments = cellData.segments;
    outStream << segments.size() << " ";

    for (Segment segment : segments) {
      const SegmentData &segmentData = segments_[segment];

      const vector<Synapse> &synapses = segmentData.synapses;
      outStream << synapses.size() << " ";

      for (Synapse synapse : synapses) {
        const SynapseData &synapseData = synapses_[synapse];
        outStream << synapseData.presynapticCell << " ";
        outStream << synapseData.permanence << " ";
      }
      outStream << endl;
    }
    outStream << endl;
  }
  outStream << endl;

  outStream << "~Connections" << endl;
}


void Connections::load(std::istream &inStream) {
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "Connections");

  // Check the saved version.
  int version;
  inStream >> version;
  NTA_CHECK(version == 2);

  // Retrieve simple variables
  UInt        numCells;
  Permanence  connectedThreshold;
  inStream >> numCells;
  inStream >> connectedThreshold;
  initialize(numCells, connectedThreshold);

  for (UInt cell = 0; cell < numCells; cell++) {

    UInt numSegments;
    inStream >> numSegments;

    for (SegmentIdx j = 0; j < numSegments; j++) {
      Segment segment = createSegment( cell );

      UInt numSynapses;
      inStream >> numSynapses;

      for (SynapseIdx k = 0; k < numSynapses; k++) {
        CellIdx     presyn;
        Permanence  perm;
        inStream >> presyn;
        inStream >> perm;

        createSynapse( segment, presyn, perm );
      }
    }
  }

  inStream >> marker;
  NTA_CHECK(marker == "~Connections");
}


CellIdx Connections::numCells() const { return (CellIdx)cells_.size(); }

Segment Connections::numSegments() const {
  return (Segment)(segments_.size() - destroyedSegments_.size());
}

SegmentIdx Connections::numSegments(CellIdx cell) const {
  return (SegmentIdx)cells_[cell].segments.size();
}

Synapse Connections::numSynapses() const {
  return (Synapse)(synapses_.size() - destroyedSynapses_.size());
}

SynapseIdx Connections::numSynapses(Segment segment) const {
  return (SynapseIdx)segments_[segment].synapses.size();
}

bool Connections::operator==(const Connections &other) const {
  if (cells_.size() != other.cells_.size())
    return false;

  for (CellIdx i = 0; i < cells_.size(); ++i) {
    const CellData &cellData = cells_[i];
    const CellData &otherCellData = other.cells_[i];

    if (cellData.segments.size() != otherCellData.segments.size()) {
      return false;
    }

    for (SegmentIdx j = 0; j < (SegmentIdx)cellData.segments.size(); ++j) {
      Segment segment = cellData.segments[j];
      const SegmentData &segmentData = segments_[segment];
      Segment otherSegment = otherCellData.segments[j];
      const SegmentData &otherSegmentData = other.segments_[otherSegment];

      if (segmentData.synapses.size() != otherSegmentData.synapses.size() ||
          segmentData.cell != otherSegmentData.cell) {
        return false;
      }

      for (SynapseIdx k = 0; k < (SynapseIdx)segmentData.synapses.size(); ++k) {
        Synapse synapse = segmentData.synapses[k];
        const SynapseData &synapseData = synapses_[synapse];
        Synapse otherSynapse = otherSegmentData.synapses[k];
        const SynapseData &otherSynapseData = other.synapses_[otherSynapse];

        if (synapseData.presynapticCell != otherSynapseData.presynapticCell ||
            synapseData.permanence != otherSynapseData.permanence) {
          return false;
        }

        // Two functionally identical instances may have different flatIdxs.
        NTA_ASSERT(synapseData.segment == segment);
        NTA_ASSERT(otherSynapseData.segment == otherSegment);
      }
    }
  }

  return true;
}

bool Connections::operator!=(const Connections &other) const {
  return !(*this == other);
}
