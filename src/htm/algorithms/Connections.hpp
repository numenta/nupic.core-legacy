/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2014-2016, Numenta, Inc.
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
 * ---------------------------------------------------------------------- */

/** @file
 * Definitions for the Connections class in C++
 */

#ifndef NTA_CONNECTIONS_HPP
#define NTA_CONNECTIONS_HPP

#include <limits>
#include <map>
#include <unordered_map>
#include <set>
#include <utility>
#include <vector>
#include <deque>

#include <htm/types/Types.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/types/Sdr.hpp>

namespace htm {

//TODO instead of typedefs, use templates for proper type-checking?
using CellIdx   = htm::ElemSparse; // CellIdx must match with ElemSparse, defined in Sdr.hpp
using SegmentIdx= UInt16; /** Index of segment in cell. */
using SynapseIdx= UInt16; /** Index of synapse in segment. */
using Segment   = UInt32;    /** Index of segment's data. */
using Synapse   = UInt32;    /** Index of synapse's data. */
using Permanence= Real32; //TODO experiment with half aka float16
constexpr const Permanence minPermanence = 0.0f;
constexpr const Permanence maxPermanence = 1.0f;



/**
 * SynapseData class used in Connections.
 *
 * @b Description
 * The SynapseData contains the underlying data for a synapse.
 *
 * @param presynapticCellIdx
 * Cell that this synapse gets input from.
 *
 * @param permanence
 * Permanence of synapse.
 */
struct SynapseData: public Serializable {
  CellIdx presynapticCell;
  Permanence permanence;
  Segment segment;
  Synapse presynapticMapIndex_;
  Synapse id;

  SynapseData() {}

  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const {
    ar(cereal::make_nvp("perm", permanence),
      cereal::make_nvp("presyn", presynapticCell));
  }
  template<class Archive>
  void load_ar(Archive & ar) {
    ar( permanence, presynapticCell);
  }

};

/**
 * SegmentData class used in Connections.
 *
 * @b Description
 * The SegmentData contains the underlying data for a Segment.
 *
 * @param synapses
 * Synapses on this segment.
 *
 * @param cell
 * The cell that this segment is on.
 */
struct SegmentData {
  SegmentData(const CellIdx cell, Segment id, UInt32 lastUsed = 0) : cell(cell), numConnected(0), lastUsed(lastUsed), id(id) {} //default constructor

  std::vector<Synapse> synapses;
  CellIdx cell; //mother cell that this segment originates from
  SynapseIdx numConnected; //number of permanences from `synapses` that are >= synPermConnected, ie connected synapses
  UInt32 lastUsed = 0; //last used time (iteration). Used for segment pruning by "least recently used" (LRU) in `createSegment`
  Segment id; 
};

/**
 * CellData class used in Connections.
 * A cell consists of segments and in Connections is indexed by CellIdx.
 *
 * @b Description
 * The CellData contains the underlying data for a Cell.
 *
 * @param segments
 * Segments on this cell.
 *
 */
struct CellData {
  std::vector<Segment> segments;
};

/**
 * A base class for Connections event handlers.
 *
 * @b Description
 * This acts as a plug-in point for logging / visualizations.
 */
class ConnectionsEventHandler {
public:
  virtual ~ConnectionsEventHandler() {}

  /**
   * Called after a segment is created.
   */
  virtual void onCreateSegment(Segment segment) {}

  /**
   * Called before a segment is destroyed.
   */
  virtual void onDestroySegment(Segment segment) {}

  /**
   * Called after a synapse is created.
   */
  virtual void onCreateSynapse(Synapse synapse) {}

  /**
   * Called before a synapse is destroyed.
   */
  virtual void onDestroySynapse(Synapse synapse) {}

  /**
   * Called after a synapse's permanence crosses the connected threshold.
   */
  virtual void onUpdateSynapsePermanence(Synapse synapse,
                                         Permanence permanence) {}
};

/**
 * Connections implementation in C++.
 *
 * @b Description
 * The Connections class is a data structure that represents the
 * connections of a collection of cells. It is used in the HTM
 * learning algorithms to store and access data related to the
 * connectivity of cells.
 *
 * Its main utility is to provide a common, optimized data structure
 * that all HTM learning algorithms can use. It is flexible enough to
 * support any learning algorithm that operates on a collection of cells.
 *
 * Each type of connection (proximal, distal basal, apical) should be
 * represented by a different instantiation of this class. This class
 * will help compute the activity along those connections due to active
 * input cells. The responsibility for what effect that activity has on
 * the cells and connections lies in the user of this class.
 *
 * This class is optimized to store connections between cells, and
 * compute the activity of cells due to input over the connections.
 *
 * This class assigns each segment a unique "flatIdx" so that it's
 * possible to use a simple vector to associate segments with values.
 * Create a vector of length `connections.segmentFlatListLength()`,
 * iterate over segments and update the vector at index `segment`.
 *
 */
class Connections : public Serializable
 {
public:
  static const UInt16 VERSION = 2;

  /**
   * Connections empty constructor.
   * (Does not call `initialize`.)
   */
  Connections(){};

  /**
   * Connections constructor.
   *
   * @param numCells           Number of cells.
   * @param connectedThreshold Permanence threshold for synapses connecting or
   *                           disconnecting.
   *
   * @params timeseries - Optional, default false.  If true AdaptSegment will not
   * apply the same learning update to a synapse on consequetive cycles, because
   * then staring at the same object for too long will mess up the synapses.
   * IE Highly correlated inputs will cause the synapse permanences to saturate.
   * This change allows it to work with timeseries data which moves very slowly,
   * instead of the usual HTM inputs which reliably change every cycle.  See
   * also (Kropff & Treves, 2007. http://dx.doi.org/10.2976/1.2793335).
   */
  Connections(const CellIdx numCells, 
	      const Permanence connectedThreshold = 0.5f,
              const bool timeseries = false);

  virtual ~Connections() {}

  /**
   * Initialize connections.
   *
   * @param numCells           Number of cells.
   * @param connectedThreshold Permanence threshold for synapses connecting or
   *                           disconnecting.
   * @param timeseries         See constructor.
   */
  void initialize(const CellIdx numCells, 
		  const Permanence connectedThreshold = 0.5f,
                  const bool timeseries = false);

  /**
   * Creates a segment on the specified cell.
   *
   * @param cell Cell to create segment on.
   *
   * @param maxSegmetsPerCell Optional. Enforce limit on maximum number of segments that can be
   * created on a Cell. If the limit is exceeded, call `destroySegment` to remove least used segments 
   * (ordered by LRU `SegmentData.lastUsed`). Default value is numeric_limits::max() of the data-type, 
   * so effectively disabled. 
   *
   * @retval Unique ID of the created segment `seg`. Use `dataForSegment(seg)` to obtain the segment's data. 
   * Use  `idxOfSegmentOnCell()` to get SegmentIdx of `seg` on this `cell`. 
   *
   */
  Segment createSegment(const CellIdx cell, 
		        const SegmentIdx maxSegmentsPerCell = std::numeric_limits<SegmentIdx>::max());

  /**
   * Creates a synapse on the specified segment.
   *
   * @param segment         Segment to create synapse on.
   * @param presynapticCell Cell to synapse on.
   * @param permanence      Initial permanence of new synapse.
   *
   * @reval Created synapse.
   */
  Synapse createSynapse(const Segment segment,
                        const CellIdx presynapticCell,
                        Permanence permanence);

  /**
   * Destroys segment.
   *
   * @param segment Segment to destroy.
   */
  void destroySegment(const Segment segment);

  /**
   * Destroys synapse.
   *
   * @param synapse Synapse to destroy.
   */
  void destroySynapse(const Synapse synapse);

  /**
   * Updates a synapse's permanence.
   *
   * @param synapse    Synapse to update.
   * @param permanence New permanence.
   */
  void updateSynapsePermanence(const Synapse synapse, 
		               Permanence permanence);

  /**
   * Gets the segments for a cell.
   *
   * @param cell Cell to get segments for.
   *
   * @retval Segments on cell.
   */
  const std::vector<Segment> &segmentsForCell(const CellIdx cell) const {
    return cells_[cell].segments;
  }

  /**
   * Gets the synapses for a segment.
   *
   * @param segment Segment to get synapses for.
   *
   * @retval Synapses on segment.
   */
  const std::vector<Synapse> &synapsesForSegment(const Segment segment) const {
    NTA_ASSERT(segment < segments_.size()) << "Segment out of bounds! " << segment;
    return segments_[segment].synapses;
  }

  /**
   * Gets the cell that this segment is on.
   *
   * @param segment Segment to get the cell for.
   *
   * @retval Cell that this segment is on.
   */
  CellIdx cellForSegment(const Segment segment) const {
    return segments_[segment].cell;
  }

  /**
   * Gets the index of this segment on its respective cell.
   *
   * @param segment Segment to get the idx for.
   *
   * @retval Index of the segment.
   */
  SegmentIdx idxOnCellForSegment(const Segment segment) const;

  /**
   * Get the cell for each provided segment.
   *
   * @param segments
   * The segments to query
   *
   * @param cells
   * Output array with the same length as 'segments'
   */
  void mapSegmentsToCells(const Segment *segments_begin,
                          const Segment *segments_end,
                          CellIdx *cells_begin) const;

  /**
   * Gets the segment that this synapse is on.
   *
   * @param synapse Synapse to get Segment for.
   *
   * @retval Segment that this synapse is on.
   */
  Segment segmentForSynapse(const Synapse synapse) const {
    return synapses_[synapse].segment;
  }

  /**
   * Gets the data for a segment.
   *
   * @param segment Segment to get data for.
   *
   * @retval Segment data.
   */
  const SegmentData &dataForSegment(const Segment segment) const {
    return segments_[segment];
  }
  SegmentData& dataForSegment(const Segment segment) { //editable access, needed by SP 
    return segments_[segment];
  }

  /**
   * Gets the data for a synapse.
   *
   * @param synapse Synapse to get data for.
   *
   * @retval Synapse data.
   */
  const SynapseData &dataForSynapse(const Synapse synapse) const {
    return synapses_[synapse];
  }

  /**
   * Get the segment at the specified cell and offset.
   *
   * @param cell The cell that the segment is on.
   * @param idx The index of the segment on the cell.
   *
   * @retval Segment
   */
  Segment getSegment(const CellIdx cell, const SegmentIdx idx) const {
    return cells_[cell].segments[idx];
  }

  /**
   * Get the vector length needed to use segments as indices.
   *
   * @retval A vector length
   */
  size_t segmentFlatListLength() const { return segments_.size(); };

  /**
   * Compare two segments. Returns true if a < b.
   *
   * Segments are ordered first by cell, then by their order on the cell.
   *
   * @param a Left segment to compare
   * @param b Right segment to compare
   *
   * @retval true if a < b, false otherwise.
   */
  bool compareSegments(const Segment a, const Segment b) const;

  /**
   * Returns the synapses for the source cell that they synapse on.
   *
   * @param presynapticCell(int) Source cell index
   *
   * @return Synapse indices
   */
  std::vector<Synapse> synapsesForPresynapticCell(const CellIdx presynapticCell) const;

  /**
   * For use with time-series datasets.
   */
  void reset();

  /**
   * Compute the segment excitations for a vector of active presynaptic
   * cells.
   *
   * The output vectors aren't grown or cleared. They must be
   * preinitialized with the length returned by
   * getSegmentFlatVectorLength().
   *
   * @param numActiveConnectedSynapsesForSegment
   * An output vector for active connected synapse counts per segment.
   *
   * @param numActivePotentialSynapsesForSegment
   * An output vector for active potential synapse counts per segment.
   *
   * @param activePresynapticCells
   * Active cells in the input.
   *
   * @param bool learn : enable learning updates (default true)
   *
   */
  void computeActivity(std::vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
                       std::vector<SynapseIdx> &numActivePotentialSynapsesForSegment,
                       const std::vector<CellIdx> &activePresynapticCells,
		       const bool learn = true);

  void computeActivity(std::vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
                       const std::vector<CellIdx> &activePresynapticCells,
		       const bool learn = true);

  /**
   * The primary method in charge of learning.   Adapts the permanence values of
   * the synapses based on the input SDR.  Learning is applied to a single
   * segment.  Permanence values are increased for synapses connected to input
   * bits that are turned on, and decreased for synapses connected to inputs
   * bits that are turned off.
   *
   * @param segment  Index of segment to apply learning to.  Is returned by 
   *        method getSegment.
   * @param inputVector  An SDR
   * @param increment  Change in permanence for synapses with active presynapses.
   * @param decrement  Change in permanence for synapses with inactive presynapses.
   * @param pruneZeroSynapses (default false) If set, synapses that reach minPermanence(aka. "zero")
   *        are removed. This is used in TemporalMemory.  If the segment becomes empty due to these
   *        removed synapses, we remove the segment (see @ref `destroySegment`).
   */
  void adaptSegment(const Segment segment,
                    const SDR &inputs,
                    const Permanence increment,
                    const Permanence decrement,
		    const bool pruneZeroSynapses = false);

  /**
   * Ensures a minimum number of connected synapses.  This raises permance
   * values until the desired number of synapses have permanences above the
   * connectedThreshold.  This is applied to a single segment.
   *
   * @param segment  Index of segment in connections. Is returned by method getSegment.
   * @param segmentThreshold  Desired number of connected synapses.
   */
  void raisePermanencesToThreshold(const Segment    segment,
                                   const UInt       segmentThreshold);


  /**
   *  iteration: ever increasing step count. 
   *  Increases each main call to "compute". Since connections has more
   *  methods that are called instead of compute (adaptSegment, computeActivity,..)
   *  this counter is increased in @ref `computeActivity` as it is called by both
   *  SP & TM. 
   */
//!  const UInt32& iteration = iteration_; //FIXME cannot construct iteration like this?
  UInt32 iteration() const { return iteration_; }


  /**
   * Ensures that the number of connected synapses is sane.  This method
   * controls the sparsity of the synaptic connections, which is important for
   * the segment to detect things.  If there are too few connections then the
   * segment will not detect anything, and if there are too many connections
   * then the segment will detect everything.
   *
   * See file: docs/synapse_competition.docx
   *
   * This method connects and disconnects synapses by uniformly changing the
   * permanences of all synapses on the segment.
   *
   * @param segment  Index of segment in connections. Is returned by method getSegment.
   * @param minimumSynapses Minimum number of connected synapses allowed on this segment (inclusive).
   * @param maximumSynapses Maximum number of connected synapses allowed on this segment (inclusive).
   */
  void synapseCompetition(  const Segment    segment,
                            const SynapseIdx minimumSynapses,
                            const SynapseIdx maximumSynapses);


  /**
   * Modify all permanence on the given segment, uniformly.
   *
   * @param segment  Index of segment on cell. Is returned by method getSegment.
   * @param delta  Change in permanence value
   */
  void bumpSegment(const Segment segment, const Permanence delta);

  /**
   * Destroy the synapses with the lowest permanence values.  This method is
   * useful for making room for more synapses on a segment which is already
   * full.
   *
   * @param segment - Index of segment in Connections, to be modified.
   * @param nDestroy - Must be greater than or equal to zero!
   * @param excludeCells - Presynaptic cells which will NOT have any synapses destroyed.
   */
  void destroyMinPermanenceSynapses(const Segment segment, Int nDestroy,
                                    const SDR_sparse_t &excludeCells = {});

  /**
   * Print diagnostic info
   */
  friend std::ostream& operator<< (std::ostream& stream, const Connections& self);


  // Serialization
  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const {
    // make this look like a queue of items to be sent. 
    // and a queue of sizes so we can distribute the 
		// correct number for each level when deserializing.
    std::deque<SynapseData> syndata;
    std::deque<size_t> sizes;
    sizes.push_back(cells_.size());
    for (CellData cellData : cells_) {
      const std::vector<Segment> &segments = cellData.segments;
      sizes.push_back(segments.size());
      for (Segment segment : segments) {
        const SegmentData &segmentData = segments_[segment];
        const std::vector<Synapse> &synapses = segmentData.synapses;
        sizes.push_back(synapses.size());
        for (Synapse synapse : synapses) {
          const SynapseData &synapseData = synapses_[synapse];
          syndata.push_back(synapseData);
        }
      }
    }
    ar(CEREAL_NVP(connectedThreshold_));
    ar(CEREAL_NVP(sizes));
    ar(CEREAL_NVP(syndata));
    ar(CEREAL_NVP(iteration_));
  }

  template<class Archive>
  void load_ar(Archive & ar) {
    std::deque<size_t> sizes;
    std::deque<SynapseData> syndata;
    ar(CEREAL_NVP(connectedThreshold_));
    ar(CEREAL_NVP(sizes));
    ar(CEREAL_NVP(syndata));

    CellIdx numCells = static_cast<CellIdx>(sizes.front()); sizes.pop_front();
    initialize(numCells, connectedThreshold_);
    for (UInt cell = 0; cell < numCells; cell++) {
      size_t numSegments = sizes.front(); sizes.pop_front();
      for (SegmentIdx j = 0; j < static_cast<SegmentIdx>(numSegments); j++) {
        Segment segment = createSegment( cell );

        size_t numSynapses = sizes.front(); sizes.pop_front();
        for (SynapseIdx k = 0; k < static_cast<SynapseIdx>(numSynapses); k++) {
          SynapseData& syn = syndata.front(); syndata.pop_front();
          createSynapse( segment, syn.presynapticCell, syn.permanence );
        }
      }
    }
    ar(CEREAL_NVP(iteration_));
  }

  /**
   * Gets the number of cells.
   *
   * @retval Number of cells.
   */
  size_t numCells() const noexcept { return cells_.size(); }

  constexpr Permanence getConnectedThreshold() const noexcept { return connectedThreshold_; }

  /**
   * Gets the number of segments.
   *
   * @retval Number of segments.
   */
  size_t numSegments() const { 
	  NTA_ASSERT(segments_.size() >= destroyedSegments_.size());
	  return segments_.size() - destroyedSegments_.size(); }

  /**
   * Gets the number of segments on a cell.
   *
   * @retval Number of segments.
   */
  size_t numSegments(const CellIdx cell) const { 
	  return cells_[cell].segments.size(); 
  }

  /**
   * Gets the number of synapses.
   *
   * @retval Number of synapses.
   */
  size_t numSynapses() const {
    NTA_ASSERT(synapses_.size() >= destroyedSynapses_.size());
    return synapses_.size() - destroyedSynapses_.size();
  }

  /**
   * Gets the number of synapses on a segment.
   *
   * @retval Number of synapses.
   */
  size_t numSynapses(const Segment segment) const { 
	  return segments_[segment].synapses.size(); 
  }

  /**
   * Comparison operator.
   */
  bool operator==(const Connections &other) const;
  inline bool operator!=(const Connections &other) const { return !operator==(other); }

  /**
   * Add a connections events handler.
   *
   * The Connections instance takes ownership of the eventHandlers
   * object. Don't delete it. When calling from Python, call
   * eventHandlers.__disown__() to avoid garbage-collecting the object
   * while this instance is still using it. It will be deleted on
   * `unsubscribe`.
   *
   * @param handler
   * An object implementing the ConnectionsEventHandler interface
   *
   * @retval Unsubscribe token
   */
  UInt32 subscribe(ConnectionsEventHandler *handler);

  /**
   * Remove an event handler.
   *
   * @param token
   * The return value of `subscribe`.
   */
  void unsubscribe(UInt32 token);

protected:
  /**
   * Check whether this segment still exists on its cell.
   *
   * @param Segment
   *
   * @retval True if it's still in its cell's segment list.
   */
  bool segmentExists_(const Segment segment) const;

  /**
   * Check whether this synapse still exists on its segment.
   *
   * @param Synapse
   *
   * @retval True if it's still in its segment's synapse list.
   */
  bool synapseExists_(const Synapse synapse) const;

  /**
   * Remove a synapse from presynaptic maps.
   *
   * @param Synapse Index of synapse in presynaptic vector.
   *
   * @param vector<Synapse> ynapsesForPresynapticCell must a vector from be
   * either potentialSynapsesForPresynapticCell_ or
   * connectedSynapsesForPresynapticCell_, depending on whether the synapse is
   * connected or not.
   *
   * @param vector<Synapse> segmentsForPresynapticCell must be a vector from
   * either potentialSegmentsForPresynapticCell_ or
   * connectedSegmentsForPresynapticCell_, depending on whether the synapse is
   * connected or not.
   */
  void removeSynapseFromPresynapticMap_(const Synapse index,
                              std::vector<Synapse> &synapsesForPresynapticCell,
                              std::vector<Segment> &segmentsForPresynapticCell);

private:
  std::vector<CellData>    cells_;
  std::vector<SegmentData> segments_;
  std::vector<Segment>     destroyedSegments_;
  std::vector<SynapseData> synapses_;
  std::vector<Synapse>     destroyedSynapses_;
  Permanence               connectedThreshold_; //TODO make const
  UInt32 iteration_ = 0;

  // Extra bookkeeping for faster computing of segment activity.
 
  struct identity { constexpr size_t operator()( const CellIdx t ) const noexcept { return t; };   };	//TODO in c++20 use std::identity 

  std::unordered_map<CellIdx, std::vector<Synapse>, identity> potentialSynapsesForPresynapticCell_;
  std::unordered_map<CellIdx, std::vector<Synapse>, identity> connectedSynapsesForPresynapticCell_;
  std::unordered_map<CellIdx, std::vector<Segment>, identity> potentialSegmentsForPresynapticCell_;
  std::unordered_map<CellIdx, std::vector<Segment>, identity> connectedSegmentsForPresynapticCell_;

  Segment nextSegmentOrdinal_ = 0;
  Synapse nextSynapseOrdinal_ = 0;

  // These three members should be used when working with highly correlated
  // data. The vectors store the permanence changes made by adaptSegment.
  bool timeseries_;
  std::vector<Permanence> previousUpdates_;
  std::vector<Permanence> currentUpdates_;

  //for prune statistics
  Synapse prunedSyns_ = 0; //how many synapses have been removed?
  Segment prunedSegs_ = 0;

  //for listeners
  UInt32 nextEventToken_;
  std::map<UInt32, ConnectionsEventHandler *> eventHandlers_;
}; // end class Connections

} // end namespace htm

#endif // NTA_CONNECTIONS_HPP
