/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2016, Numenta, Inc.
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
 * Definitions for the Temporal Memory in C++
 */

#ifndef NTA_TEMPORAL_MEMORY_HPP
#define NTA_TEMPORAL_MEMORY_HPP

#include <htm/algorithms/Connections.hpp>
#include <htm/types/Types.hpp>
#include <htm/types/Sdr.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/utils/Random.hpp>
#include <vector>


namespace htm {

using namespace std;
using namespace htm;

/**
 * Temporal Memory implementation in C++.
 *
 * Example usage:
 *
 *     SpatialPooler sp(inputDimensions, columnDimensions, <parameters>);
 *     TemporalMemory tm(columnDimensions, <parameters>);
 *
 *     while (true) {
 *        <get input vector, streaming spatiotemporal information>
 *        sp.compute(inputVector, learn, activeColumns)
 *        tm.compute(activeColumns, learn)
 *        <do something with the tm, e.g. classify tm.getActiveCells()>
 *     }
 */
class TemporalMemory : public Serializable
{
public:
  TemporalMemory();

  /**
   * Initialize the temporal memory (TM) using the given parameters.
   *
   * @param columnDimensions
   * Dimensions of the column space
   *
   * @param cellsPerColumn
   * Number of cells per column
   *
   * @param activationThreshold
   * If the number of active connected synapses on a segment is at least
   * this threshold, the segment is said to be active.
   *
   * @param initialPermanence
   * Initial permanence of a new synapse.
   *
   * @param connectedPermanence
   * If the permanence value for a synapse is greater than this value, it
   * is said to be connected.
   *
   * @param minThreshold
   * If the number of potential synapses active on a segment is at least
   * this threshold, it is said to be "matching" and is eligible for
   * learning.
   *
   * @param maxNewSynapseCount
   * The maximum number of synapses added to a segment during learning.
   *
   * @param permanenceIncrement
   * Amount by which permanences of synapses are incremented during
   * learning.
   *
   * @param permanenceDecrement
   * Amount by which permanences of synapses are decremented during
   * learning.
   *
   * @param predictedSegmentDecrement
   * Amount by which segments are punished for incorrect predictions.
   *
   * Note: A good value is just a bit larger than
   * (the column-level sparsity * permanenceIncrement). So, if column-level
   * sparsity is 2% and permanenceIncrement is 0.01, this parameter should be
   * something like 4% * 0.01 = 0.0004).
   *
   * @param seed
   * Seed for the random number generator.
   *
   * @param maxSegmentsPerCell
   * The maximum number of segments per cell.
   * The value you can choose here is limited by the type SegmentIdx
   * in Connections.hpp, change it if you need larger values. 
   *
   * @param maxSynapsesPerSegment
   * The maximum number of synapses per segment.
   * The value you can choose here is limited by the type SynapseIdx
   * in Connections.hpp, change it there if you needed to use large values.
   *
   * @param checkInputs
   * Whether to check that the activeColumns are sorted without
   * duplicates. Disable this for a small speed boost.
   * DEPRECATED: The SDR class now enforces these properties.
   *
   * @param externalPredictiveInputs
   * Number of external predictive inputs.  These values are not related to this
   * TM, they represent input from a different region.  This TM will form
   * synapses with these inputs in addition to the cells which are part of this
   * TemporalMemory.  If this is given (and greater than 0) then the active
   * cells and winner cells of these external inputs must be given to methods
   * TM.compute and TM.activateDendrites
   */
  TemporalMemory(
      vector<CellIdx> columnDimensions,
      CellIdx         cellsPerColumn              = 32,
      SynapseIdx      activationThreshold         = 13,
      Permanence      initialPermanence           = 0.21,
      Permanence      connectedPermanence         = 0.50,
      SynapseIdx      minThreshold                = 10,
      SynapseIdx      maxNewSynapseCount          = 20,
      Permanence      permanenceIncrement         = 0.10,
      Permanence      permanenceDecrement         = 0.10,
      Permanence      predictedSegmentDecrement   = 0.0,
      Int             seed                        = 42,
      SegmentIdx      maxSegmentsPerCell          = 255,
      SynapseIdx      maxSynapsesPerSegment       = 255,
      bool            checkInputs                 = true,
      UInt            externalPredictiveInputs    = 0);

  virtual void
  initialize(
    vector<CellIdx>  columnDimensions            = {2048},
    CellIdx          cellsPerColumn              = 32,
    SynapseIdx       activationThreshold         = 13,
    Permanence    initialPermanence           = 0.21,
    Permanence    connectedPermanence         = 0.50,
    SynapseIdx    minThreshold                = 10,
    SynapseIdx    maxNewSynapseCount          = 20,
    Permanence    permanenceIncrement         = 0.10,
    Permanence    permanenceDecrement         = 0.10,
    Permanence    predictedSegmentDecrement   = 0.0,
    Int           seed                        = 42,
    SegmentIdx    maxSegmentsPerCell          = 255,
    SynapseIdx    maxSynapsesPerSegment       = 255,
    bool          checkInputs                 = true,
    UInt          externalPredictiveInputs    = 0);

  virtual ~TemporalMemory();

  //----------------------------------------------------------------------
  //  Main functions
  //----------------------------------------------------------------------

  /**
   * Get the version number of for the TM implementation.
   *
   * @returns Integer version number.
   */
  virtual UInt version() const;

  /**
   * Indicates the start of a new sequence.
   * Resets sequence state of the TM.
   */
  virtual void reset();

  /**
   * Calculate the active cells, using the current active columns and
   * dendrite segments. Grow and reinforce synapses.
   *
   * @param activeColumns
   * A sorted list of active column indices.
   *
   * @param learn
   * If true, reinforce / punish / grow synapses.
   */
  void activateCells(const SDR &activeColumns, 
                     const bool learn = true);

  /**
   * Calculate dendrite segment activity, using the current active cells.  Call
   * this method before calling getPredictiveCells, getActiveSegments, or
   * getMatchingSegments.  In each time step, only the first call to this
   * method has an effect, subsequent calls assume that the prior results are
   * still valid.
   *
   * @param learn
   * If true, segment activations will be recorded. This information is
   * used during segment cleanup.
   *
   * @param externalPredictiveInputsActive
   * (optional) SDR of active external predictive inputs.
   *
   * @param externalPredictiveInputsWinners
   * (optional) SDR of winning external predictive inputs.  When learning, only these
   * inputs are considered active.  
   * externalPredictiveInputsWinners must be a subset of externalPredictiveInputsActive.  
   *
   * See TM::compute() for details of the parameters. 
   *
   */
  void activateDendrites(const bool learn,
                         const SDR &externalPredictiveInputsActive, 
                         const SDR &externalPredictiveInputsWinners);

  inline void activateDendrites(const bool learn = true) {
    const SDR externalPredictiveInputsActive(std::vector<UInt>{ externalPredictiveInputs });
    const SDR externalPredictiveInputsWinners(std::vector<UInt>{externalPredictiveInputs });
    activateDendrites(learn, externalPredictiveInputsActive, externalPredictiveInputsWinners);
  }

  /**
   * Perform one time step of the Temporal Memory algorithm.
   *
   * This method calls activateDendrites, then calls activateCells. Using
   * the TemporalMemory via its compute method ensures that you'll always
   * be able to call getActiveCells at the end of the time step.
   *
   * @param activeColumns
   * Sorted SDR of active columns.
   *
   * @param learn
   * Whether or not learning is enabled.
   *
   * @param externalPredictiveInputsActive
   * (optional) Vector of active external predictive inputs.  
   * TM must be set up with the 'externalPredictiveInputs' constructor parameter for this use.
   *
   * @param externalPredictiveInputsWinners
   * (optional) Vector of winning external predictive inputs.  When learning, only these
   * inputs are considered active.  
   * externalPredictiveInputsWinners must be a subset of externalPredictiveInputsActive.  
   */
  virtual void compute(const SDR &activeColumns, 
                       const bool learn,
                       const SDR &externalPredictiveInputsActive, 
                       const SDR &externalPredictiveInputsWinners);

  virtual void compute(const SDR &activeColumns, 
                       const bool learn = true);

  // ==============================
  //  Helper functions
  // ==============================

  /**
   * Create a segment on the specified cell. This method calls
   * createSegment on the underlying connections, and it does some extra
   * bookkeeping. Unit tests should call this method, and not
   * connections.createSegment().
   *
   * @param cell
   * Cell to add a segment to.
   *
   * @return Segment
   * The created segment.
   */
  Segment createSegment(const CellIdx& cell) { 
	  return connections.createSegment(cell, maxSegmentsPerCell_); }

  /**
   * Returns the indices of cells that belong to a mini-column.
   *
   * @param column Column index
   *
   * @return (vector<CellIdx>) Cell indices
   */
  vector<CellIdx> cellsForColumn(CellIdx column);

  /**
   * Returns the number of cells in this layer.
   *
   * @return (size_t) Number of cells
   */
  size_t numberOfCells(void) const { return connections.numCells(); }

  /**
   * Returns the indices of the active cells.
   *
   * @returns (std::vector<CellIdx>) Vector of indices of active cells.
   */
  vector<CellIdx> getActiveCells() const; //TODO remove
  void getActiveCells(SDR &activeCells) const;

  /**
   * @return SDR with indices of the predictive cells.
   * SDR dimensions are {TM column dims x TM cells per column}
   */
  SDR getPredictiveCells() const;

  /**
   * Returns the indices of the winner cells.
   *
   * @returns (std::vector<CellIdx>) Vector of indices of winner cells.
   */
  vector<CellIdx> getWinnerCells() const; //TODO remove?
  void getWinnerCells(SDR &winnerCells) const;

  vector<Segment> getActiveSegments() const;
  vector<Segment> getMatchingSegments() const;

  /**
   * Returns the dimensions of the columns in the region.
   *
   * @returns Integer number of column dimension
   */
  vector<CellIdx> getColumnDimensions() const { return columnDimensions_; }

  /**
   * Returns the total number of columns.
   *
   * @returns Integer number of column numbers
   */
  size_t numberOfColumns() const { return numColumns_; }

  /**
   * Returns the number of cells per column.
   *
   * @returns Integer number of cells per column
   */
  size_t getCellsPerColumn() const { return cellsPerColumn_; }

  /**
   * Returns the activation threshold.
   *
   * @returns Integer number of the activation threshold
   */
  SynapseIdx getActivationThreshold() const;
  void setActivationThreshold(const SynapseIdx);

  /**
   * Returns the initial permanence.
   *
   * @returns Initial permanence
   */
  Permanence getInitialPermanence() const;
  void setInitialPermanence(const Permanence);

  /**
   * Returns the connected permanance.
   *
   * @returns Returns the connected permanance
   */
  Permanence getConnectedPermanence() const;

  /**
   * Returns the minimum threshold.
   *
   * @returns Integer number of minimum threshold
   */
  SynapseIdx getMinThreshold() const;
  void setMinThreshold(const SynapseIdx);

  /**
   * Returns the maximum number of synapses that can be added to a segment
   * in a single time step.
   *
   * @returns Integer number of maximum new synapse count
   */
  SynapseIdx getMaxNewSynapseCount() const;
  void setMaxNewSynapseCount(const SynapseIdx);

  /**
   * Get and set the checkInputs parameter.
   */
  bool getCheckInputs() const;
  void setCheckInputs(bool);

  /**
   * Returns the permanence increment.
   *
   * @returns Returns the Permanence increment
   */
  Permanence getPermanenceIncrement() const;
  void setPermanenceIncrement(Permanence);

  /**
   * Returns the permanence decrement.
   *
   * @returns Returns the Permanence decrement
   */
  Permanence getPermanenceDecrement() const;
  void setPermanenceDecrement(Permanence);

  /**
   * Returns the predicted Segment decrement.
   *
   * @returns Returns the segment decrement
   */
  Permanence getPredictedSegmentDecrement() const;
  void setPredictedSegmentDecrement(Permanence);

  /**
   * Returns the maxSegmentsPerCell.
   *
   * @returns Max segments per cell
   */
  SegmentIdx getMaxSegmentsPerCell() const;

  /**
   * Returns the maxSynapsesPerSegment.
   *
   * @returns Max synapses per segment
   */
  SynapseIdx getMaxSynapsesPerSegment() const;

  /**
   * Save (serialize) / Load (deserialize) the current state of the spatial pooler
   * to the specified stream.
   *
   * @param Archive & ar   a Cereal container.
   */
  // a container to hold the data for one sequence item during serialization
  struct container_ar {
    SegmentIdx idx;
    CellIdx cell;
    SynapseIdx syn;

    template<class Archive>
    void save_ar(Archive & ar) const {
      ar(CEREAL_NVP(idx),
         CEREAL_NVP(cell),
         CEREAL_NVP(syn));
    }
    template<class Archive>
    void load_ar(Archive & ar) {
      ar(CEREAL_NVP(idx),
         CEREAL_NVP(cell),
         CEREAL_NVP(syn));
    }
  };

  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const {
    ar(CEREAL_NVP(numColumns_),
       CEREAL_NVP(cellsPerColumn_),
       CEREAL_NVP(activationThreshold_),
       CEREAL_NVP(initialPermanence_),
       CEREAL_NVP(connectedPermanence_),
       CEREAL_NVP(minThreshold_),
       CEREAL_NVP(maxNewSynapseCount_),
       CEREAL_NVP(checkInputs_),
       CEREAL_NVP(permanenceIncrement_),
       CEREAL_NVP(permanenceDecrement_),
       CEREAL_NVP(predictedSegmentDecrement_),
       CEREAL_NVP(externalPredictiveInputs_),
       CEREAL_NVP(maxSegmentsPerCell_),
       CEREAL_NVP(maxSynapsesPerSegment_),
       CEREAL_NVP(rng_),
       CEREAL_NVP(columnDimensions_),
       CEREAL_NVP(activeCells_),
       CEREAL_NVP(winnerCells_),
       CEREAL_NVP(segmentsValid_),
       CEREAL_NVP(anomaly_),
       CEREAL_NVP(connections));

    cereal::size_type numActiveSegments = activeSegments_.size();
    ar( cereal::make_size_tag(numActiveSegments));
    for (Segment segment : activeSegments_) {
      struct container_ar c;
      c.cell = connections.cellForSegment(segment);
      const vector<Segment> &segments = connections.segmentsForCell(c.cell);

      c.idx = (SegmentIdx)std::distance(
                          segments.begin(), 
                          std::find(segments.begin(), 
                          segments.end(), segment));
      c.syn = numActiveConnectedSynapsesForSegment_[segment];
      ar(c); // to keep iteration counts correct, only serialize one item per iteration.
    }

    cereal::size_type numMatchingSegments = matchingSegments_.size();
    ar(cereal::make_size_tag(numMatchingSegments));
    for (Segment segment : matchingSegments_) {
      struct container_ar c;
      c.cell = connections.cellForSegment(segment);
      const vector<Segment> &segments = connections.segmentsForCell(c.cell);

      c.idx = (SegmentIdx)std::distance(
                          segments.begin(), 
                          std::find(segments.begin(), 
                          segments.end(), segment));
      c.syn = numActivePotentialSynapsesForSegment_[segment];
      ar(c);
    }

  }
  template<class Archive>
  void load_ar(Archive & ar) {
    ar(CEREAL_NVP(numColumns_),
       CEREAL_NVP(cellsPerColumn_),
       CEREAL_NVP(activationThreshold_),
       CEREAL_NVP(initialPermanence_),
       CEREAL_NVP(connectedPermanence_),
       CEREAL_NVP(minThreshold_),
       CEREAL_NVP(maxNewSynapseCount_),
       CEREAL_NVP(checkInputs_),
       CEREAL_NVP(permanenceIncrement_),
       CEREAL_NVP(permanenceDecrement_),
       CEREAL_NVP(predictedSegmentDecrement_),
       CEREAL_NVP(externalPredictiveInputs_),
       CEREAL_NVP(maxSegmentsPerCell_),
       CEREAL_NVP(maxSynapsesPerSegment_),
       CEREAL_NVP(rng_),
       CEREAL_NVP(columnDimensions_),
       CEREAL_NVP(activeCells_),
       CEREAL_NVP(winnerCells_),
       CEREAL_NVP(segmentsValid_),
       CEREAL_NVP(anomaly_),
       CEREAL_NVP(connections));

    numActiveConnectedSynapsesForSegment_.assign(connections.segmentFlatListLength(), 0);
    cereal::size_type numActiveSegments;
    ar(cereal::make_size_tag(numActiveSegments));
    activeSegments_.resize(static_cast<size_t>(numActiveSegments));
    for (size_t i = 0; i < static_cast<size_t>(numActiveSegments); i++) {
      struct container_ar c;
      ar(c);  
      Segment segment = connections.getSegment(c.cell, c.idx);
      activeSegments_[i] = segment;
      numActiveConnectedSynapsesForSegment_[segment] = c.syn;
    }

    numActivePotentialSynapsesForSegment_.assign(connections.segmentFlatListLength(), 0);
    cereal::size_type numMatchingSegments;
    ar(cereal::make_size_tag(numMatchingSegments));
    matchingSegments_.resize(static_cast<size_t>(numMatchingSegments));
    for (size_t i = 0; i < static_cast<size_t>(numMatchingSegments); i++) {
      struct container_ar c;
      ar(c);
      Segment segment = connections.getSegment(c.cell, c.idx);
      matchingSegments_[i] = segment;
      numActivePotentialSynapsesForSegment_[segment] = c.syn;
    }
  }


  virtual bool operator==(const TemporalMemory &other) const;
  inline bool operator!=(const TemporalMemory &other) const { return not this->operator==(other); }

  //----------------------------------------------------------------------
  // Debugging helpers
  //----------------------------------------------------------------------

  /**
   * Print diagnostic info
   */
  friend std::ostream& operator<< (std::ostream& stream, const TemporalMemory& self);

  /**
   * Print the main TM creation parameters
   */
  void printParameters(std::ostream& out=std::cout) const;

  /**
   * Returns the index of the (mini-)column that a cell belongs to.
   * 
   * Mini columns are an organizational unit in TM, 
   * each mini column consists for cellsPerColumns cells. 
   * There's no topology between cells within a mini-column, cells
   * are organized as a flat array 
   * `col{i} = [cell{i*CPS}, cell{i*CPS +1}, ..., cell{i*CPS + CPS-1}], 
   * where CPS stands for cellsPerColumn`
   *
   * @param cell Cell index
   *
   * @return (int) Column index
   */
  UInt columnForCell(const CellIdx cell) const;

  /**
   *  cellsToColumns
   *  converts active cells to columnar representation, 
   *  see columnForCell() for details.
   *
   *  @param const SDR& cells - input cells, size must be a multiple of cellsPerColumn; ie. 
   *    all SDRs obtained from TM's get*Cells(SDR) are valid. 
   *    The SDR cells dimensions must be: {TM.getColumnDimensions, TM.getCellsPerColumn()}
   *
   *  @return SDR cols - which is size of TM's getColumnDimensions()
   *
   */
  SDR cellsToColumns(const SDR& cells) const;

protected:
  //all these could be const
  CellIdx numColumns_;
  vector<CellIdx> columnDimensions_;
  CellIdx cellsPerColumn_;
  SynapseIdx activationThreshold_;
  SynapseIdx minThreshold_;
  SynapseIdx maxNewSynapseCount_;
  bool checkInputs_;
  Permanence initialPermanence_;
  Permanence connectedPermanence_;
  Permanence permanenceIncrement_;
  Permanence permanenceDecrement_;
  Permanence predictedSegmentDecrement_;
  UInt externalPredictiveInputs_;
  SegmentIdx maxSegmentsPerCell_;
  SynapseIdx maxSynapsesPerSegment_;

private:
  vector<CellIdx> activeCells_;
  vector<CellIdx> winnerCells_;
  bool segmentsValid_;
  vector<Segment> activeSegments_;
  vector<Segment> matchingSegments_;
  vector<SynapseIdx> numActiveConnectedSynapsesForSegment_;
  vector<SynapseIdx> numActivePotentialSynapsesForSegment_;

  Real anomaly_;

  Random rng_;

public:
  Connections connections;
  const UInt &externalPredictiveInputs = externalPredictiveInputs_;
  /*
   *  anomaly score computed for the current inputs
   *  (auto-updates after each call to TM::compute())
   *
   *  @return a float value from computeRawAnomalyScore()
   *  from Anomaly.hpp
   */
  const Real &anomaly = anomaly_;
};

} // namespace htm

#endif // NTA_TEMPORAL_MEMORY_HPP
