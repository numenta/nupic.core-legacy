/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * Definitions for the Spatial Pooler in C++
 */

#ifndef NTA_spatial_pooler_HPP
#define NTA_spatial_pooler_HPP

#include <iostream>
#include <vector>
#include <iomanip> // std::setprecision
#include <htm/algorithms/Connections.hpp>
#include <htm/types/Types.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/types/Sdr.hpp>


namespace htm {

using namespace std;

/**
 * CLA spatial pooler implementation in C++.
 *
 * ### Description
 * The Spatial Pooler is responsible for creating a sparse distributed
 * representation of the input. Given an input it computes a set of sparse
 * active columns and simultaneously updates its permanences, duty cycles,
 * etc.
 *
 * The primary public interfaces to this function are the "initialize"
 * and "compute" methods.
 *
 * Example usage:
 *
 *     SpatialPooler sp;
 *     sp.initialize(inputDimensions, columnDimensions, <parameters>);
 *     while (true) {
 *        <get input vector>
 *        sp.compute(inputVector, learn, activeColumns)
 *        <do something with output>
 *     }
 *
 */
class SpatialPooler : public Serializable
{
public:

  const Real MAX_LOCALAREADENSITY = 0.5f; //require atleast 2 areas

  SpatialPooler();
  SpatialPooler(const vector<UInt> inputDimensions, const vector<UInt> columnDimensions,
                UInt potentialRadius = 16u, Real potentialPct = 0.5f,
                bool globalInhibition = true, 
		Real localAreaDensity = 0.05f, //5%
                UInt stimulusThreshold = 0u, 
		Real synPermInactiveDec = 0.008f,
                Real synPermActiveInc = 0.05f, 
		Real synPermConnected = 0.1f,
                Real minPctOverlapDutyCycles = 0.001f,
                UInt dutyCyclePeriod = 1000u, 
		Real boostStrength = 0.0f,
                Int seed = 1, 
		UInt spVerbosity = 0u, 
		bool wrapAround = true);

  virtual ~SpatialPooler() {}

  // equals operators
  virtual bool operator==(const SpatialPooler& o) const;
  inline bool operator!=(const SpatialPooler& o) const { return !this->operator==(o); }
  inline bool equals(const SpatialPooler& o) const { return this->operator==(o); } //equals is for PY


  /**
  Initialize the spatial pooler using the given parameters.

  @param inputDimensions A list of integers representing the
        dimensions of the input vector. Format is [height, width,
        depth, ...], where each value represents the size of the
        dimension. For a topology of one dimesion with 100 inputs
        use [100]. For a two dimensional topology of 10x5
        use [10,5].

  @param columnDimensions A list of integers representing the
        dimensions of the columns in the region. Format is [height,
        width, depth, ...], where each value represents the size of
        the dimension. For a topology of one dimesion with 2000
        columns use 2000, or [2000]. For a three dimensional
        topology of 32x64x16 use [32, 64, 16].

  @param potentialRadius This parameter deteremines the extent of the
        input that each column can potentially be connected to. This
        can be thought of as the input bits that are visible to each
        column, or a 'receptive field' of the field of vision. A large
        enough value will result in global coverage, meaning
        that each column can potentially be connected to every input
        bit. This parameter defines a square (or hyper square) area: a
        column will have a max square potential pool with sides of
        length `(2 * potentialRadius + 1)`, rounded to fit into each dimension.

  @param potentialPct The percent of the inputs, within a column's
        potential radius, that a column can be connected to. If set to
        1, the column will be connected to every input within its
        potential radius. This parameter is used to give each column a
        unique potential pool when a large potentialRadius causes
        overlap between the columns. At initialization time we choose
        ((2*potentialRadius + 1)^(# inputDimensions) * potentialPct)
        input bits to comprise the column's potential pool.

  @param globalInhibition If true, then during inhibition phase the
        winning columns are selected as the most active columns from the
        region as a whole. Otherwise, the winning columns are selected
        with resepct to their local neighborhoods. Global inhibition
        boosts performance significantly but there is no topology at the
        output.

  @param localAreaDensity The desired density of active columns within
        a local inhibition area (the size of which is set by the
        internally calculated inhibitionRadius, which is in turn
        determined from the average size of the connected potential
        pools of all columns). The inhibition logic will insure that at
        most N columns remain ON within a local inhibition area, where
        N = localAreaDensity * (total number of columns in inhibition
        area)
        Default: 0.05 (5%)	

  @param stimulusThreshold This is a number specifying the minimum
        number of synapses that must be active in order for a column to
        turn ON. The purpose of this is to prevent noisy input from
        activating columns.

  @param synPermInactiveDec The amount by which the permanence of an
        inactive synapse is decremented in each learning step.

  @param synPermActiveInc The amount by which the permanence of an
        active synapse is incremented in each round.

  @param synPermConnected The default connected threshold. Any synapse
        whose permanence value is above the connected threshold is
        a "connected synapse", meaning it can contribute to
        the cell's firing.

  @param minPctOverlapDutyCycles A number between 0 and 1.0, used to set
        a floor on how often a column should have at least
        stimulusThreshold active inputs. Periodically, each column looks
        at the overlap duty cycle of all other column within its
        inhibition radius and sets its own internal minimal acceptable
        duty cycle to: minPctDutyCycleBeforeInh * max(other columns'
        duty cycles). On each iteration, any column whose overlap duty
        cycle falls below this computed value will get all of its
        permanence values boosted up by synPermActiveInc. Raising all
        permanences in response to a sub-par duty cycle before
        inhibition allows a cell to search for new inputs when either
        its previously learned inputs are no longer ever active, or when
        the vast majority of them have been "hijacked" by other columns.

  @param dutyCyclePeriod The period used to calculate duty cycles.
        Higher values make it take longer to respond to changes in
        boost. Shorter values make it potentially more unstable and
        likely to oscillate.

  @param boostStrength A number greater or equal than 0, used to
        control boosting strength. 
	No boosting is applied if it is set to 0.0, (runs faster due to skipped code).
        The strength of boosting increases as a function of boostStrength.
        Boosting encourages columns to have similar activeDutyCycles as their
        neighbors, which will lead to more efficient use of columns. However,
        too much boosting may also lead to instability of SP outputs.


  @param seed Seed for our random number generator. If seed is < 0
        a randomly generated seed is used. The behavior of the spatial
        pooler is deterministic once the seed is set.

  @param spVerbosity spVerbosity level: 0, 1, 2, or 3

  @param wrapAround boolean value that determines whether or not inputs
        at the beginning and end of an input dimension are considered
        neighbors for the purpose of mapping inputs to columns.

   */
  virtual void
  initialize(const vector<UInt>& inputDimensions, 
	     const vector<UInt>& columnDimensions,
             UInt potentialRadius = 16u, 
	     Real potentialPct = 0.5f,
             bool globalInhibition = true, 
	     Real localAreaDensity = 0.05f,
             UInt stimulusThreshold = 0u,
             Real synPermInactiveDec = 0.01f, Real synPermActiveInc = 0.1f,
             Real synPermConnected = 0.1f, Real minPctOverlapDutyCycles = 0.001f,
             UInt dutyCyclePeriod = 1000u, Real boostStrength = 0.0f,
             Int seed = 1, UInt spVerbosity = 0u, bool wrapAround = true);


  /**
  This is the main workshorse method of the SpatialPooler class. This
  method takes an input SDR and computes the set of output active
  columns. If 'learn' is set to True, this method also performs
  learning.

  @param input An SDR that comprises the input to the spatial pooler.  The size
        of the SDR must mach total number of input bits implied by the
        constructor (also returned by the method getNumInputs).

  @param learn A boolean value indicating whether learning should be
        performed. Learning entails updating the permanence values of
        the synapses, duty cycles, etc. Learning is typically on but
        setting learning to 'off' is useful for analyzing the current
        state of the SP. For example, you might want to feed in various
        inputs and examine the resulting SDR's. Note that if learning
        is off, boosting is turned off and columns that have never won
        will be removed from activeVector.  TODO: we may want to keep
        boosting on even when learning is off.

  @param active An SDR representing the winning columns after
        inhibition. The size of the SDR is equal to the number of
        columns (also returned by the method getNumColumns).
   */
  virtual void compute(const SDR &input, const bool learn, SDR &active);


  /**
   * Get the version number of this spatial pooler.

   * @returns Integer version number.
   */
  virtual UInt version() const { return version_; };

  /**
  save_ar()/load_ar() Serialize the current state of the spatial pooler to the
  specified file and deserialize it.

  @param Archive& ar  See Serializable.hpp
   */
  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    ar(CEREAL_NVP(inputDimensions_),
       CEREAL_NVP(columnDimensions_));
    ar(CEREAL_NVP(numInputs_),
       CEREAL_NVP(numColumns_),
       CEREAL_NVP(potentialRadius_),
       CEREAL_NVP(potentialPct_),
       CEREAL_NVP(initConnectedPct_),
       CEREAL_NVP(globalInhibition_),
       CEREAL_NVP(localAreaDensity_),
       CEREAL_NVP(stimulusThreshold_),
       CEREAL_NVP(inhibitionRadius_),
       CEREAL_NVP(dutyCyclePeriod_),
       CEREAL_NVP(boostStrength_),
       CEREAL_NVP(iterationNum_),
       CEREAL_NVP(iterationLearnNum_),
       CEREAL_NVP(spVerbosity_),
       CEREAL_NVP(updatePeriod_),
       CEREAL_NVP(synPermInactiveDec_),
       CEREAL_NVP(synPermActiveInc_),
       CEREAL_NVP(synPermBelowStimulusInc_),
       CEREAL_NVP(synPermConnected_),
       CEREAL_NVP(minPctOverlapDutyCycles_),
       CEREAL_NVP(wrapAround_));
    ar(CEREAL_NVP(boostFactors_));
    ar(CEREAL_NVP(overlapDutyCycles_));
    ar(CEREAL_NVP(activeDutyCycles_));
    ar(CEREAL_NVP(minOverlapDutyCycles_));
    ar(CEREAL_NVP(connections_));
    ar(CEREAL_NVP(rng_));
  }
  // FOR Cereal Deserialization
  template<class Archive>
  void load_ar(Archive& ar) {
    ar(CEREAL_NVP(inputDimensions_),
       CEREAL_NVP(columnDimensions_));
    ar(CEREAL_NVP(numInputs_),
       CEREAL_NVP(numColumns_),
       CEREAL_NVP(potentialRadius_),
       CEREAL_NVP(potentialPct_),
       CEREAL_NVP(initConnectedPct_),
       CEREAL_NVP(globalInhibition_),
       CEREAL_NVP(localAreaDensity_),
       CEREAL_NVP(stimulusThreshold_),
       CEREAL_NVP(inhibitionRadius_),
       CEREAL_NVP(dutyCyclePeriod_),
       CEREAL_NVP(boostStrength_),
       CEREAL_NVP(iterationNum_),
       CEREAL_NVP(iterationLearnNum_),
       CEREAL_NVP(spVerbosity_),
       CEREAL_NVP(updatePeriod_),
       CEREAL_NVP(synPermInactiveDec_),
       CEREAL_NVP(synPermActiveInc_),
       CEREAL_NVP(synPermBelowStimulusInc_),
       CEREAL_NVP(synPermConnected_),
       CEREAL_NVP(minPctOverlapDutyCycles_),
       CEREAL_NVP(wrapAround_));
    ar(CEREAL_NVP(boostFactors_));
    ar(CEREAL_NVP(overlapDutyCycles_));
    ar(CEREAL_NVP(activeDutyCycles_));
    ar(CEREAL_NVP(minOverlapDutyCycles_));
    ar(CEREAL_NVP(connections_));
    ar(CEREAL_NVP(rng_));

    // initialize ephemeral members
    overlaps_.resize(numColumns_);
    boostedOverlaps_.resize(numColumns_);
  }

  /**
  Returns the dimensions of the columns in the region.

  @returns Integer number of column dimension.
  */
  vector<UInt> getColumnDimensions() const;

  /**
  Returns the dimensions of the input vector.

  @returns Integer vector of input dimension.
  */
  vector<UInt> getInputDimensions() const;

  /**
  Returns the total number of columns.

  @returns Integer number of column numbers.
  */
  UInt getNumColumns() const;

  /**
  Returns the total number of inputs.

  @returns Integer number of inputs.
  */
  UInt getNumInputs() const;

  /**
  Returns the potential radius.

  @returns Integer number of potential radius.
  */
  UInt getPotentialRadius() const;

  /**
  Sets the potential radius.

  @param potentialRadius integer number of potential raduis.
  */
  void setPotentialRadius(UInt potentialRadius);
  /**
  Returns the potential percent.

  @returns real number of the potential percent.
  */
  Real getPotentialPct() const;

  /**
  Sets the potential percent.

  @param potentialPct real number of potential percent.
  */
  void setPotentialPct(Real potentialPct);

  /**
  @returns boolen value of whether global inhibition is enabled.
  */
  bool getGlobalInhibition() const;

  /**
  Sets global inhibition.

  @param globalInhibition boolen varable of whether global inhibition is
  enabled.
  */
  void setGlobalInhibition(bool globalInhibition);

  /**
  Returns the local area density. Returns a value less than 0 if parameter
  is unused.

  @returns real number of local area density.
  */
  Real getLocalAreaDensity() const;

  /**
  Sets the local area density. Invalidates the 'numActivePerInhArea'
  parameter.
  @param localAreaDensity real number of local area density.
  */
  void setLocalAreaDensity(Real localAreaDensity);

  /**
  Returns the stimulus threshold.

  @returns integer number of stimulus threshold.
  */
  UInt getStimulusThreshold() const;

  /**
  Sets the stimulus threshold.

  @param stimulusThreshold (positive) integer number of stimulus threshold
  */
  void setStimulusThreshold(UInt stimulusThreshold);

  /**
  Returns the inhibition radius.

  @returns (positive) integer of inhibition radius/
  */
  UInt getInhibitionRadius() const;
  /**
  Sets the inhibition radius.

  @param inhibitionRadius integer of inhibition radius.
  */
  void setInhibitionRadius(UInt inhibitionRadius);

  /**
  Returns the duty cycle period.

  @returns integer of duty cycle period.
  */
  UInt getDutyCyclePeriod() const;

  /**
  Sets the duty cycle period.

  @param dutyCyclePeriod integer number of duty cycle period.
  */
  void setDutyCyclePeriod(UInt dutyCyclePeriod);

  /**
  Returns the maximum boost value.

  @returns real number of the maximum boost value.
  */
  Real getBoostStrength() const;

  /**
  Sets the strength of boost.

  @param boostStrength real number of boosting strength,
  must be larger than 0.0
  */
  void setBoostStrength(Real boostStrength);

  /**
  Returns the iteration number.

  @returns integer number of iteration number.
  */
  UInt getIterationNum() const;

  /**
  Sets the iteration number.

  @param iterationNum integer number of iteration number.
  */
  void setIterationNum(UInt iterationNum);

  /**
  Returns the learning iteration number.

  @returns integer of the learning iteration number.
  */
  UInt getIterationLearnNum() const;

  /**
  Sets the learning iteration number.

  @param iterationLearnNum integer of learning iteration number.
  */
  void setIterationLearnNum(UInt iterationLearnNum);

  /**
  Returns the verbosity level.

  @returns integer of the verbosity level.
  */
  UInt getSpVerbosity() const;

  /**
  Sets the verbosity level.

  @param spVerbosity integer of verbosity level.
  */
  void setSpVerbosity(UInt spVerbosity);

  /**
  Returns boolean value of wrapAround which indicates if receptive
  fields should wrap around from the beginning the input dimensions
  to the end.

  @returns the boolean value of wrapAround.
  */
  bool getWrapAround() const;

  /**
  Sets wrapAround.

  @param wrapAround boolean value
  */
  void setWrapAround(bool wrapAround);

  /**
  Returns the update period.

  @returns integer of update period.
  */
  UInt getUpdatePeriod() const;
  /**
  Sets the update period.

  @param updatePeriod integer of update period.
  */
  void setUpdatePeriod(UInt updatePeriod);

  /**
  Returns the permanence increment amount for active synapses
  inputs.

  @returns real number of the permanence increment amount for active synapses
  inputs.
  */
  Real getSynPermActiveInc() const;
  /**
  Sets the permanence increment amount for active synapses
  inputs.

  @param synPermActiveInc real number of the permanence increment amount
  for active synapses inputs, must be >0.
  */
  void setSynPermActiveInc(Real synPermActiveInc);

  /**
  Returns the permanence decrement amount for inactive synapses.

  @returns real number of the permanence decrement amount for inactive synapses.
  */
  Real getSynPermInactiveDec() const;
  /**
  Returns the permanence decrement amount for inactive synapses.

  @param synPermInactiveDec real number of the permanence decrement amount for
  inactive synapses.
  */
  void setSynPermInactiveDec(Real synPermInactiveDec);

  /**
  Returns the permanence increment amount for columns that have not been
  recently active.

  @returns positive real number of the permanence increment amount for columns
  that have not been recently active.
  */
  Real getSynPermBelowStimulusInc() const;
  /**
  Sets the permanence increment amount for columns that have not been
  recently active.

  @param synPermBelowStimulusInc real number of the permanence increment amount
  for columns that have not been recently active, must be larger than 0.
  */
  void setSynPermBelowStimulusInc(Real synPermBelowStimulusInc);

  /**
  Returns the permanence amount that qualifies a synapse as
  being connected.

  @returns real number of the permanence amount
  that qualifies a synapse as being connected.
  */
  Real getSynPermConnected() const;

  /**
  Returns the maximum permanence amount a synapse can
  achieve.

  @returns real number of the max permanence amount.
  */
  Real getSynPermMax() const;

  /**
  Returns the minimum tolerated overlaps, given as percent of
  neighbors overlap score.

  @returns real number of the minimum tolerated overlaps.
  */
  Real getMinPctOverlapDutyCycles() const;
  /**
  Sets the minimum tolerated overlaps, given as percent of
  neighbors overlap score.

  @param minPctOverlapDutyCycles real number of the minimum tolerated overlaps.
  */
  void setMinPctOverlapDutyCycles(Real minPctOverlapDutyCycles);

  /**
  Returns the boost factors for all columns. 'boostFactors' size must
  match the number of columns.

  @param boostFactors real array to store boost factors of all columns.
  */
  void getBoostFactors(Real boostFactors[]) const;
  /**
  Sets the boost factors for all columns. 'boostFactors' size must
  match the number of columns.

  @param boostFactors real array of boost factors of all columns.
  */
  void setBoostFactors(Real boostFactors[]);

  /**
  Returns the overlap duty cycles for all columns. 'overlapDutyCycles'
  size must match the number of columns.

  @param overlapDutyCycles real array to store overlap duty cycles for all
  columns.
  */
  void getOverlapDutyCycles(Real overlapDutyCycles[]) const;
  /**
  Sets the overlap duty cycles for all columns. 'overlapDutyCycles'
  size must match the number of columns.

  @param overlapDutyCycles real array of the overlap duty cycles for all
  columns.
  */
  void setOverlapDutyCycles(const Real overlapDutyCycles[]);

  /**
  Returns the activity duty cycles for all columns. 'activeDutyCycles'
  size must match the number of columns.

  @param activeDutyCycles real array to store activity duty cycles for all
  columns.
  */
  void getActiveDutyCycles(Real activeDutyCycles[]) const;
  /**
  Sets the activity duty cycles for all columns. 'activeDutyCycles'
  size must match the number of columns.

  @param activeDutyCycles real array of the activity duty cycles for all
  columns.
  */
  void setActiveDutyCycles(const Real activeDutyCycles[]);

  /**
  Returns the minimum overlap duty cycles for all columns.

  @param minOverlapDutyCycles real arry to store mininum overlap duty cycles for
  all columns. 'minOverlapDutyCycles' size must match the number of columns.
  */
  void getMinOverlapDutyCycles(Real minOverlapDutyCycles[]) const;
  /**
  Sets the minimum overlap duty cycles for all columns.
  '_minOverlapDutyCycles' size must match the number of columns.

  @param minOverlapDutyCycles real array of the minimum overlap duty cycles for
  all columns.
  */
  void setMinOverlapDutyCycles(const Real minOverlapDutyCycles[]);

  /**
  Returns the potential mapping for a given column. 'potential' size
  must match the number of inputs.

  @param column integer of column index.

  @param potential integer array of potential mapping for the selected column.
  */
  void getPotential(UInt column, UInt potential[]) const;
  /**
  Sets the potential mapping for a given column. 'potential' size
  must match the number of inputs.

  @param column integer of column index.

  @param potential integer array of potential mapping for the selected column.
  */
  void setPotential(UInt column, const UInt potential[]);

  /**
  Returns the permanence values for a given column. 'permanence' size
  must match the number of inputs.

  @param column integer of column index.

  @param permanence real array to store permanence values for the selected
  column.
  */
  void getPermanence(UInt column, Real permanence[]) const;
  /**
  Sets the permanence values for a given column. 'permanence' size
  must match the number of inputs.

  @param column integer of column index.

  @param permanence real array of permanence values for the selected column.
  */
  void setPermanence(UInt column, const Real permanence[]);

  /**
  Returns the connected synapses for a given column.
  'connectedSynapses' size must match the number of inputs.

  @param column integer of column index.

  @param connectedSynapses integer array to store the connected synapses for a
  given column.
  */
  void getConnectedSynapses(UInt column, UInt connectedSynapses[]) const;

  /**
  Returns the number of connected synapses for all columns.
  'connectedCounts' size must match the number of columns.

  @param connectedCounts integer array to store the connected synapses for all
  columns.
  */
  void getConnectedCounts(UInt connectedCounts[]) const;


  /**
  Returns the overlap score for each column.
   */
  const std::vector<SynapseIdx> &getOverlaps() const;

  /**
  Returns the boosted overlap score for each column.
   */
  const vector<Real> &getBoostedOverlaps() const;

  ///////////////////////////////////////////////////////////
  //
  // Implementation methods. all methods below this line are
  // NOT part of the public API


  void boostOverlaps_(const vector<SynapseIdx> &overlaps, vector<Real> &boostedOverlaps) const;

  /**
    Maps a column to its respective input index, keeping to the topology of
    the region. It takes the index of the column as an argument and determines
    what is the index of the flattened input vector that is to be the center of
    the column's potential pool. It distributes the columns over the inputs
    uniformly. The return value is an integer representing the index of the
    input bit. Examples of the expected output of this method:
    * If the topology is one dimensional, and the column index is 0, this
      method will return the input index 0. If the column index is 1, and there
      are 3 columns over 7 inputs, this method will return the input index 3.
    * If the topology is two dimensional, with column dimensions [3, 5] and
      input dimensions [7, 11], and the column index is 3, the method
      returns input index 8.

    ----------------------------
    @param index       The index identifying a column in the permanence,
    potential and connectivity matrices.
    @param wrapAround  A boolean value indicating that boundaries should be
                       ignored.
    Used only during initialization.
  */
  UInt initMapColumn_(UInt column) const;

  /**
    Maps a column to its input bits.

    This method encapsultes the topology of
    the region. It takes the index of the column as an argument and determines
    what are the indices of the input vector that are located within the
    column's potential pool. The return value is a list containing the indices
    of the input bits. The current implementation of the base class only
    supports a 1 dimensional topology of columns with a 1 dimensional topology
    of inputs. To extend this class to support 2-D topology you will need to
    override this method. Examples of the expected output of this method:
    * If the potentialRadius is greater than or equal to the entire input
      space, (global visibility), then this method returns an array filled with
      all the indices
    * If the topology is one dimensional, and the potentialRadius is 5, this
      method will return an array containing 5 consecutive values centered on
      the index of the column (wrapping around if necessary).
    * If the topology is two dimensional (not implemented), and the
      potentialRadius is 5, the method should return an array containing 25
      '1's, where the exact indices are to be determined by the mapping from
      1-D index to 2-D position.
    Used only at initialization.

    ----------------------------
    @param column         An int index identifying a column in the permanence,
    potential and connectivity matrices.

    @param wrapAround  A boolean value indicating that boundaries should be
                       ignored.
  */
  vector<UInt> initMapPotential_(UInt column, bool wrapAround);

  /**
  Returns a randomly generated permanence value for a synapses that is
  initialized in a connected state.

  The basic idea here is to initialize
  permanence values very close to synPermConnected so that a small number of
  learning steps could make it disconnected or connected.

  Note: experimentation was done a long time ago on the best way to initialize
  permanence values, but the history for this particular scheme has been lost.

  @returns real number of a randomly generated permanence value for a synapses
  that is initialized in a connected state.
  */
  Real initPermConnected_();
  /**
      Returns a randomly generated permanence value for a synapses that is to be
      initialized in a non-connected state.

      @returns real number of a randomly generated permanence value for a
     synapses that is to be initialized in a non-connected state.
  */
  Real initPermNonConnected_();

  /**
    Initializes the permanences of a column. The method
    returns a 1-D array the size of the input, where each entry in the
    array represents the initial permanence value between the input bit
    at the particular index in the array, and the column represented by
    the 'index' parameter.

    @param potential      A int vector specifying the potential pool of the
    column. Permanence values will only be generated for input bits

                    corresponding to indices for which the mask value is 1.
    @param connectedPct   A real value between 0 or 1 specifying the percent of
    the input bits that will start off in a connected state.
  */
  vector<Real> initPermanence_(const vector<UInt> &potential, Real connectedPct);

  void clip_(vector<Real> &perm) const;

  void raisePermanencesToThreshold_(vector<Real> &perm,
                                    const vector<UInt> &potential) const;

  /**
     This function determines each column's overlap with the current
     input vector.

     The overlap of a column is the number of synapses for that column
     that are connected (permanence value is greater than
     '_synPermConnected') to input bits which are turned on. The
     implementation takes advantage of the SparseBinaryMatrix class to
     perform this calculation efficiently.

     @param inputVector
     a int array of 0's and 1's that comprises the input to the spatial
     pooler.

     @param overlap
     an int vector containing the overlap score for each column. The
     overlap score for a column is defined as the number of synapses in
     a "connected state" (connected synapses) that are connected to
     input bits which are turned on.
  */
  void calculateOverlap_(const SDR &input, vector<SynapseIdx> &overlap, const bool learn = true);
  void calculateOverlapPct_(const vector<SynapseIdx> &overlaps, vector<Real> &overlapPct) const;

  /**
      Performs inhibition. This method calculates the necessary values needed to
      actually perform inhibition and then delegates the task of picking the
      active columns to helper functions.


      @param overlaps       an array containing the overlap score for each
     column. The overlap score for a column is defined as the number of synapses
     in a "connected state" (connected synapses) that are connected to input
     bits which are turned on.

      @param activeColumns an int array containing the indices of the active
     columns.
  */
  void inhibitColumns_(const vector<Real> &overlaps,
                       vector<CellIdx> &activeColumns) const;

  /**
     Perform global inhibition.

     Performing global inhibition entails picking the top 'numActive'
     columns with the highest overlap score in the entire region. At
     most half of the columns in a local neighborhood are allowed to be
     active. Columns with an overlap score below the 'stimulusThreshold'
     are always inhibited.

     @param overlaps
     a real array containing the overlap score for each column. The
     overlap score for a column is defined as the number of synapses in
     a "connected state" (connected synapses) that are connected to
     input bits which are turned on.

     @param density
     a real number of the fraction of columns to survive inhibition.

     @param activeColumns
     an int array containing the indices of the active columns.
  */
  void inhibitColumnsGlobal_(const vector<Real> &overlaps, Real density,
                             vector<UInt> &activeColumns) const;

  /**
     Performs local inhibition.

     Local inhibition is performed on a column by column basis. Each
     column observes the overlaps of its neighbors and is selected if
     its overlap score is within the top 'numActive' in its local
     neighborhood. At most half of the columns in a local neighborhood
     are allowed to be active. Columns with an overlap score below the
     'stimulusThreshold' are always inhibited.

     ----------------------------
     @param overlaps
     an array containing the overlap score for each column. The overlap
     score for a column is defined as the number of synapses in a
     "connected state" (connected synapses) that are connected to input
     bits which are turned on.

     @param density
     The fraction of columns to survive inhibition. This value is only
     an intended target. Since the surviving columns are picked in a
     local fashion, the exact fraction of surviving columns is likely to
     vary.

     @param activeColumns
     an int array containing the indices of the active columns.
  */
  void inhibitColumnsLocal_(const vector<Real> &overlaps, Real density,
                            vector<UInt> &activeColumns) const;

  /**
      The primary method in charge of learning.

      Adapts the permanence values of
      the synapses based on the input vector, and the chosen columns after
      inhibition round. Permanence values are increased for synapses connected
     to input bits that are turned on, and decreased for synapses connected to
      inputs bits that are turned off.

      ----------------------------
      @param inputVector    an int array of 0's and 1's that comprises the input
     to the spatial pooler. There exists an entry in the array for every input
     bit.

      @param  activeColumns  an int vector containing the indices of the columns
     that survived inhibition.
   */
  void adaptSynapses_(const SDR &input, const SDR &active);

  /**
      This method increases the permanence values of synapses of columns whose
      activity level has been too low. Such columns are identified by having an
      overlap duty cycle that drops too much below those of their peers. The
      permanence values for such columns are increased.
  */
  void bumpUpWeakColumns_();

  /**
      Update the inhibition radius. The inhibition radius is a meausre of the
      square (or hypersquare) of columns that each a column is "connected to"
      on average. Since columns are not connected to each other directly, we
      determine this quantity by first figuring out how many *inputs* a column
     is connected to, and then multiplying it by the total number of columns
     that exist for each input. For multiple dimension the aforementioned
      calculations are averaged over all dimensions of inputs and columns. This
      value is meaningless if global inhibition is enabled.
  */
  void updateInhibitionRadius_();

  /**
      REturns the average number of columns per input, taking into account the
     topology of the inputs and columns. This value is used to calculate the
     inhibition radius. This function supports an arbitrary number of
     dimensions. If the number of column dimensions does not match the number of
     input dimensions, we treat the missing, or phantom dimensions as 'ones'.

      @returns real number of the average number of columns per input.
  */
  Real avgColumnsPerInput_() const;

  /**
      The range of connectedSynapses per column, averaged for each dimension.
      This vaule is used to calculate the inhibition radius. This variation of
      the function supports arbitrary column dimensions.

      @param column An int number identifying a column in the permanence,
     potential and connectivity matrices.
  */
  Real avgConnectedSpanForColumnND_(UInt column) const;

  /**
      Updates the minimum duty cycles defining normal activity for a column. A
      column with activity duty cycle below this minimum threshold is boosted.
  */
  void updateMinDutyCycles_();

  /**
      Updates the minimum duty cycles in a global fashion. Sets the minimum duty
      cycles for the overlap and activation of all columns to be a percent of
     the maximum in the region, specified by minPctOverlapDutyCycle and
      minPctActiveDutyCycle respectively. Functionally it is equivalent to
      _updateMinDutyCyclesLocal, but this function exploits the globalilty of
     the computation to perform it in a straightforward, and more efficient
     manner.
  */
  void updateMinDutyCyclesGlobal_();

  /**
  Updates the minimum duty cycles. The minimum duty cycles are determined
  locally. Each column's minimum duty cycles are set to be a percent of the
  maximum duty cycles in the column's neighborhood. Unlike
  _updateMinDutyCycles
  */
  void updateMinDutyCyclesLocal_();

  /**
      Updates a duty cycle estimate with a new value. This is a helper
      function that is used to update several duty cycle variables in
      the Column class, such as: overlapDutyCucle, activeDutyCycle,
      minPctDutyCycleBeforeInh, minPctDutyCycleAfterInh, etc. returns
      the updated duty cycle. Duty cycles are updated according to the following
      formula:
      @verbatim


                    (period - 1)*dutyCycle + newValue
        dutyCycle := ----------------------------------
                                period
      @endverbatim

      ----------------------------
      @param dutyCycles     A real array containing one or more duty cycle
     values that need to be updated.

      @param newValues      A int vector used to update the duty cycle.

      @param period         A int number indicating the period of the duty cycle

      @return type void, the argument dutyCycles is updated with new values.
  */
  static void updateDutyCyclesHelper_(vector<Real> &dutyCycles,
                                      const SDR &newValues, 
                                      const UInt period);

  /**
  Updates the duty cycles for each column. The OVERLAP duty cycle is a moving
  average of the number of inputs which overlapped with the each column. The
  ACTIVITY duty cycles is a moving average of the frequency of activation for
  each column.

  @param overlaps       an int vector containing the overlap score for each
  column. The overlap score for a column is defined as the number of synapses in
  a "connected state" (connected synapses) that are connected to input bits
  which are turned on.

  @param activeArray  An int array containing the indices of the active columns,
                  the sprase set of columns which survived inhibition
  */
  void updateDutyCycles_(const vector<SynapseIdx> &overlaps, SDR &active);

  /**
    Update the boost factors for all columns. The boost factors are used to
    increase the overlap of inactive columns to improve their chances of
    becoming active, and hence encourage participation of more columns in the
    learning process. The boosting function is a curve defined as:
    boostFactors = exp[ - boostStrength * (dutyCycle - targetDensity)]
    Intuitively this means that columns that have been active at the target
    activation level have a boost factor of 1, meaning their overlap is not
    boosted. Columns whose active duty cycle drops too much below that of their
    neighbors are boosted depending on how infrequently they have been active.
    Columns that has been active more than the target activation level have
    a boost factor below 1, meaning their overlap is suppressed

    The boostFactor depends on the activeDutyCycle via an exponential function:

                boostFactor
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> activeDutyCycle
                      |
                targetDensity
      @endverbatim
    */
  void updateBoostFactors_();

  /**
  Update boost factors when local inhibition is enabled. In this case,
  the target activation level for each column is estimated as the
  average activation level for columns in its neighborhood.
  */
  void updateBoostFactorsLocal_();

  /**
  Update boost factors when global inhibition is enabled. All columns
  share the same target activation level in this case, which is the
  sparsity of spatial pooler.
  */
  void updateBoostFactorsGlobal_();

  /**
  Updates counter instance variables each round.

    @param learn          a boolean value indicating whether learning should be
                    performed. Learning entails updating the  permanence
                    values of the synapses, and hence modifying the 'state'
                    of the model. setting learning to 'off' might be useful
                    for indicating separate training vs. testing sets.
   */
  void updateBookeepingVars_(bool learn);

  /**
  @returns boolean value indicating whether enough rounds have passed to warrant
  updates of duty cycles
  */
  bool isUpdateRound_() const;

  //-------------------------------------------------------------------
  // Debugging helpers
  //-------------------------------------------------------------------

  /**
   Print the given UInt array in a nice format
  */
  void printState(const vector<UInt> &state, std::ostream& out=std::cout) const ;
  /**
  Print the given Real array in a nice format
  */
  void printState(const vector<Real> &state, std::ostream& out=std::cout) const;

  /**
  Print the main SP creation parameters to stdout.
   */
  void printParameters(std::ostream& out=std::cout) const;

  friend std::ostream& operator<< (std::ostream& stream, const SpatialPooler& self);


protected:
  UInt numInputs_;
  UInt numColumns_;
  vector<UInt> columnDimensions_;
  vector<UInt> inputDimensions_;
  UInt potentialRadius_;
  Real potentialPct_;
  Real initConnectedPct_;
  bool globalInhibition_;
  Real localAreaDensity_;
  UInt stimulusThreshold_;
  UInt inhibitionRadius_;
  UInt dutyCyclePeriod_;
  Real boostStrength_;
  UInt iterationNum_;
  UInt iterationLearnNum_;
  UInt spVerbosity_;
  bool wrapAround_;
  UInt updatePeriod_;

  Real synPermInactiveDec_;
  Real synPermActiveInc_;
  Real synPermBelowStimulusInc_;
  Real synPermConnected_;

  vector<Real> boostFactors_;
  vector<Real> overlapDutyCycles_;
  vector<Real> activeDutyCycles_;
  vector<Real> minOverlapDutyCycles_;
  vector<Real> minActiveDutyCycles_;

  Real minPctOverlapDutyCycles_;

  /*
   * Each mini-column is represented in the connections class by a single cell.
   * Each mini-column has a single segment.  Because all of these regularities,
   * each mini-column's index is also its Cell and Segment index.
   */
  Connections connections_;

  vector<SynapseIdx> overlaps_;
  vector<Real> boostedOverlaps_;


  UInt version_;
  Random rng_;

public:
  const Connections &connections = connections_;
};

std::ostream & operator<<(std::ostream & out, const SpatialPooler &sp);


} // end namespace htm
#endif // NTA_spatial_pooler_HPP
