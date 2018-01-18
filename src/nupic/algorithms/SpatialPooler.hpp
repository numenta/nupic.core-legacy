/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
 * Definitions for the Spatial Pooler in C++
 */

#ifndef NTA_spatial_pooler_HPP
#define NTA_spatial_pooler_HPP

#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <capnp/message.h>
#include <nupic/math/SparseBinaryMatrix.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/proto/SpatialPoolerProto.capnp.h>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>

using namespace std;

namespace nupic
{
  namespace algorithms
  {
    namespace spatial_pooler
    {

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
      class SpatialPooler : public Serializable<SpatialPoolerProto>
      {
        public:
          SpatialPooler();
          SpatialPooler(vector<UInt> inputDimensions,
                        vector<UInt> columnDimensions,
                        UInt potentialRadius=16,
                        Real potentialPct=0.5,
                        bool globalInhibition=true,
                        Real localAreaDensity=-1.0,
                        UInt numActiveColumnsPerInhArea=10,
                        UInt stimulusThreshold=0,
                        Real synPermInactiveDec=0.008,
                        Real synPermActiveInc=0.05,
                        Real synPermConnected=0.1,
                        Real minPctOverlapDutyCycles=0.001,
                        UInt dutyCyclePeriod=1000,
                        Real boostStrength=0.0,
                        Int seed=1,
                        UInt spVerbosity=0,
                        bool wrapAround=true);


          virtual ~SpatialPooler() {}

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
                length (2 * potentialRadius + 1).

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
                area). If localAreaDensity is set to a negative value output
                sparsity will be determined by the numActivePerInhArea.

          @param numActiveColumnsPerInhArea An alternate way to control the sparsity of
                active columns. If numActivePerInhArea is specified then
                localAreaDensity must be less than 0, and vice versa. When
                numActivePerInhArea > 0, the inhibition logic will insure that
                at most 'numActivePerInhArea' columns remain ON within a local
                inhibition area (the size of which is set by the internally
                calculated inhibitionRadius). When using this method, as columns
                learn and grow their effective receptive fields, the
                inhibitionRadius will grow, and hence the net density of the
                active columns will *decrease*. This is in contrast to the
                localAreaDensity method, which keeps the density of active
                columns the same regardless of the size of their receptive
                fields.

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
          control boosting strength. No boosting is applied if it is set to 0.
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
          virtual void initialize(vector<UInt> inputDimensions,
                                  vector<UInt> columnDimensions,
                                  UInt potentialRadius=16,
                                  Real potentialPct=0.5,
                                  bool globalInhibition=true,
                                  Real localAreaDensity=-1.0,
                                  UInt numActiveColumnsPerInhArea=10,
                                  UInt stimulusThreshold=0,
                                  Real synPermInactiveDec=0.01,
                                  Real synPermActiveInc=0.1,
                                  Real synPermConnected=0.1,
                                  Real minPctOverlapDutyCycles=0.001,
                                  UInt dutyCyclePeriod=1000,
                                  Real boostStrength=0.0,
                                  Int seed=1,
                                  UInt spVerbosity=0,
                                  bool wrapAround=true);

          /**
          This is the main workshorse method of the SpatialPooler class. This
          method takes an input vector and computes the set of output active
          columns. If 'learn' is set to True, this method also performs
          learning.

          @param inputVector An array of integer 0's and 1's that comprises
                the input to the spatial pooler. The length of the
                array must match the total number of input bits implied by
                the constructor (also returned by the method getNumInputs). In
                cases where the input is multi-dimensional, inputVector is a
                flattened array of inputs.

          @param learn A boolean value indicating whether learning should be
                performed. Learning entails updating the permanence values of
                the synapses, duty cycles, etc. Learning is typically on but
                setting learning to 'off' is useful for analyzing the current
                state of the SP. For example, you might want to feed in various
                inputs and examine the resulting SDR's. Note that if learning
                is off, boosting is turned off and columns that have never won
                will be removed from activeVector.  TODO: we may want to keep
                boosting on even when learning is off.

          @param activeVector An array representing the winning columns after
                inhibition. The size of the array is equal to the number of
                columns (also returned by the method getNumColumns). This array
                will be populated with 1's at the indices of the active columns,
                and 0's everywhere else. In the case where the output is
                multi-dimensional, activeVector represents a flattened array
                of outputs.
           */
          virtual void compute(UInt inputVector[], bool learn,
                               UInt activeVector[]);

          /**
           Removes the set of columns who have never been active from the set
           of active columns selected in the inhibition round. Such columns
           cannot represent learned pattern and are therefore meaningless if
           only inference is required.

           @param activeArray  An array of 1's and 0's representing winning
                 columns calculated by the 'compute' method after disabling
                 any columns that are not learned.
          */
          void stripUnlearnedColumns(UInt activeArray[]) const;

          /**
           * Get the version number of this spatial pooler.

           * @returns Integer version number.
           */
          virtual UInt version() const {
            return version_;
          };

          /**
          Save (serialize) the current state of the spatial pooler to the
          specified file.

          @param fd A valid file descriptor.
           */
          virtual void save(ostream& outStream) const;

          using Serializable::write;
          virtual void write(SpatialPoolerProto::Builder& proto) const override;

          /**
          Load (deserialize) and initialize the spatial pooler from the
          specified input stream.

          @param inStream A valid istream.
           */
          virtual void load(istream& inStream);

          using Serializable::read;
          virtual void read(SpatialPoolerProto::Reader& proto) override;

          /**
          Returns the number of bytes that a save operation would result in.
          Note: this method is currently somewhat inefficient as it just does
          a full save into an ostream and counts the resulting size.

          @returns Integer number of bytes
           */
          virtual UInt persistentSize() const;

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

          @param globalInhibition boolen varable of whether global inhibition is enabled.
          */
          void setGlobalInhibition(bool globalInhibition);

          /**
          Returns the number of active columns per inhibition area.

          @returns integer number of active columns per inhbition area, Returns a
          value less than 0 if parameter is unuse.
          */
          Int getNumActiveColumnsPerInhArea() const;


          /**
          Sets the number of active columns per inhibition area. Invalidates the
          'localAreaDensity' parameter.

          @param numActiveColumnsPerInhArea integer number of active columns per inhibition area.
          */
          void setNumActiveColumnsPerInhArea(UInt numActiveColumnsPerInhArea);

          /**
          Returns the local area density. Returns a value less than 0 if parameter
          is unused".

          @returns real number of local area density.
          */
          Real getLocalAreaDensity() const;

          /**
          Sets the local area density. Invalidates the 'numActivePerInhArea'
          parameter".

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
          Returns the permanence trim threshold.

          @returns real number of the permanence trim threshold.
          */
          Real getSynPermTrimThreshold() const;
          /**
          Sets the permanence trim threshold.

          @param synPermTrimThreshold real number of the permanence trim threshold.
          */
          void setSynPermTrimThreshold(Real synPermTrimThreshold);

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

          @param synPermInactiveDec real number of the permanence decrement amount for inactive synapses.
          */
          void setSynPermInactiveDec(Real synPermInactiveDec);

          /**
          Returns the permanence increment amount for columns that have not been
          recently active.

          @returns positive real number of the permanence increment amount for columns that have not been
          recently active.
          */
          Real getSynPermBelowStimulusInc() const;
          /**
          Sets the permanence increment amount for columns that have not been
          recently active.

          @param synPermBelowStimulusInc real number of the permanence increment amount for columns that have not been
          recently active, must be larger than 0.
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
          Sets the permanence amount that qualifies a synapse as
          being connected.

          @param synPermConnected real number of the permanence amount that qualifies a synapse as
          being connected.
          */
          void setSynPermConnected(Real synPermConnected);

          /**
          Returns the maximum permanence amount a synapse can
          achieve.

          @returns real number of the max permanence amount.
          */
          Real getSynPermMax() const;
          /**
          Sets the maximum permanence amount a synapse can
          achieve.

          @param synPermCMax real number of the maximum permanence
          amount that a synapse can achieve.
          */
          void setSynPermMax(Real synPermMax);

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

          @param overlapDutyCycles real array to store overlap duty cycles for all columns.
          */
          void getOverlapDutyCycles(Real overlapDutyCycles[]) const;
          /**
          Sets the overlap duty cycles for all columns. 'overlapDutyCycles'
          size must match the number of columns.

          @param overlapDutyCycles real array of the overlap duty cycles for all columns.
          */
          void setOverlapDutyCycles(Real overlapDutyCycles[]);

          /**
          Returns the activity duty cycles for all columns. 'activeDutyCycles'
          size must match the number of columns.

          @param activeDutyCycles real array to store activity duty cycles for all columns.
          */
          void getActiveDutyCycles(Real activeDutyCycles[]) const;
          /**
          Sets the activity duty cycles for all columns. 'activeDutyCycles'
          size must match the number of columns.

          @param activeDutyCycles real array of the activity duty cycles for all columns.
          */
          void setActiveDutyCycles(Real activeDutyCycles[]);

          /**
          Returns the minimum overlap duty cycles for all columns.

          @param minOverlapDutyCycles real arry to store mininum overlap duty cycles for all columns.
          'minOverlapDutyCycles' size must match the number of columns.
          */
          void getMinOverlapDutyCycles(Real minOverlapDutyCycles[]) const;
          /**
          Sets the minimum overlap duty cycles for all columns.
          '_minOverlapDutyCycles' size must match the number of columns.

          @param minOverlapDutyCycles real array of the minimum overlap duty cycles for all columns.
          */
          void setMinOverlapDutyCycles(Real minOverlapDutyCycles[]);

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
          void setPotential(UInt column, UInt potential[]);

          /**
          Returns the permanence values for a given column. 'permanence' size
          must match the number of inputs.

          @param column integer of column index.

          @param permanence real array to store permanence values for the selected column.
          */
          void getPermanence(UInt column, Real permanence[]) const;
          /**
          Sets the permanence values for a given column. 'permanence' size
          must match the number of inputs.

          @param column integer of column index.

          @param permanence real array of permanence values for the selected column.
          */
          void setPermanence(UInt column, Real permanence[]);

          /**
          Returns the connected synapses for a given column.
          'connectedSynapses' size must match the number of inputs.

          @param column integer of column index.

          @param connectedSynapses integer array to store the connected synapses for a given column.
          */
          void getConnectedSynapses(UInt column, UInt connectedSynapses[]) const;

          /**
          Returns the number of connected synapses for all columns.
          'connectedCounts' size must match the number of columns.

          @param connectedCounts integer array to store the connected synapses for all columns.
          */
          void getConnectedCounts(UInt connectedCounts[]) const;

          /**
          Print the main SP creation parameters to stdout.
           */
          void printParameters() const;

          /**
          Returns the overlap score for each column.
           */
          const vector<UInt>& getOverlaps() const;

          /**
          Returns the boosted overlap score for each column.
           */
          const vector<Real>& getBoostedOverlaps() const;


          ///////////////////////////////////////////////////////////
          //
          // Implementation methods. all methods below this line are
          // NOT part of the public API

          void toDense_(vector<UInt>& sparse,
                        UInt dense[],
                        UInt n);

          void boostOverlaps_(vector<UInt>& overlaps,
                              vector<Real>& boostedOverlaps);

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
            @param index       The index identifying a column in the permanence, potential
                               and connectivity matrices.
            @param wrapAround  A boolean value indicating that boundaries should be
                               ignored.
          */
          UInt mapColumn_(UInt column);

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

            ----------------------------
            @param column         An int index identifying a column in the permanence, potential
                            and connectivity matrices.

            @param wrapAround  A boolean value indicating that boundaries should be
                               ignored.
          */
          vector<UInt> mapPotential_(UInt column, bool wrapAround);

          /**
          Returns a randomly generated permanence value for a synapses that is
          initialized in a connected state.

          The basic idea here is to initialize
          permanence values very close to synPermConnected so that a small number of
          learning steps could make it disconnected or connected.

          Note: experimentation was done a long time ago on the best way to initialize
          permanence values, but the history for this particular scheme has been lost.

          @returns real number of a randomly generated permanence value for a synapses that is
          initialized in a connected state.
          */
          Real initPermConnected_();
          /**
              Returns a randomly generated permanence value for a synapses that is to be
              initialized in a non-connected state.

              @returns real number of a randomly generated permanence value for a synapses that is to be
              initialized in a non-connected state.
          */
          Real initPermNonConnected_();


          /**
            Initializes the permanences of a column. The method
            returns a 1-D array the size of the input, where each entry in the
            array represents the initial permanence value between the input bit
            at the particular index in the array, and the column represented by
            the 'index' parameter.

            @param potential      A int vector specifying the potential pool of the column.
                            Permanence values will only be generated for input bits

                            corresponding to indices for which the mask value is 1.
            @param connectedPct   A real value between 0 or 1 specifying the percent of the input
                            bits that will start off in a connected state.
          */
          vector<Real> initPermanence_(vector<UInt>& potential,
                                       Real connectedPct);
          void clip_(vector<Real>& perm, bool trim);

        /**
            This method updates the permanence matrix with a column's new permanence
            values.

            The column is identified by its index, which reflects the row in
            the matrix, and the permanence is given in 'dense' form, i.e. a full
            array containing all the zeros as well as the non-zero values. It is in
            charge of implementing 'clipping' - ensuring that the permanence values are
            always between 0 and 1 - and 'trimming' - enforcing sparsity by zeroing out
            all permanence values below '_synPermTrimThreshold'. It also maintains
            the consistency between 'self._permanences' (the matrix storing the
            permanence values), 'self._connectedSynapses', (the matrix storing the bits
            each column is connected to), and 'self._connectedCounts' (an array storing
            the number of input bits each column is connected to). Every method wishing
            to modify the permanence matrix should do so through this method.

            ----------------------------
            @param perm           An int vector of permanence values for a column. The array is
                            "dense", i.e. it contains an entry for each input bit, even
                            if the permanence value is 0.

            @param column          An int number identifying a column in the permanence, potential
                            and connectivity matrices.

            @param  raisePerm      a boolean value indicating whether the permanence values
                            should be raised until a minimum number are synapses are in
                            a connected state. Should be set to 'false' when a direct
                            assignment is required.
        */
          void updatePermanencesForColumn_(vector<Real>& perm, UInt column,
                                           bool raisePerm=true);
          UInt countConnected_(vector<Real>& perm);
          UInt raisePermanencesToThreshold_(vector<Real>& perm,
                                            vector<UInt>& potential);

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
          void calculateOverlap_(UInt inputVector[],
                                 vector<UInt>& overlap);
          void calculateOverlapPct_(vector<UInt>& overlaps,
                                    vector<Real>& overlapPct);


          bool isWinner_(Real score, vector<pair<UInt, Real> >& winners,
                         UInt numWinners);

          void addToWinners_(UInt index, Real score,
                             vector<pair<UInt, Real> >& winners);

          /**
              Performs inhibition. This method calculates the necessary values needed to
              actually perform inhibition and then delegates the task of picking the
              active columns to helper functions.


              @param overlaps       an array containing the overlap score for each  column.
                              The overlap score for a column is defined as the number
                              of synapses in a "connected state" (connected synapses)
                              that are connected to input bits which are turned on.

              @param activeColumns an int array containing the indices of the active columns.
          */
          void inhibitColumns_(
            const vector<Real>& overlaps,
            vector<UInt>& activeColumns);

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
          void inhibitColumnsGlobal_(
            const vector<Real>& overlaps,
            Real density,
            vector<UInt>& activeColumns);

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
          void inhibitColumnsLocal_(
            const vector<Real>& overlaps,
            Real density,
            vector<UInt>& activeColumns);

          /**
              The primary method in charge of learning.

              Adapts the permanence values of
              the synapses based on the input vector, and the chosen columns after
              inhibition round. Permanence values are increased for synapses connected to
              input bits that are turned on, and decreased for synapses connected to
              inputs bits that are turned off.

              ----------------------------
              @param inputVector    an int array of 0's and 1's that comprises the input to
                              the spatial pooler. There exists an entry in the array
                              for every input bit.

              @param  activeColumns  an int vector containing the indices of the columns that
                              survived inhibition.
                    */
          void adaptSynapses_(UInt inputVector[],
                              vector<UInt>& activeColumns);

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
              determine this quantity by first figuring out how many *inputs* a column is
              connected to, and then multiplying it by the total number of columns that
              exist for each input. For multiple dimension the aforementioned
              calculations are averaged over all dimensions of inputs and columns. This
              value is meaningless if global inhibition is enabled.
          */
          void updateInhibitionRadius_();

          /**
              REturns the average number of columns per input, taking into account the topology
              of the inputs and columns. This value is used to calculate the inhibition
              radius. This function supports an arbitrary number of dimensions. If the
              number of column dimensions does not match the number of input dimensions,
              we treat the missing, or phantom dimensions as 'ones'.

              @returns real number of the average number of columns per input.
          */
          Real avgColumnsPerInput_();

          /**
              The range of connected synapses for column. This is used to
              calculate the inhibition radius. This variation of the function only
              supports a 1 dimensional column topology.

              @param column An int number identifying a column in the permanence, potential
                              and connectivity matrices.
          */
          Real avgConnectedSpanForColumn1D_(UInt column);

          /**
              The range of connectedSynapses per column, averaged for each dimension.
              This vaule is used to calculate the inhibition radius. This variation of
              the  function only supports a 2 dimensional column topology.

              @param column An int number identifying a column in the permanence, potential
                              and connectivity matrices.
          */
          Real avgConnectedSpanForColumn2D_(UInt column);


          /**
              The range of connectedSynapses per column, averaged for each dimension.
              This vaule is used to calculate the inhibition radius. This variation of
              the function supports arbitrary column dimensions.

              @param column An int number identifying a column in the permanence, potential
                              and connectivity matrices.
          */
          Real avgConnectedSpanForColumnND_(UInt column);

          /**
              Updates the minimum duty cycles defining normal activity for a column. A
              column with activity duty cycle below this minimum threshold is boosted.
          */
          void updateMinDutyCycles_();

          /**
              Updates the minimum duty cycles in a global fashion. Sets the minimum duty
              cycles for the overlap and activation of all columns to be a percent of the
              maximum in the region, specified by minPctOverlapDutyCycle and
              minPctActiveDutyCycle respectively. Functionally it is equivalent to
              _updateMinDutyCyclesLocal, but this function exploits the globalilty of the
              computation to perform it in a straightforward, and more efficient manner.
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
              @param dutyCycles     A real array containing one or more duty cycle values that need
                              to be updated.

              @param newValues      A int vector used to update the duty cycle.

              @param period         A int number indicating the period of the duty cycle
          */
          static  void updateDutyCyclesHelper_(vector<Real>& dutyCycles,
                                               vector<UInt>& newValues,
                                               UInt period);

          /**
          Updates the duty cycles for each column. The OVERLAP duty cycle is a moving
          average of the number of inputs which overlapped with the each column. The
          ACTIVITY duty cycles is a moving average of the frequency of activation for
          each column.

          @param overlaps       an int vector containing the overlap score for each column.
                          The overlap score for a column is defined as the number
                          of synapses in a "connected state" (connected synapses)
                          that are connected to input bits which are turned on.

          @param activeArray  An int array containing the indices of the active columns,
                          the sprase set of columns which survived inhibition
          */
          void updateDutyCycles_(vector<UInt>& overlaps,
                                 UInt activeArray[]);

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
          @returns boolean value indicating whether enough rounds have passed to warrant updates of
          duty cycles
          */
          bool isUpdateRound_();

          /**
          Initialize the random seed

          @param seed 64bit int of random seed
          */
          void seed_(UInt64 seed);

          //-------------------------------------------------------------------
          // Debugging helpers
          //-------------------------------------------------------------------

          /**
           Print the given UInt array in a nice format
          */
          void printState(vector<UInt> &state);
          /**
          Print the given Real array in a nice format
          */
          void printState(vector<Real> &state);

        protected:
          UInt numInputs_;
          UInt numColumns_;
          vector<UInt> columnDimensions_;
          vector<UInt> inputDimensions_;
          UInt potentialRadius_;
          Real potentialPct_;
          Real initConnectedPct_;
          bool globalInhibition_;
          Int numActiveColumnsPerInhArea_;
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

          Real synPermMin_;
          Real synPermMax_;
          Real synPermTrimThreshold_;
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

          SparseMatrix<UInt,Real,Int,Real64> permanences_;
          SparseBinaryMatrix<UInt, UInt> potentialPools_;
          SparseBinaryMatrix<UInt, UInt> connectedSynapses_;
          vector<UInt> connectedCounts_;

          vector<UInt> overlaps_;
          vector<Real> overlapsPct_;
          vector<Real> boostedOverlaps_;
          vector<UInt> activeColumns_;
          vector<Real> tieBreaker_;

          UInt version_;
          Random rng_;

      };

    } // end namespace spatial_pooler
  } // end namespace algorithms
} // end namespace nupic
#endif // NTA_spatial_pooler_HPP
