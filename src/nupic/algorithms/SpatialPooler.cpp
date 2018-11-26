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
 * Implementation of SpatialPooler
 */

#include <string>
#include <algorithm>
#include <iterator> //begin()

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/math/Math.hpp>
#include <nupic/math/Topology.hpp>
#include <nupic/utils/VectorHelpers.hpp> 

#define VERSION 2  // version for stream serialization

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::spatial_pooler;
using namespace nupic::math::topology;
using nupic::utils::VectorHelpers;

static const Real PERMANENCE_EPSILON = 0.000001f;

// Round f to 5 digits of precision. This is used to set
// permanence values and help avoid small amounts of drift between
// platforms/implementations
static Real round5_(const Real f)
{
  return ((Real) ((Int) (f * 100000.0f))) / 100000.0f;
}

class CoordinateConverter2D { //TODO move to Topology

public:
  CoordinateConverter2D(UInt nrows, UInt ncols)
      : // TODO param nrows is unused
        ncols_(ncols) 
	{
	  NTA_ASSERT(ncols > 0u);
	}
  UInt toRow(UInt index) const { return index / ncols_; };
  UInt toCol(UInt index) const { return index % ncols_; };
  UInt toIndex(UInt row, UInt col) const { return row * ncols_ + col; };

private:
  UInt ncols_;
};

class CoordinateConverterND {

public:
  CoordinateConverterND(const vector<UInt> &dimensions) {
    NTA_ASSERT(!dimensions.empty());

    dimensions_ = dimensions;
    UInt b = 1u;
    for (Size i = dimensions.size(); i > 0u; i--) {
      bounds_.insert(bounds_.begin(), b);
      b *= dimensions[i-1];
    }
  }

  void toCoord(UInt index, vector<UInt> &coord) const {
    coord.clear();
    for (Size i = 0u; i < bounds_.size(); i++) {
      coord.push_back((index / bounds_[i]) % dimensions_[i]);
    }
  };

  UInt toIndex(vector<UInt> &coord) const {
    UInt index = 0;
    for (Size i = 0; i < coord.size(); i++) {
      index += coord[i] * bounds_[i];
    }
    return index;
  };

private:
  vector<UInt> dimensions_;
  vector<UInt> bounds_;
};

SpatialPooler::SpatialPooler() {
  // The current version number.
  version_ = 2;
}

SpatialPooler::SpatialPooler(
    const vector<UInt> inputDimensions, const vector<UInt> columnDimensions,
    UInt potentialRadius, Real potentialPct, bool globalInhibition,
    Real localAreaDensity, Int numActiveColumnsPerInhArea,
    UInt stimulusThreshold, Real synPermInactiveDec, Real synPermActiveInc,
    Real synPermConnected, Real minPctOverlapDutyCycles, UInt dutyCyclePeriod,
    Real boostStrength, Int seed, UInt spVerbosity, bool wrapAround)
    : SpatialPooler::SpatialPooler()
{
  // The current version number for serialzation.
  version_ = VERSION;

  initialize(inputDimensions,
             columnDimensions,
             potentialRadius,
             potentialPct,
             globalInhibition,
             localAreaDensity,
             numActiveColumnsPerInhArea,
             stimulusThreshold,
             synPermInactiveDec,
             synPermActiveInc,
             synPermConnected,
             minPctOverlapDutyCycles,
             dutyCyclePeriod,
             boostStrength,
             seed,
             spVerbosity,
             wrapAround);
}

vector<UInt> SpatialPooler::getColumnDimensions() const {
  return columnDimensions_;
}

vector<UInt> SpatialPooler::getInputDimensions() const {
  return inputDimensions_;
}

UInt SpatialPooler::getNumColumns() const { return numColumns_; }

UInt SpatialPooler::getNumInputs() const { return numInputs_; }

UInt SpatialPooler::getPotentialRadius() const { return potentialRadius_; }

void SpatialPooler::setPotentialRadius(UInt potentialRadius) {
  NTA_CHECK(potentialRadius < numInputs_);
  potentialRadius_ = potentialRadius;
}

Real SpatialPooler::getPotentialPct() const { return potentialPct_; }

void SpatialPooler::setPotentialPct(Real potentialPct) {
  NTA_CHECK(potentialPct > 0.0f && potentialPct <= 1.0f); 
  potentialPct_ = potentialPct;
}

bool SpatialPooler::getGlobalInhibition() const { return globalInhibition_; }

void SpatialPooler::setGlobalInhibition(bool globalInhibition) {
  globalInhibition_ = globalInhibition;
}

Int SpatialPooler::getNumActiveColumnsPerInhArea() const {
  return numActiveColumnsPerInhArea_;
}

void SpatialPooler::setNumActiveColumnsPerInhArea(UInt numActiveColumnsPerInhArea) {
  NTA_CHECK(numActiveColumnsPerInhArea > 0u && numActiveColumnsPerInhArea <= numColumns_); //TODO this boundary could be smarter
  numActiveColumnsPerInhArea_ = numActiveColumnsPerInhArea;
  localAreaDensity_ = DISABLED;  //MUTEX with localAreaDensity
}

Real SpatialPooler::getLocalAreaDensity() const { return localAreaDensity_; }

void SpatialPooler::setLocalAreaDensity(Real localAreaDensity) {
  NTA_CHECK(localAreaDensity > 0.0f && localAreaDensity <= 1.0f); 
  localAreaDensity_ = localAreaDensity;
  numActiveColumnsPerInhArea_ = DISABLED; //MUTEX with numActiveColumnsPerInhArea
}

UInt SpatialPooler::getStimulusThreshold() const { return stimulusThreshold_; }

void SpatialPooler::setStimulusThreshold(UInt stimulusThreshold) {
  stimulusThreshold_ = stimulusThreshold;
}

UInt SpatialPooler::getInhibitionRadius() const { return inhibitionRadius_; }

void SpatialPooler::setInhibitionRadius(UInt inhibitionRadius) {
  inhibitionRadius_ = inhibitionRadius;
}

UInt SpatialPooler::getDutyCyclePeriod() const { return dutyCyclePeriod_; }

void SpatialPooler::setDutyCyclePeriod(UInt dutyCyclePeriod) {
  dutyCyclePeriod_ = dutyCyclePeriod;
}

Real SpatialPooler::getBoostStrength() const { return boostStrength_; }

void SpatialPooler::setBoostStrength(Real boostStrength) {
  NTA_CHECK(boostStrength >= 0.0f);
  boostStrength_ = boostStrength;
}

UInt SpatialPooler::getIterationNum() const { return iterationNum_; }

void SpatialPooler::setIterationNum(UInt iterationNum) {
  iterationNum_ = iterationNum;
}

UInt SpatialPooler::getIterationLearnNum() const { return iterationLearnNum_; }

void SpatialPooler::setIterationLearnNum(UInt iterationLearnNum) {
  iterationLearnNum_ = iterationLearnNum;
}

UInt SpatialPooler::getSpVerbosity() const { return spVerbosity_; }

void SpatialPooler::setSpVerbosity(UInt spVerbosity) {
  spVerbosity_ = spVerbosity;
}

bool SpatialPooler::getWrapAround() const { return wrapAround_; }

void SpatialPooler::setWrapAround(bool wrapAround) { wrapAround_ = wrapAround; }

UInt SpatialPooler::getUpdatePeriod() const { return updatePeriod_; }

void SpatialPooler::setUpdatePeriod(UInt updatePeriod) {
  updatePeriod_ = updatePeriod;
}

Real SpatialPooler::getSynPermTrimThreshold() const {
  return synPermTrimThreshold_;
}

void SpatialPooler::setSynPermTrimThreshold(Real synPermTrimThreshold) {
  NTA_CHECK(synPermTrimThreshold >= synPermMin_ &&
             synPermTrimThreshold <= synPermMax_);
  synPermTrimThreshold_ = synPermTrimThreshold;
}

Real SpatialPooler::getSynPermActiveInc() const { return synPermActiveInc_; }

void SpatialPooler::setSynPermActiveInc(Real synPermActiveInc) {
  NTA_CHECK(synPermActiveInc > 0.0f && synPermActiveInc < synPermMax_);
  synPermActiveInc_ = synPermActiveInc;
}

Real SpatialPooler::getSynPermInactiveDec() const {
  return synPermInactiveDec_;
}

void SpatialPooler::setSynPermInactiveDec(Real synPermInactiveDec) {
  NTA_CHECK(synPermInactiveDec >= 0.0f && synPermInactiveDec <= synPermMax_); 
  synPermInactiveDec_ = synPermInactiveDec;
}

Real SpatialPooler::getSynPermBelowStimulusInc() const {
  return synPermBelowStimulusInc_;
}

void SpatialPooler::setSynPermBelowStimulusInc(Real synPermBelowStimulusInc) {
  NTA_CHECK(synPermBelowStimulusInc > 0.0f && synPermBelowStimulusInc <= synPermMax_);
  synPermBelowStimulusInc_ = synPermBelowStimulusInc;
}

Real SpatialPooler::getSynPermConnected() const { return synPermConnected_; }

void SpatialPooler::setSynPermConnected(Real synPermConnected) {
  NTA_CHECK(synPermConnected > synPermMin_ && synPermConnected <= synPermMax_);
  synPermConnected_ = synPermConnected;
}

Real SpatialPooler::getSynPermMax() const { return synPermMax_; }

void SpatialPooler::setSynPermMax(Real synPermMax) { 
	NTA_CHECK(synPermMax > synPermMin_);
	synPermMax_ = synPermMax; 
}

Real SpatialPooler::getMinPctOverlapDutyCycles() const {
  return minPctOverlapDutyCycles_;
}

void SpatialPooler::setMinPctOverlapDutyCycles(Real minPctOverlapDutyCycles) {
  NTA_CHECK(minPctOverlapDutyCycles > 0.0f && minPctOverlapDutyCycles <= 1.0f);
  minPctOverlapDutyCycles_ = minPctOverlapDutyCycles;
}

void SpatialPooler::getBoostFactors(Real boostFactors[]) const { //TODO make vector
  copy(boostFactors_.begin(), boostFactors_.end(), boostFactors);
}

void SpatialPooler::setBoostFactors(Real boostFactors[]) {
  boostFactors_.assign(&boostFactors[0], &boostFactors[numColumns_]);
}

void SpatialPooler::getOverlapDutyCycles(Real overlapDutyCycles[]) const {
  copy(overlapDutyCycles_.begin(), overlapDutyCycles_.end(), overlapDutyCycles);
}

void SpatialPooler::setOverlapDutyCycles(const Real overlapDutyCycles[]) {
  overlapDutyCycles_.assign(&overlapDutyCycles[0],
                            &overlapDutyCycles[numColumns_]);
}

void SpatialPooler::getActiveDutyCycles(Real activeDutyCycles[]) const {
  copy(activeDutyCycles_.begin(), activeDutyCycles_.end(), activeDutyCycles);
}

void SpatialPooler::setActiveDutyCycles(const Real activeDutyCycles[]) {
  activeDutyCycles_.assign(&activeDutyCycles[0],
                           &activeDutyCycles[numColumns_]);
}

void SpatialPooler::getMinOverlapDutyCycles(Real minOverlapDutyCycles[]) const {
  copy(minOverlapDutyCycles_.begin(), minOverlapDutyCycles_.end(),
       minOverlapDutyCycles);
}

void SpatialPooler::setMinOverlapDutyCycles(const Real minOverlapDutyCycles[]) {
  minOverlapDutyCycles_.assign(&minOverlapDutyCycles[0],
                               &minOverlapDutyCycles[numColumns_]);
}

void SpatialPooler::getPotential(UInt column, UInt potential[]) const {
  NTA_ASSERT(column < numColumns_);
  potentialPools_.getRow(column, &potential[0], &potential[numInputs_]);
}

void SpatialPooler::setPotential(UInt column, const UInt potential[]) {
  NTA_CHECK(column < numColumns_);
  potentialPools_.rowFromDense(column, &potential[0], &potential[numInputs_]);
}

void SpatialPooler::getPermanence(UInt column, Real permanences[]) const {
  NTA_ASSERT(column < numColumns_);
  permanences_.getRowToDense(column, permanences);
}

void SpatialPooler::setPermanence(UInt column, const Real permanences[]) {
  NTA_ASSERT(column < numColumns_);
  vector<Real> perm;
  perm.assign(&permanences[0], &permanences[numInputs_]);
  updatePermanencesForColumn_(perm, column, false);
}

void SpatialPooler::getConnectedSynapses(UInt column,
                                         UInt connectedSynapses[]) const {
  NTA_ASSERT(column < numColumns_);
  connectedSynapses_.getRow(column, &connectedSynapses[0],
                            &connectedSynapses[numInputs_]);
}

void SpatialPooler::getConnectedCounts(UInt connectedCounts[]) const {
  copy(connectedCounts_.begin(), connectedCounts_.end(), connectedCounts);
}

const vector<UInt> &SpatialPooler::getOverlaps() const { return overlaps_; }

const vector<Real> &SpatialPooler::getBoostedOverlaps() const {
  return boostedOverlaps_;
}

void SpatialPooler::initialize(
    const vector<UInt> inputDimensions, const vector<UInt> columnDimensions,
    UInt potentialRadius, Real potentialPct, bool globalInhibition,
    Real localAreaDensity, Int numActiveColumnsPerInhArea,
    UInt stimulusThreshold, Real synPermInactiveDec, Real synPermActiveInc,
    Real synPermConnected, Real minPctOverlapDutyCycles, UInt dutyCyclePeriod,
    Real boostStrength, Int seed, UInt spVerbosity, bool wrapAround) {

  numInputs_ = 1u;
  inputDimensions_.clear();
  for (auto &inputDimension : inputDimensions) {
    NTA_CHECK(inputDimension > 0) << "Input dimensions must be positive integers!";
    numInputs_ *= inputDimension;
    inputDimensions_.push_back(inputDimension);
  }
  numColumns_ = 1u;
  columnDimensions_.clear();
  for (auto &columnDimension : columnDimensions) {
    NTA_CHECK(columnDimension > 0) << "Column dimensions must be positive integers!"; 
    numColumns_ *= columnDimension;
    columnDimensions_.push_back(columnDimension);
  }
  NTA_CHECK(numColumns_ > 0);
  NTA_CHECK(numInputs_ > 0);
  NTA_CHECK(inputDimensions_.size() == columnDimensions_.size());
  
  NTA_CHECK((numActiveColumnsPerInhArea > 0 && localAreaDensity < 0) ||
            (localAreaDensity > 0 && localAreaDensity <= MAX_LOCALAREADENSITY 
	     && numActiveColumnsPerInhArea < 0)
	   ) << numActiveColumnsPerInhArea << " vs " << localAreaDensity;
  numActiveColumnsPerInhArea_ = numActiveColumnsPerInhArea;
  localAreaDensity_ = localAreaDensity;
  
  rng_ = Random(seed);

  potentialRadius_ = potentialRadius > numInputs_ ? numInputs_ : potentialRadius;
  NTA_CHECK(potentialPct > 0 && potentialPct <= 1);
  potentialPct_ = potentialPct;
  globalInhibition_ = globalInhibition;
  stimulusThreshold_ = stimulusThreshold;
  synPermInactiveDec_ = synPermInactiveDec;
  synPermActiveInc_ = synPermActiveInc;
  synPermBelowStimulusInc_ = synPermConnected / 10.0f;
  synPermConnected_ = synPermConnected;
  minPctOverlapDutyCycles_ = minPctOverlapDutyCycles;
  dutyCyclePeriod_ = dutyCyclePeriod;
  boostStrength_ = boostStrength;
  spVerbosity_ = spVerbosity;
  wrapAround_ = wrapAround;
  synPermMin_ = 0.0f;
  synPermMax_ = 1.0f;
  synPermTrimThreshold_ = synPermActiveInc / 2.0f;
  NTA_CHECK(synPermTrimThreshold_ < synPermConnected_);
  updatePeriod_ = 50u;
  initConnectedPct_ = 0.5f;
  iterationNum_ = 0u;
  iterationLearnNum_ = 0u;

  tieBreaker_.resize(numColumns_);
  for (Size i = 0; i < numColumns_; i++) {
    tieBreaker_[i] = 0.01f * rng_.getReal64();
  }

  potentialPools_.resize(numColumns_, numInputs_);
  permanences_.resize(numColumns_, numInputs_);
  connectedSynapses_.resize(numColumns_, numInputs_);
  connectedCounts_.resize(numColumns_);

  overlapDutyCycles_.assign(numColumns_, 0);
  activeDutyCycles_.assign(numColumns_, 0);
  minOverlapDutyCycles_.assign(numColumns_, 0.0);
  boostFactors_.assign(numColumns_, 1);
  overlaps_.resize(numColumns_);
  overlapsPct_.resize(numColumns_);
  boostedOverlaps_.resize(numColumns_);

  inhibitionRadius_ = 0;

  for (Size i = 0; i < numColumns_; ++i) {
    vector<UInt> potential = mapPotential_(i, wrapAround_);
    vector<Real> perm = initPermanence_(potential, initConnectedPct_);
    potentialPools_.rowFromDense(i, potential.begin(), potential.end());
    updatePermanencesForColumn_(perm, i, true);
  }

  updateInhibitionRadius_();

  if (spVerbosity_ > 0) {
    printParameters();
    std::cout << "CPP SP seed                 = " << seed << std::endl;
  }
}

void SpatialPooler::compute(const UInt inputArray[], bool learn, UInt activeArray[]) {
  updateBookeepingVars_(learn);
  calculateOverlap_(inputArray, overlaps_);
  calculateOverlapPct_(overlaps_, overlapsPct_);

  if (learn) {
    boostOverlaps_(overlaps_, boostedOverlaps_);
  } else {
    boostedOverlaps_.assign(overlaps_.begin(), overlaps_.end());
  }

  inhibitColumns_(boostedOverlaps_, activeColumns_);
  const vector<UInt> spars = VectorHelpers::sparseToBinary<UInt>(activeColumns_, numColumns_);
  copy(begin(spars), end(spars), activeArray);


  if (learn) {
    adaptSynapses_(inputArray, activeColumns_);
    updateDutyCycles_(overlaps_, activeArray);
    bumpUpWeakColumns_();
    updateBoostFactors_();
    if (isUpdateRound_()) {
      updateInhibitionRadius_();
      updateMinDutyCycles_();
    }
  }
}


void SpatialPooler::stripUnlearnedColumns(UInt activeArray[]) const {
  for (Size i = 0; i < numColumns_; i++) {
    if (activeDutyCycles_[i] == 0) { //TODO make sparse
      activeArray[i] = 0u;
    }
  }
}


void SpatialPooler::boostOverlaps_(const vector<UInt> &overlaps,
                                   vector<Real> &boosted) const {
  for (UInt i = 0; i < numColumns_; i++) {
    boosted[i] = overlaps[i] * boostFactors_[i];
  }
}

UInt SpatialPooler::mapColumn_(UInt column) const {
  NTA_ASSERT(column < numColumns_);
  vector<UInt> columnCoords;
  const CoordinateConverterND columnConv(columnDimensions_);
  columnConv.toCoord(column, columnCoords);

  vector<UInt> inputCoords;
  inputCoords.reserve(columnCoords.size());
  for (Size i = 0; i < columnCoords.size(); i++) {
    const Real inputCoord = ((Real)columnCoords[i] + 0.5f) *
                            (inputDimensions_[i] / (Real)columnDimensions_[i]);
    inputCoords.push_back(floor(inputCoord));
  }

  const CoordinateConverterND inputConv(inputDimensions_);
  return inputConv.toIndex(inputCoords);
}

vector<UInt> SpatialPooler::mapPotential_(UInt column, bool wrapAround) {
  NTA_ASSERT(column < numColumns_);
  const UInt centerInput = mapColumn_(column);

  vector<UInt> columnInputs;
  if (wrapAround) {
    for (UInt input : WrappingNeighborhood(centerInput, potentialRadius_, inputDimensions_)) {
      columnInputs.push_back(input);
    }
  } else {
    for (UInt input :
         Neighborhood(centerInput, potentialRadius_, inputDimensions_)) {
      columnInputs.push_back(input);
    }
  }

  const UInt numPotential = round(columnInputs.size() * potentialPct_);
  const auto selectedInputs = rng_.sample<UInt>(columnInputs, numPotential);
  const vector<UInt> potential = VectorHelpers::sparseToBinary<UInt>(selectedInputs, numInputs_);
  return potential;
}


Real SpatialPooler::initPermConnected_() {
  Real p =
      synPermConnected_ + (synPermMax_ - synPermConnected_) * rng_.getReal64();

  return round5_(p);
}


Real SpatialPooler::initPermNonConnected_() {
  Real p = synPermConnected_ * rng_.getReal64();
  return round5_(p);
}


vector<Real> SpatialPooler::initPermanence_(const vector<UInt> &potential, //TODO make potential sparse
                                            Real connectedPct) {
  vector<Real> perm(numInputs_, 0);
  for (UInt i = 0; i < numInputs_; i++) {
    if (potential[i] < 1) {
      continue;
    }

    if (rng_.getReal64() <= connectedPct) {
      perm[i] = initPermConnected_();
    } else {
      perm[i] = initPermNonConnected_();
    }
    perm[i] = perm[i] < synPermTrimThreshold_ ? 0.0f : perm[i];
  }

  return perm;
}


void SpatialPooler::clip_(vector<Real> &perm, bool trim) const {
  const Real minVal = trim ? synPermTrimThreshold_ : synPermMin_;
  for (auto &elem : perm) {
    elem = elem > synPermMax_ ? synPermMax_ : elem; //crop upper bound
    elem = elem < minVal ? synPermMin_ : elem; //crop lower
  }
}


void SpatialPooler::updatePermanencesForColumn_(vector<Real> &perm, UInt column,
                                                bool raisePerm) {
  NTA_ASSERT(column < numColumns_);

  if (raisePerm) {
    const vector<UInt> potential = potentialPools_.getSparseRow(column);
    raisePermanencesToThreshold_(perm, potential);
  }

  UInt numConnected = 0u;
  vector<Real> connectedSparse;
  for (Size i = 0; i < perm.size(); ++i) { //TODO use binary2Sparse
    if (perm[i] >= synPermConnected_ - PERMANENCE_EPSILON) {
      connectedSparse.push_back(i);
      ++numConnected;
    }
  }

  clip_(perm, true);
  connectedSynapses_.replaceSparseRow(column, connectedSparse.begin(), 
                                      connectedSparse.end());
  permanences_.setRowFromDense(column, perm);
  connectedCounts_[column] = numConnected;
}


UInt SpatialPooler::countConnected_(const vector<Real> &perm) const {
  UInt numConnected = 0;
  for (auto &elem : perm) {
    if (elem >= synPermConnected_ - PERMANENCE_EPSILON) {
      ++numConnected;
    }
  }
  return numConnected;
}


UInt SpatialPooler::raisePermanencesToThreshold_(vector<Real>& perm,
                                                 const vector<UInt>& potential) const
{
  clip_(perm, false);
  UInt numConnected = countConnected_(perm);
  while (numConnected < stimulusThreshold_) //TODO avoid the while-true loop, grow syns in 1 step
  {
    for (auto & elem : potential) {
      perm[elem] += synPermBelowStimulusInc_;
    }
    numConnected = countConnected_(perm);
  }
  return numConnected;
}


void SpatialPooler::updateInhibitionRadius_() {
  if (globalInhibition_) {
    inhibitionRadius_ =
        *max_element(columnDimensions_.begin(), columnDimensions_.end());
    return;
  }

  Real connectedSpan = 0.0f;
  for (UInt i = 0; i < numColumns_; i++) {
    connectedSpan += avgConnectedSpanForColumnND_(i);
  }
  connectedSpan /= numColumns_;
  const Real columnsPerInput = avgColumnsPerInput_();
  const Real diameter = connectedSpan * columnsPerInput;
  Real radius = (diameter - 1) / 2.0f;
  radius = max((Real)1.0, radius);
  inhibitionRadius_ = UInt(round(radius));
}


void SpatialPooler::updateMinDutyCycles_() {
  if (globalInhibition_ ||
      inhibitionRadius_ >
          *max_element(columnDimensions_.begin(), columnDimensions_.end())) {
    updateMinDutyCyclesGlobal_();
  } else {
    updateMinDutyCyclesLocal_();
  }
}


void SpatialPooler::updateMinDutyCyclesGlobal_() {
  const Real maxOverlapDutyCycles =
      *max_element(overlapDutyCycles_.begin(), overlapDutyCycles_.end());

  fill(minOverlapDutyCycles_.begin(), minOverlapDutyCycles_.end(),
       minPctOverlapDutyCycles_ * maxOverlapDutyCycles);
}


void SpatialPooler::updateMinDutyCyclesLocal_() {
  for (UInt i = 0; i < numColumns_; i++) {
    Real maxActiveDuty = 0.0f;
    Real maxOverlapDuty = 0.0f;
    if (wrapAround_) {
     for(auto column : WrappingNeighborhood(i, inhibitionRadius_, columnDimensions_)) {
      maxActiveDuty = max(maxActiveDuty, activeDutyCycles_[column]);
      maxOverlapDuty = max(maxOverlapDuty, overlapDutyCycles_[column]);
     }
    } else { 
     for(auto column: Neighborhood(i, inhibitionRadius_, columnDimensions_)) {
      maxActiveDuty = max(maxActiveDuty, activeDutyCycles_[column]);
      maxOverlapDuty = max(maxOverlapDuty, overlapDutyCycles_[column]);
      }
    }

    minOverlapDutyCycles_[i] = maxOverlapDuty * minPctOverlapDutyCycles_;
  }
}


void SpatialPooler::updateDutyCycles_(const vector<UInt> &overlaps,
                                      const UInt activeArray[]) {
  vector<UInt> newOverlapVal(numColumns_, 0);
  vector<UInt> newActiveVal(numColumns_, 0);

  for (UInt i = 0; i < numColumns_; i++) {
    newOverlapVal[i] = overlaps[i] > 0 ? 1 : 0;
    newActiveVal[i] = activeArray[i] > 0 ? 1 : 0;
  }

  const UInt period =
      dutyCyclePeriod_ > iterationNum_ ? iterationNum_ : dutyCyclePeriod_;

  updateDutyCyclesHelper_(overlapDutyCycles_, newOverlapVal, period);
  updateDutyCyclesHelper_(activeDutyCycles_, newActiveVal, period);
}


Real SpatialPooler::avgColumnsPerInput_() const {
  const UInt numDim = max(columnDimensions_.size(), inputDimensions_.size());
  Real columnsPerInput = 0.0f;
  for (UInt i = 0; i < numDim; i++) {
    const Real col = (i < columnDimensions_.size()) ? columnDimensions_[i] : 1;
    const Real input = (i < inputDimensions_.size()) ? inputDimensions_[i] : 1;
    columnsPerInput += col / input;
  }
  return columnsPerInput / numDim;
}


Real SpatialPooler::avgConnectedSpanForColumn1D_(UInt column) const {
  NTA_ASSERT(column < numColumns_);
  NTA_ASSERT(inputDimensions_.size() == 1);

  const vector<UInt> connectedSparse = connectedSynapses_.getSparseRow(column);
  if (connectedSparse.empty())
    return 0;
  auto minmax = minmax_element(connectedSparse.begin(), connectedSparse.end());
  return *minmax.second /*max*/ - *minmax.first /*min*/ + 1;
}


Real SpatialPooler::avgConnectedSpanForColumn2D_(UInt column) const {
  NTA_ASSERT(column < numColumns_);
  NTA_ASSERT(inputDimensions_.size() == 2);

  const UInt nrows = inputDimensions_[0];
  const UInt ncols = inputDimensions_[1];

  const CoordinateConverter2D conv(nrows, ncols);

  const vector<UInt> connectedSparse = connectedSynapses_.getSparseRow(column);
  vector<UInt> rows, cols;
  for (auto &elem : connectedSparse) {
    rows.push_back(conv.toRow(elem));
    cols.push_back(conv.toCol(elem));
  }

  if (rows.empty() && cols.empty()) {
    return 0;
  }

  auto minmaxRows = minmax_element(rows.begin(), rows.end());
  const UInt rowSpan = *minmaxRows.second /*max*/ - *minmaxRows.first /*min*/ + 1;

  auto minmaxCols = minmax_element(cols.begin(), cols.end());
  const UInt colSpan = *minmaxCols.second - *minmaxCols.first + 1;

  return (rowSpan + colSpan) / 2.0f;
}


Real SpatialPooler::avgConnectedSpanForColumnND_(UInt column) const {
  NTA_ASSERT(column < numColumns_);
  
  const UInt numDimensions = inputDimensions_.size();
  const vector<UInt> connectedSparse = connectedSynapses_.getSparseRow(column);
  if (connectedSparse.empty()) {
    return 0;
  }

  vector<UInt> maxCoord(numDimensions, 0);
  vector<UInt> minCoord(numDimensions, *max_element(inputDimensions_.begin(),
                                                    inputDimensions_.end()));
  const CoordinateConverterND conv(inputDimensions_);

  for (auto &elem : connectedSparse) {
    vector<UInt> columnCoord;
    conv.toCoord(elem, columnCoord);
    for (size_t j = 0; j < columnCoord.size(); j++) {
      maxCoord[j] = max(maxCoord[j], columnCoord[j]); //FIXME this computation may be flawed
      minCoord[j] = min(minCoord[j], columnCoord[j]);
    }
  }

  UInt totalSpan = 0;
  for (size_t j = 0; j < inputDimensions_.size(); j++) {
    totalSpan += maxCoord[j] - minCoord[j] + 1;
  }

  return (Real)totalSpan / inputDimensions_.size();
}


void SpatialPooler::adaptSynapses_(const UInt inputVector[], //TODO make sparse
                                   const vector<UInt> &activeColumns) {
  vector<Real> permChanges(numInputs_, -1 * synPermInactiveDec_);
  for (UInt i = 0; i < numInputs_; i++) {
    if (inputVector[i] > 0) {
      permChanges[i] = synPermActiveInc_;
    }
  }

  for (Size i = 0; i < activeColumns.size(); i++) {
    const UInt column = activeColumns[i];
    const vector<UInt> potential = potentialPools_.getSparseRow(column);
    vector<Real> perm(numInputs_, 0);
    permanences_.getRowToDense(column, perm);
    for (auto & elem : potential) {
        perm[elem] += permChanges[elem];
    }

    updatePermanencesForColumn_(perm, column, true);
  }
}


void SpatialPooler::bumpUpWeakColumns_() {
  for (UInt i = 0; i < numColumns_; i++) {
    if (overlapDutyCycles_[i] >= minOverlapDutyCycles_[i]) {
      continue;
    }
    vector<Real> perm(numInputs_, 0);
    const vector<UInt> potential = potentialPools_.getSparseRow(i);
    permanences_.getRowToDense(i, perm);
    for (auto & elem : potential) {
      perm[elem] += synPermBelowStimulusInc_;
    }
    updatePermanencesForColumn_(perm, i, false);
  }
}


void SpatialPooler::updateDutyCyclesHelper_(vector<Real> &dutyCycles,
                                            const vector<UInt> &newValues,
                                            UInt period) {
  NTA_ASSERT(period > 0);
  NTA_ASSERT(dutyCycles.size() == newValues.size());

  for (Size i = 0; i < dutyCycles.size(); i++) {
    dutyCycles[i] = (dutyCycles[i] * (period - 1) + newValues[i]) / period;
  }
}


void SpatialPooler::updateBoostFactors_() {
  if (globalInhibition_) {
    updateBoostFactorsGlobal_();
  } else {
    updateBoostFactorsLocal_();
  }
}


void SpatialPooler::updateBoostFactorsGlobal_() {
  Real targetDensity;
  if (numActiveColumnsPerInhArea_ > 0) {
    UInt inhibitionArea =
        pow((Real)(2 * inhibitionRadius_ + 1), (Real)columnDimensions_.size());
    inhibitionArea = min(inhibitionArea, numColumns_);
    NTA_ASSERT(inhibitionArea > 0);
    targetDensity = ((Real)numActiveColumnsPerInhArea_) / inhibitionArea;
    targetDensity = min(targetDensity, (Real)MAX_LOCALAREADENSITY);
  } else {
    targetDensity = localAreaDensity_;
  }

  for (UInt i = 0; i < numColumns_; ++i) {
    boostFactors_[i] = exp((targetDensity - activeDutyCycles_[i]) * boostStrength_);
  }
}


void SpatialPooler::updateBoostFactorsLocal_() {
  for (UInt i = 0; i < numColumns_; ++i) {
    UInt numNeighbors = 0u;
    Real localActivityDensity = 0.0f;

    if (wrapAround_) {
      for(auto neighbor: WrappingNeighborhood(i, inhibitionRadius_, columnDimensions_)) {
        localActivityDensity += activeDutyCycles_[neighbor];
        numNeighbors += 1;
      }
    } else {
      for(auto neighbor: Neighborhood(i, inhibitionRadius_, columnDimensions_)) {
        localActivityDensity += activeDutyCycles_[neighbor];
        numNeighbors += 1;
      }
    }

    const Real targetDensity = localActivityDensity / numNeighbors;
    boostFactors_[i] =
        exp((targetDensity - activeDutyCycles_[i]) * boostStrength_);
  }
}


void SpatialPooler::updateBookeepingVars_(bool learn) {
  iterationNum_++;
  if (learn) {
    iterationLearnNum_++;
  }
}


void SpatialPooler::calculateOverlap_(const UInt inputVector[],
                                      vector<UInt> &overlaps) const {
  overlaps.assign(numColumns_, 0);
  connectedSynapses_.rightVecSumAtNZ(inputVector, inputVector + numInputs_,
                                     overlaps.begin(), overlaps.end());
}


void SpatialPooler::calculateOverlapPct_(const vector<UInt> &overlaps,
                                         vector<Real> &overlapPct) const {
  overlapPct.assign(numColumns_, 0);
  for (UInt i = 0; i < numColumns_; i++) {
    if (connectedCounts_[i] != 0) {
      overlapPct[i] = ((Real)overlaps[i]) / connectedCounts_[i];
    } 
  }
}


void SpatialPooler::inhibitColumns_(const vector<Real> &overlaps,
                                    vector<UInt> &activeColumns) const {
  Real density = localAreaDensity_;
  if (numActiveColumnsPerInhArea_ > 0) {
    UInt inhibitionArea =
        pow((Real)(2 * inhibitionRadius_ + 1), (Real)columnDimensions_.size());
    inhibitionArea = min(inhibitionArea, numColumns_);
    density = ((Real)numActiveColumnsPerInhArea_) / inhibitionArea;
    density = min(density, (Real)MAX_LOCALAREADENSITY);
  }

  if (globalInhibition_ ||
      inhibitionRadius_ >
          *max_element(columnDimensions_.begin(), columnDimensions_.end())) {
    inhibitColumnsGlobal_(overlaps, density, activeColumns);
  } else {
    inhibitColumnsLocal_(overlaps, density, activeColumns);
  }
}


void SpatialPooler::inhibitColumnsGlobal_(const vector<Real> &overlaps,
                                          Real density,
                                          vector<UInt> &activeColumns) const {
  NTA_ASSERT(!overlaps.empty());
  NTA_ASSERT(density > 0.0f && density <= 1.0f);

  activeColumns.clear();
  const UInt numDesired = (UInt)(density * numColumns_);
  NTA_CHECK(numDesired > 0) << "Not enough columns (" << numColumns_ << ") "
                            << "for desired density (" << density << ").";
  // Sort the columns by the amount of overlap.  First make a list of all of the
  // column indexes.
  activeColumns.reserve(numColumns_);
  for(UInt i = 0; i < numColumns_; i++)
    activeColumns.push_back(i);
  // Compare the column indexes by their overlap.
  auto compare = [&overlaps](const UInt &a, const UInt &b) -> bool
    {return overlaps[a] > overlaps[b];};
  // Do a partial sort to divide the winners from the losers.  This sort is
  // faster than a regular sort because it stops after it partitions the
  // elements about the Nth element, with all elements on their correct side of
  // the Nth element.
  std::nth_element(
    activeColumns.begin(),
    activeColumns.begin() + numDesired,
    activeColumns.end(),
    compare);
  // Remove the columns which lost the competition.
  activeColumns.resize(numDesired);
  // Finish sorting the winner columns by their overlap.
  std::sort(activeColumns.begin(), activeColumns.end(), compare);
  // Remove sub-threshold winners
  while( !activeColumns.empty() && 
         overlaps[activeColumns.back()] < stimulusThreshold_)
      activeColumns.pop_back();
}


void SpatialPooler::inhibitColumnsLocal_(const vector<Real> &overlaps,
                                         Real density,
                                         vector<UInt> &activeColumns) const {
  activeColumns.clear();

  // Tie-breaking: when overlaps are equal, columns that have already been
  // selected are treated as "bigger".
  vector<bool> activeColumnsDense(numColumns_, false);

  for (UInt column = 0; column < numColumns_; column++) {
    if (overlaps[column] < stimulusThreshold_) {
      continue;
    }
    
    UInt numNeighbors = 0;
    UInt numBigger = 0;


      if (wrapAround_) {
        for(auto neighbor: WrappingNeighborhood(column, inhibitionRadius_,columnDimensions_)) {
          if (neighbor == column) {
            continue;
          }
          numNeighbors++;

          const Real difference = overlaps[neighbor] - overlaps[column];
          if (difference > 0 || (difference == 0 && activeColumnsDense[neighbor])) {
            numBigger++;
          }
	}
      } else {
        for(auto neighbor: Neighborhood(column, inhibitionRadius_, columnDimensions_)) {
          if (neighbor == column) {
            continue;
          }
          numNeighbors++;

          const Real difference = overlaps[neighbor] - overlaps[column];
          if (difference > 0 || (difference == 0 && activeColumnsDense[neighbor])) {
            numBigger++;
          }
	}
      }

      const UInt numActive = (UInt)(0.5f + (density * (numNeighbors + 1)));
      if (numBigger < numActive) {
        activeColumns.push_back(column);
        activeColumnsDense[column] = true;
      }
  }
}


bool SpatialPooler::isUpdateRound_() const {
  return (iterationNum_ % updatePeriod_) == 0;
}


UInt SpatialPooler::persistentSize() const {
  // TODO: this won't scale!
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  this->save(s);
  return s.str().size();
}


void SpatialPooler::save(ostream &outStream) const {
  // Write a starting marker and version.
  outStream << std::setprecision(std::numeric_limits<Real>::max_digits10);
  outStream << "SpatialPooler" << endl;
  outStream << version_ << endl;

  // Store the simple variables first.
  outStream << numInputs_ << " " << numColumns_ << " " << potentialRadius_
            << " ";

  outStream << potentialPct_ << " ";
  outStream << initConnectedPct_ << " " << globalInhibition_ << " " 
	  << numActiveColumnsPerInhArea_ << " " << localAreaDensity_ << " ";

  outStream << stimulusThreshold_ << " " << inhibitionRadius_ << " "
            << dutyCyclePeriod_ << " ";

  outStream << boostStrength_ << " ";

  outStream << iterationNum_ << " " << iterationLearnNum_ << " " << spVerbosity_
            << " " << updatePeriod_ << " ";

  outStream << synPermMin_ << " " << synPermMax_ << " " 
    << synPermTrimThreshold_ << " " << synPermInactiveDec_ << " "
    << synPermActiveInc_ << " " << synPermBelowStimulusInc_ << " "
    << synPermConnected_ << " " << minPctOverlapDutyCycles_ << " ";

  outStream << wrapAround_ << " " << endl;

  // Store vectors.
  outStream << inputDimensions_.size() << " ";
  for (auto &elem : inputDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << columnDimensions_.size() << " ";
  for (auto &elem : columnDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  for (UInt i = 0; i < numColumns_; i++) {
    outStream << boostFactors_[i] << " ";
  }
  outStream << endl;

  for (UInt i = 0; i < numColumns_; i++) {
    outStream << overlapDutyCycles_[i] << " ";
  }
  outStream << endl;

  for (UInt i = 0; i < numColumns_; i++) {
    outStream <<  activeDutyCycles_[i] << " ";
  }
  outStream << endl;

  for (UInt i = 0; i < numColumns_; i++) {
    outStream << minOverlapDutyCycles_[i] << " ";
  }
  outStream << endl;

  for (UInt i = 0; i < numColumns_; i++) {
    outStream << tieBreaker_[i] << " ";
  }
  outStream << endl;

  // Store matrices.
  for (UInt i = 0; i < numColumns_; i++) {
    vector<UInt> pot;
    pot.resize(potentialPools_.nNonZerosOnRow(i));
    pot = potentialPools_.getSparseRow(i);
    outStream << pot.size() << endl;
    for (auto &elem : pot) {
      outStream << elem << " ";
    }
    outStream << endl;
  }
  outStream << endl;

  for (UInt i = 0; i < numColumns_; i++) {
    vector<pair<UInt, Real>> perm;
    perm.resize(permanences_.nNonZerosOnRow(i));
    outStream << perm.size() << endl;
    permanences_.getRowToSparse(i, perm.begin());
    for (auto &elem : perm) {
      outStream << elem.first << " " << elem.second << " ";
    }
    outStream << endl;
  }
  outStream << endl;

  outStream << rng_ << endl;

  outStream << "~SpatialPooler" << endl;
}

// Implementation note: this method sets up the instance using data from
// inStream. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void SpatialPooler::load(istream &inStream) {
  // Current version
  version_ = 2;

  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "SpatialPooler");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version == version_);

  // Retrieve simple variables
  inStream >> numInputs_ >> numColumns_ >> potentialRadius_ >> potentialPct_ >>
      initConnectedPct_ >> globalInhibition_ >> numActiveColumnsPerInhArea_ >>
      localAreaDensity_ >> stimulusThreshold_ >> inhibitionRadius_ >>
      dutyCyclePeriod_ >> boostStrength_ >> iterationNum_ >>
      iterationLearnNum_ >> spVerbosity_ >> updatePeriod_

      >> synPermMin_ >> synPermMax_ >> synPermTrimThreshold_ >>
      synPermInactiveDec_ >> synPermActiveInc_ >> synPermBelowStimulusInc_ >>
      synPermConnected_ >> minPctOverlapDutyCycles_;
  inStream >> wrapAround_;

  // Retrieve vectors.
  UInt numInputDimensions;
  inStream >> numInputDimensions;
  inputDimensions_.resize(numInputDimensions);
  for (UInt i = 0; i < numInputDimensions; i++) {
    inStream >> inputDimensions_[i];
  }

  UInt numColumnDimensions;
  inStream >> numColumnDimensions;
  columnDimensions_.resize(numColumnDimensions);
  for (UInt i = 0; i < numColumnDimensions; i++) {
    inStream >> columnDimensions_[i];
  }

  boostFactors_.resize(numColumns_);
  for (UInt i = 0; i < numColumns_; i++) {
    inStream >> boostFactors_[i];
  }

  overlapDutyCycles_.resize(numColumns_);
  for (UInt i = 0; i < numColumns_; i++) {
    inStream >> overlapDutyCycles_[i];
  }

  activeDutyCycles_.resize(numColumns_);
  for (UInt i = 0; i < numColumns_; i++) {
    inStream >> activeDutyCycles_[i];
  }

  minOverlapDutyCycles_.resize(numColumns_);
  for (UInt i = 0; i < numColumns_; i++) {
    inStream >> minOverlapDutyCycles_[i];
  }

  tieBreaker_.resize(numColumns_);
  for (UInt i = 0; i < numColumns_; i++) {
    inStream >> tieBreaker_[i];
  }

  // Retrieve matrices.
  potentialPools_.resize(numColumns_, numInputs_);
  for (UInt i = 0; i < numColumns_; i++) {
    UInt nNonZerosOnRow;
    inStream >> nNonZerosOnRow;
    vector<UInt> pot(nNonZerosOnRow, 0);
    for (UInt j = 0; j < nNonZerosOnRow; j++) {
      inStream >> pot[j];
    }
    potentialPools_.replaceSparseRow(i, pot.begin(), pot.end());
  }

  permanences_.resize(numColumns_, numInputs_);
  connectedSynapses_.resize(numColumns_, numInputs_);
  connectedCounts_.resize(numColumns_);
  for (UInt i = 0; i < numColumns_; i++) {
    UInt nNonZerosOnRow;
    inStream >> nNonZerosOnRow;
    vector<Real> perm(numInputs_, 0);

    for (UInt j = 0; j < nNonZerosOnRow; j++) {
      UInt index;
      Real value;
      inStream >> index;
      inStream >> value;
      perm[index] = value;
    }
    updatePermanencesForColumn_(perm, i, false);
  }

  inStream >> rng_;

  inStream >> marker;
  NTA_CHECK(marker == "~SpatialPooler");

  // initialize ephemeral members
  overlaps_.resize(numColumns_);
  overlapsPct_.resize(numColumns_);
  boostedOverlaps_.resize(numColumns_);
}


//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main SP creation parameters
void SpatialPooler::printParameters() const {
  std::cout << "------------CPP SpatialPooler Parameters ------------------\n";
  std::cout
      << "iterationNum                = " << getIterationNum() << std::endl
      << "iterationLearnNum           = " << getIterationLearnNum() << std::endl
      << "numInputs                   = " << getNumInputs() << std::endl
      << "numColumns                  = " << getNumColumns() << std::endl
      << "numActiveColumnsPerInhArea  = " << getNumActiveColumnsPerInhArea()
      << std::endl
      << "potentialPct                = " << getPotentialPct() << std::endl
      << "globalInhibition            = " << getGlobalInhibition() << std::endl
      << "localAreaDensity            = " << getLocalAreaDensity() << std::endl
      << "stimulusThreshold           = " << getStimulusThreshold() << std::endl
      << "synPermActiveInc            = " << getSynPermActiveInc() << std::endl
      << "synPermInactiveDec          = " << getSynPermInactiveDec()
      << std::endl
      << "synPermConnected            = " << getSynPermConnected() << std::endl
      << "minPctOverlapDutyCycles     = " << getMinPctOverlapDutyCycles()
      << std::endl
      << "dutyCyclePeriod             = " << getDutyCyclePeriod() << std::endl
      << "boostStrength               = " << getBoostStrength() << std::endl
      << "spVerbosity                 = " << getSpVerbosity() << std::endl
      << "wrapAround                  = " << getWrapAround() << std::endl
      << "version                     = " << version() << std::endl;
}

void SpatialPooler::printState(vector<UInt> &state) {
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::cout << state[i] << " ";
  }
  std::cout << "]\n";
}

void SpatialPooler::printState(vector<Real> &state) {
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::printf("%6.3f ", state[i]);
  }
  std::cout << "]\n";
}
