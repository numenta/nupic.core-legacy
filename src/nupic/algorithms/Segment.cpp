/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <algorithm> // sort
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include <assert.h>
#include <map>
#include <nupic/math/ArrayAlgo.hpp> // is_in
#include <nupic/math/StlIo.hpp>     // binary_save
#include <nupic/utils/Log.hpp>
#include <nupic/utils/Random.hpp>

#include <nupic/algorithms/Segment.hpp>

using namespace nupic::algorithms::Cells4;

//----------------------------------------------------------------------
/**
 * Utility routine. Given a src cell index, prints synapse as:
 *  [column, cell within col]
 */
void printSynapse(UInt srcCellIdx, UInt nCellsPerCol) {
  UInt col = (UInt)(srcCellIdx / nCellsPerCol);
  UInt cell = srcCellIdx - col * nCellsPerCol;
  std::cout << "[" << col << "," << cell << "]  ";
}

//----------------------------------------------------------------------
Segment::Segment(InSynapses _s, Real frequency, bool seqSegFlag,
                 Real permConnected, UInt iteration)
    : _totalActivations(1), _positiveActivations(1), _lastActiveIteration(0),
      _lastPosDutyCycle(1.0 / iteration), _lastPosDutyCycleIteration(iteration),
      _seqSegFlag(seqSegFlag), _frequency(frequency), _synapses(std::move(_s)),
      _nConnected(0) {
  for (UInt i = 0; i != _synapses.size(); ++i)
    if (_synapses[i].permanence() >= permConnected)
      ++_nConnected;

  std::sort(_synapses.begin(), _synapses.end(), InSynapseOrder());
  NTA_ASSERT(invariants());
}

//--------------------------------------------------------------------------------
Segment &Segment::operator=(const Segment &o) {
  if (&o != this) {
    _seqSegFlag = o._seqSegFlag;
    _frequency = o._frequency;
    _synapses = o._synapses;
    _nConnected = o._nConnected;
    _totalActivations = o._totalActivations;
    _positiveActivations = o._positiveActivations;
    _lastActiveIteration = o._lastActiveIteration;
    _lastPosDutyCycle = o._lastPosDutyCycle;
    _lastPosDutyCycleIteration = o._lastPosDutyCycleIteration;
  }
  NTA_ASSERT(invariants());
  return *this;
}

//--------------------------------------------------------------------------------
bool Segment::operator==(const Segment &other) const {
  if (_totalActivations != other._totalActivations ||
      _positiveActivations != other._positiveActivations ||
      _lastActiveIteration != other._lastActiveIteration ||
      _lastPosDutyCycle != other._lastPosDutyCycle ||
      _lastPosDutyCycleIteration != other._lastPosDutyCycleIteration ||
      _seqSegFlag != other._seqSegFlag || _frequency != other._frequency ||
      _nConnected != other._nConnected) {
    return false;
  }
  return _synapses == other._synapses;
}

//--------------------------------------------------------------------------------
Segment::Segment(const Segment &o)
    : _totalActivations(o._totalActivations),
      _positiveActivations(o._positiveActivations),
      _lastActiveIteration(o._lastActiveIteration),
      _lastPosDutyCycle(o._lastPosDutyCycle),
      _lastPosDutyCycleIteration(o._lastPosDutyCycleIteration),
      _seqSegFlag(o._seqSegFlag), _frequency(o._frequency),
      _synapses(o._synapses), _nConnected(o._nConnected) {
  NTA_ASSERT(invariants());
}

bool Segment::isActive(const CState &activities, Real permConnected,
                       UInt activationThreshold) const {
  { NTA_ASSERT(invariants()); }

  UInt activity = 0;

  if (_nConnected < activationThreshold)
    return false;

  // TODO: maintain nPermConnected incrementally??
  for (UInt i = 0; i != size() && activity < activationThreshold; ++i)
    if (_synapses[i].permanence() >= permConnected &&
        activities.isSet(_synapses[i].srcCellIdx()))
      activity++;

  return activity >= activationThreshold;
}

//----------------------------------------------------------------------
/**
 * Compute/update and return the positive activations duty cycle of
 * this segment. This is a measure of how often this segment is
 * providing good predictions.
 *
 */
Real Segment::dutyCycle(UInt iteration, bool active, bool readOnly) {
  { NTA_ASSERT(iteration > 0); }

  Real dutyCycle = 0.0;

  // For tier 0, compute it from total number of positive activations seen
  if (iteration <= _dutyCycleTiers[1]) {
    dutyCycle = ((Real)_positiveActivations) / iteration;
    if (!readOnly) {
      _lastPosDutyCycleIteration = iteration;
      _lastPosDutyCycle = dutyCycle;
    }
    return dutyCycle;
  }

  // How old is our update?
  UInt age = iteration - _lastPosDutyCycleIteration;

  // If it's already up to date we can return our cached value
  if (age == 0 && !active)
    return _lastPosDutyCycle;

  // Figure out which alpha we're using
  Real alpha = 0;
  for (UInt tierIdx = _numTiers - 1; tierIdx > 0; tierIdx--) {
    if (iteration > _dutyCycleTiers[tierIdx]) {
      alpha = _dutyCycleAlphas[tierIdx];
      break;
    }
  }

  // Update duty cycle
  dutyCycle = pow((Real64)(1.0 - alpha), (Real64)age) * _lastPosDutyCycle;
  if (active)
    dutyCycle += alpha;

  // Update the time we computed it
  if (!readOnly) {
    _lastPosDutyCycle = dutyCycle;
    _lastPosDutyCycleIteration = iteration;
  }

  return dutyCycle;
}

UInt Segment::computeActivity(const CState &activities, Real permConnected,
                              bool connectedSynapsesOnly) const

{
  { NTA_ASSERT(invariants()); }

  UInt activity = 0;

  if (connectedSynapsesOnly) {
    for (UInt i = 0; i != size(); ++i)
      if (activities.isSet(_synapses[i].srcCellIdx()) &&
          (_synapses[i].permanence() >= permConnected))
        activity++;
  } else {
    for (UInt i = 0; i != size(); ++i)
      if (activities.isSet(_synapses[i].srcCellIdx()))
        activity++;
  }

  return activity;
}

void Segment::addSynapses(const std::set<UInt> &srcCells, Real initStrength,
                          Real permConnected) {
  auto srcCellIdx = srcCells.begin();

  for (; srcCellIdx != srcCells.end(); ++srcCellIdx) {
    _synapses.push_back(InSynapse(*srcCellIdx, initStrength));
    if (initStrength >= permConnected)
      ++_nConnected;
  }

  sort(_synapses, InSynapseOrder());
  NTA_ASSERT(invariants()); // will catch non-unique synapses
}

void Segment::decaySynapses(Real decay, std::vector<UInt> &removed,
                            Real permConnected, bool doDecay) {
  NTA_ASSERT(invariants());

  if (_synapses.empty())
    return;

  static std::vector<UInt> del;
  del.clear(); // purge residual data

  for (UInt i = 0; i != _synapses.size(); ++i) {

    int wasConnected = (int)(_synapses[i].permanence() >= permConnected);

    if (_synapses[i].permanence() < decay) {

      removed.push_back(_synapses[i].srcCellIdx());
      del.push_back(i);

    } else if (doDecay) {
      _synapses[i].permanence() -= decay;
    }

    int isConnected = (int)(_synapses[i].permanence() >= permConnected);

    _nConnected += isConnected - wasConnected;
  }

  _removeSynapses(del);

  NTA_ASSERT(invariants());
}

//--------------------------------------------------------------------------------
/**
 * Subtract decay from each synapses' permanence value.
 * Synapses whose permanence drops below 0 are removed and their indices
 * are inserted into the "removed" list.
 *
 */
void Segment::decaySynapses2(Real decay, std::vector<UInt> &removed,
                             Real permConnected) {
  NTA_ASSERT(invariants());

  if (_synapses.empty())
    return;

  static std::vector<UInt> del;
  del.clear(); // purge residual data

  for (UInt i = 0; i != _synapses.size(); ++i) {

    // Remove synapse whose permanence will go to zero or below.
    if (_synapses[i].permanence() <= decay) {

      // If it was connected, reduce our connected count
      if (_synapses[i].permanence() >= permConnected)
        _nConnected--;

      // Add this synapse to list of synapses to be removed
      removed.push_back(_synapses[i].srcCellIdx());
      del.push_back(i);

    } else {

      _synapses[i].permanence() -= decay;

      // If it was connected and is now below permanence, reduce connected count
      if ((_synapses[i].permanence() + decay >= permConnected) &&
          (_synapses[i].permanence() < permConnected))
        _nConnected--;
    }
  }

  _removeSynapses(del);

  NTA_ASSERT(invariants());
}

//-----------------------------------------------------------------------
/**
 * Sort order for InSynapse's. Cells are sorted in order of increasing
 * permanence.
 *
 */
struct InPermanenceOrder {
  inline bool operator()(const InSynapse &a, const InSynapse &b) const {
    return a.permanence() < b.permanence();
  }
};

//-----------------------------------------------------------------------
/**
 * Sort order for list of source cell indices. Cells are sorted in order of
 * increasing source cell index.
 *
 */
struct InSrcCellOrder {
  inline bool operator()(const UInt a, const UInt b) const { return a < b; }
};

//----------------------------------------------------------------------
/**
 * Free up some synapses in this segment. We always free up inactive
 * synapses (lowest permanence freed up first) before we start to free
 * up active ones.
 */
void Segment::freeNSynapses(UInt numToFree,
                            std::vector<UInt> &inactiveSynapseIndices,
                            std::vector<UInt> &inactiveSegmentIndices,
                            std::vector<UInt> &activeSynapseIndices,
                            std::vector<UInt> &activeSegmentIndices,
                            std::vector<UInt> &removed, UInt verbosity,
                            UInt nCellsPerCol, Real permMax) {
  NTA_CHECK(inactiveSegmentIndices.size() == inactiveSynapseIndices.size());
  NTA_CHECK(activeSegmentIndices.size() == activeSynapseIndices.size());
  NTA_ASSERT(numToFree <= _synapses.size());
  NTA_ASSERT(numToFree <=
             (inactiveSegmentIndices.size() + activeSegmentIndices.size()));

  if (verbosity >= 4) {
    std::cout << "\nIn CPP freeNSynapses with numToFree = " << numToFree
              << ", inactiveSynapses = ";
    for (auto &inactiveSynapseIndice : inactiveSynapseIndices) {
      printSynapse(inactiveSynapseIndice, nCellsPerCol);
    }
    std::cout << "\n";
  }

  //----------------------------------------------------------------------
  // Collect candidate synapses for deletion

  // We first choose from inactive synapses, in order of increasing permanence
  InSynapses candidates;
  for (UInt i = 0; i < inactiveSegmentIndices.size(); i++) {
    // Put in *segment indices*, not source cell indices
    candidates.push_back(
        InSynapse(inactiveSegmentIndices[i],
                  _synapses[inactiveSegmentIndices[i]].permanence()));
  }

  // If we need more, choose from active synapses in order of increasing
  // permanence values. This set has lower priority than inactive synapses
  // so we add a constant permanence value for sorting purposes
  if (candidates.size() < numToFree) {
    for (UInt i = 0; i < activeSegmentIndices.size(); i++) {
      // Put in *segment indices*, not source cell indices
      candidates.push_back(
          InSynapse(activeSegmentIndices[i],
                    _synapses[activeSegmentIndices[i]].permanence() + permMax));
    }
  }

  // Now sort the list of candidate synapses
  std::stable_sort(candidates.begin(), candidates.end(), InPermanenceOrder());

  //----------------------------------------------------------------------
  // Create the final list of synapses we will remove
  static std::vector<UInt> del;
  del.clear(); // purge residual data
  for (UInt i = 0; i < numToFree; i++) {
    del.push_back(candidates[i].srcCellIdx());
    UInt cellIdx = _synapses[candidates[i].srcCellIdx()].srcCellIdx();
    removed.push_back(cellIdx);
  }

  // Debug statements
  if (verbosity >= 4) {
    std::cout << "Removing these synapses: ";
    for (auto &elem : removed) {
      printSynapse(elem, nCellsPerCol);
    }
    std::cout << "\n";

    std::cout << "Segment BEFORE remove synapses: ";
    print(std::cout, nCellsPerCol);
    std::cout << "\n";
  }

  //----------------------------------------------------------------------
  // Remove the synapses
  if (numToFree > 0) {
    std::sort(del.begin(), del.end(), InSrcCellOrder());
    _removeSynapses(del);
  }

  // Debug statements
  if (verbosity >= 4) {
    std::cout << "Segment AFTER remove synapses: ";
    print(std::cout, nCellsPerCol);
    std::cout << "\n";
  }
}

void Segment::print(std::ostream &outStream, UInt nCellsPerCol) const {
  outStream << (_seqSegFlag ? "True " : "False ") << "dc"
            << std::setprecision(4) << _lastPosDutyCycle << " ("
            << _positiveActivations << "/" << _totalActivations << ") ";
  for (UInt i = 0; i != _synapses.size(); ++i) {
    if (nCellsPerCol > 0) {
      UInt cellIdx = _synapses[i].srcCellIdx();
      UInt col = (UInt)(cellIdx / nCellsPerCol);
      UInt cell = cellIdx - col * nCellsPerCol;
      outStream << "[" << col << "," << cell << "]" << std::setprecision(4)
                << _synapses[i].permanence() << " ";
    } else {
      outStream << _synapses[i];
    }
    if (i < _synapses.size() - 1)
      std::cout << " ";
  }
}

namespace nupic {
namespace algorithms {
namespace Cells4 {

std::ostream &operator<<(std::ostream &outStream, const Segment &seg) {
  seg.print(outStream);
  return outStream;
}

std::ostream &operator<<(std::ostream &outStream, const CState &cstate) {
  cstate.print(outStream);
  return outStream;
}

std::ostream &operator<<(std::ostream &outStream, const CStateIndexed &cstate) {
  cstate.print(outStream);
  return outStream;
}
} // namespace Cells4
} // namespace algorithms
} // namespace nupic
