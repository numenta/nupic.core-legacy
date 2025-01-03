/*
 * Copyright 2013-2015 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <cmath>
#include <map>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>

#include <nupic/algorithms/BitHistory.hpp>
#include <nupic/proto/BitHistory.capnp.h>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
namespace algorithms {
namespace cla_classifier {

const Real64 DUTY_CYCLE_UPDATE_INTERVAL = pow(3.2, 32);

BitHistory::BitHistory(UInt bitNum, int nSteps, Real64 alpha, UInt verbosity)
    : lastTotalUpdate_(-1), learnIteration_(0), alpha_(alpha),
      verbosity_(verbosity) {
  stringstream ss;
  ss << bitNum << "[" << nSteps << "]";
  id_ = ss.str();
}

void BitHistory::store(int iteration, int bucketIdx) {
  if (lastTotalUpdate_ == -1) {
    lastTotalUpdate_ = iteration;
  }

  // Get the previous duty cycle, or 0.0 for new buckets.
  map<int, Real64>::const_iterator it = stats_.find(bucketIdx);
  Real64 dc = 0.0;
  if (it != stats_.end()) {
    dc = it->second;
  }

  // Compute the new duty cycle, dcNew, at the iteration that the duty
  // cycles are currently at.
  Real64 denom = pow(1.0 - alpha_, iteration - lastTotalUpdate_);
  Real64 dcNew = -1.0;
  if (denom > 0.0) {
    dcNew = dc + (alpha_ / denom);
  }

  if (denom < 0.00001 || dcNew > DUTY_CYCLE_UPDATE_INTERVAL) {
    // Update all duty cycles to the current iteration.
    Real64 exp = pow(1.0 - alpha_, iteration - lastTotalUpdate_);
    for (map<int, Real64>::const_iterator i = stats_.begin(); i != stats_.end();
         ++i) {
      stats_[i->first] = i->second * exp;
    }

    lastTotalUpdate_ = iteration;

    dc = stats_[bucketIdx] + alpha_;
  } else {
    dc = dcNew;
  }

  // Set the new duty cycle for the specified bucket.
  stats_[bucketIdx] = dc;
}

void BitHistory::infer(int iteration, vector<Real64> *votes) {
  Real64 total = 0.0;
  // Set the vote for each bucket to the duty cycle value.
  for (map<int, Real64>::const_iterator it = stats_.begin(); it != stats_.end();
       ++it) {
    if (it->second > 0.0) {
      (*votes)[it->first] = it->second;
      total += it->second;
    }
  }

  // Normalize the duty cycles.
  if (total > 0.0) {
    for (auto &vote : *votes) {
      vote = vote / total;
    }
  }
}

void BitHistory::save(ostream &outStream) const {
  // Write out a starting marker.
  outStream << "BitHistory" << endl;

  // Save the simple variables.
  outStream << id_ << " " << lastTotalUpdate_ << " " << learnIteration_ << " "
            << alpha_ << " " << verbosity_ << " " << endl;

  // Save the bucket duty cycles.
  outStream << stats_.size() << " ";
  for (const auto &elem : stats_) {
    outStream << elem.first << " " << elem.second << " ";
  }
  outStream << endl;

  // Write out a termination marker.
  outStream << "~BitHistory" << endl;
}

void BitHistory::load(istream &inStream) {
  // Check the starting marker.
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "BitHistory");

  // Load the simple variables.
  inStream >> id_ >> lastTotalUpdate_ >> learnIteration_ >> alpha_ >>
      verbosity_;

  // Load the bucket duty cycles.
  UInt numBuckets;
  int bucketIdx;
  Real64 dutyCycle;
  inStream >> numBuckets;
  for (UInt i = 0; i < numBuckets; ++i) {
    inStream >> bucketIdx >> dutyCycle;
    stats_.insert(pair<int, Real64>(bucketIdx, dutyCycle));
  }

  // Check the termination marker.
  inStream >> marker;
  NTA_CHECK(marker == "~BitHistory");
}
void BitHistory::write(BitHistoryProto::Builder &proto) const {
  proto.setId(id_.c_str());

  auto statsList = proto.initStats(stats_.size());
  UInt i = 0;
  for (const auto &elem : stats_) {
    auto stat = statsList[i];
    stat.setIndex(elem.first);
    stat.setDutyCycle(elem.second);
    i++;
  }

  proto.setLastTotalUpdate(lastTotalUpdate_);
  proto.setLearnIteration(learnIteration_);
  proto.setAlpha(alpha_);
  proto.setVerbosity(verbosity_);
}

void BitHistory::read(BitHistoryProto::Reader &proto) {
  id_ = proto.getId().cStr();

  stats_.clear();
  for (auto stat : proto.getStats()) {
    stats_[stat.getIndex()] = stat.getDutyCycle();
  }

  lastTotalUpdate_ = proto.getLastTotalUpdate();
  learnIteration_ = proto.getLearnIteration();
  alpha_ = proto.getAlpha();
  verbosity_ = proto.getVerbosity();
}

bool BitHistory::operator==(const BitHistory &other) const {
  if (id_ != other.id_ || lastTotalUpdate_ != other.lastTotalUpdate_ ||
      learnIteration_ != other.learnIteration_ ||
      fabs(alpha_ - other.alpha_) > 0.000001 ||
      verbosity_ != other.verbosity_) {
    return false;
  }

  if (stats_.size() != other.stats_.size()) {
    return false;
  }
  for (auto it = stats_.begin(); it != stats_.end(); it++) {
    if (fabs(it->second - other.stats_.at(it->first)) > 0.000001) {
      return false;
    }
  }

  return true;
}

bool BitHistory::operator!=(const BitHistory &other) const {
  return !operator==(other);
}

} // end namespace cla_classifier
} // end namespace algorithms
} // end namespace nupic
