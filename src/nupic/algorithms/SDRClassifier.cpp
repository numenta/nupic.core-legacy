/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

#include <cmath> //exp
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <numeric> //accumulate
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>


#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/utils/Log.hpp>

using namespace std;

namespace nupic {
namespace algorithms {
namespace sdr_classifier {

/**
 * get(x,y) accessor interface for Matrix; handles sparse (missing) values
 * @return return value stored at map[row][col], or defaultVal if such field does not exist
 **/
Real64 get_(const Matrix& m, const UInt row, const UInt col, const Real64 defaultVal=0.0) {
  try {
    return m.at(row).at(col);
  } catch(std::exception& ex ) {
    return defaultVal;
  }
}


SDRClassifier::SDRClassifier(const vector<UInt> &steps, Real64 alpha,
                             Real64 actValueAlpha, UInt verbosity)
    : steps_(steps), alpha_(alpha), actValueAlpha_(actValueAlpha),
      maxInputIdx_(0), maxBucketIdx_(0), actualValues_({0.0}),
      actualValuesSet_({false}), version_(sdrClassifierVersion),
      verbosity_(verbosity) {
  sort(steps_.begin(), steps_.end());
  if (steps_.size() > 0) {
    maxSteps_ = steps_.at(steps_.size() - 1) + 1;
  } else {
    maxSteps_ = 1;
  }

  // TODO: insert maxBucketIdx / maxInputIdx hint as parameter?
  // There can be great overhead reallocating the array every time a new
  // input is seen, especially if we start at (0, 0). The client will
  // usually know what is the final maxInputIdx (typically the number
  // of columns?), and we can have heuristics using the encoder's
  // settings to get an good approximate of the maxBucketIdx, thus having
  // to reallocate this matrix only a few times, even never if we use
  // lower bounds
  for (const auto &step : steps_) {
	  Matrix m;
    weightMatrix_.emplace(step, m);
  }
}

SDRClassifier::~SDRClassifier() {}

void SDRClassifier::compute(UInt recordNum, const vector<UInt> &patternNZ,
                            const vector<UInt> &bucketIdxList,
                            const vector<Real64> &actValueList, bool category,
                            bool learn, bool infer, ClassifierResult *result) {
  // ensures that recordNum increases monotonically
  UInt lastRecordNum = -1;
  if (recordNumHistory_.size() > 0) {
    lastRecordNum = recordNumHistory_[recordNumHistory_.size() - 1];
    if (recordNum < lastRecordNum)
      NTA_THROW << "the record number has to increase monotonically";
  }

  // update pattern history if this is a new record
  if (recordNumHistory_.size() == 0 || recordNum > lastRecordNum) {
    patternNZHistory_.emplace_back(patternNZ.begin(), patternNZ.end());
    recordNumHistory_.push_back(recordNum);
    if (patternNZHistory_.size() > maxSteps_) {
      patternNZHistory_.pop_front();
      recordNumHistory_.pop_front();
    }
  }

  // if input pattern has greater index than previously seen, update
  // maxInputIdx and augment weight matrix with zero padding
  if (patternNZ.size() > 0) {
    const UInt maxInputIdx = *max_element(patternNZ.begin(), patternNZ.end());
    if (maxInputIdx > maxInputIdx_) {
      maxInputIdx_ = maxInputIdx;
    }
  }

  // if in inference mode, compute likelihood and update return value
  if (infer) {
    infer_(patternNZ, actValueList, result);
  }

  // update weights if in learning mode
  if (learn) {
    for (size_t categoryI = 0; categoryI < bucketIdxList.size(); categoryI++) {
      const UInt bucketIdx = bucketIdxList[categoryI];
      const Real64 actValue = actValueList[categoryI];
      // if bucket is greater, update maxBucketIdx_ and augment weight
      // matrix with zero-padding
      if (bucketIdx > maxBucketIdx_) {
        maxBucketIdx_ = bucketIdx;
      }

      // update rolling averages of bucket values
      while (actualValues_.size() <= maxBucketIdx_) {
        actualValues_.push_back(0.0);
        actualValuesSet_.push_back(false);
      }
      if (!actualValuesSet_[bucketIdx] || category) {
        actualValues_[bucketIdx] = actValue;
        actualValuesSet_[bucketIdx] = true;
      } else {
        actualValues_[bucketIdx] =
            ((1.0 - actValueAlpha_) * actualValues_[bucketIdx]) +
            (actValueAlpha_ * actValue);
      }
    }

    // compute errors and update weights
    auto patternIteration = patternNZHistory_.begin();
    for (auto learnRecord = recordNumHistory_.begin();
         learnRecord != recordNumHistory_.end();
         learnRecord++, patternIteration++) {
      const vector<UInt> learnPatternNZ = *patternIteration;
      const UInt nSteps = recordNum - *learnRecord;

      // update weights
      if (binary_search(steps_.begin(), steps_.end(), nSteps)) {
        const vector<Real64> error = calculateError_(bucketIdxList, learnPatternNZ, nSteps);
        Matrix& w = weightMatrix_.at(nSteps);
	NTA_ASSERT(alpha_ > 0.0);
        for (const auto& bit : learnPatternNZ) {
          for(UInt i = 0; i < error.size(); i++) {
            const auto val = get_(w, bit, i) + alpha_ * error[i];
	    if(val == 0) continue;
	    w[bit][i] = val;
          }
        }
      }
    }
  }
}

size_t SDRClassifier::persistentSize() const {
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  save(s);
  return s.str().size();
}

void SDRClassifier::infer_(const vector<UInt> &patternNZ,
                           const vector<Real64> &actValue,
                           ClassifierResult *result) {
  // add the actual values to the return value. For buckets that haven't
  // been seen yet, the actual value doesn't matter since it will have
  // zero likelihood.
  vector<Real64> *actValueVector =
      result->createVector(-1, actualValues_.size(), 0.0);
  for (UInt i = 0; i < actualValues_.size(); ++i) {
    if (actualValuesSet_[i]) {
      (*actValueVector)[i] = actualValues_[i];
    } else {
      // if doing 0-step ahead prediction, we shouldn't use any
      // knowledge of the classification input during inference
      if (steps_.at(0) == 0) {
        (*actValueVector)[i] = 0;
      } else {
        (*actValueVector)[i] = actValue[0];
      }
    }
  }

  for (auto nSteps = steps_.begin(); nSteps != steps_.end(); ++nSteps) {
    vector<Real64>* likelihoods = result->createVector(*nSteps, maxBucketIdx_ + 1, 0.0);
    for (const auto& bit : patternNZ) {
      const Matrix& w = weightMatrix_.at(*nSteps);
      for(Size i =0; i< likelihoods->size(); i++) {
        likelihoods->at(i) += get_(w, bit, i);
      }
    }
    softmax_(likelihoods->begin(), likelihoods->end());
  }
}

vector<Real64> SDRClassifier::calculateError_(const vector<UInt> &bucketIdxList,
                                              const vector<UInt> patternNZ,
                                              UInt step) {
  // compute predicted likelihoods
  vector<Real64> likelihoods(maxBucketIdx_ + 1, 0);

  for (const auto& bit : patternNZ) {
    const Matrix& w = weightMatrix_.at(step); //row
    for(Size i=0; i< likelihoods.size(); i++) {
      likelihoods[i] += get_(w, bit, i);
    }
  }
  softmax_(likelihoods.begin(), likelihoods.end());

  // compute target likelihoods
  vector<Real64> targetDistribution(maxBucketIdx_ + 1, 0.0);
  const Real64 numCategories = (Real64)bucketIdxList.size();
  for (size_t i = 0; i < bucketIdxList.size(); i++)
    targetDistribution[bucketIdxList[i]] = 1.0 / numCategories;

  NTA_ASSERT(likelihoods.size() == targetDistribution.size());
  for(UInt i = 0; i < likelihoods.size(); i++) {
    likelihoods[i] = targetDistribution[i] - likelihoods[i];
  }
  return likelihoods;
}


void SDRClassifier::softmax_(const vector<Real64>::iterator begin,
                             const vector<Real64>::iterator end) {
  const auto maxVal = *max_element(begin, end);
  for (auto itr = begin; itr != end; ++itr) { //TODO use c++17 reduce/transform
    *itr = std::exp(*itr - maxVal); // x[i] = exp(x[i] - maxVal)
  }
  const Real64 sum = std::accumulate(begin, end, 0.0); //sum of all elements raised to exp(elem) each.
  NTA_ASSERT(sum > 0.0);
  for (auto itr = begin; itr != end; ++itr) {
    *itr /= sum;
  }
}

UInt SDRClassifier::version() const { return version_; }

UInt SDRClassifier::getVerbosity() const { return verbosity_; }

void SDRClassifier::setVerbosity(UInt verbosity) { verbosity_ = verbosity; }

Real64 SDRClassifier::getAlpha() const { return alpha_; }

void SDRClassifier::save(ostream &outStream) const {
  // Write a starting marker and version.
  outStream << "SDRClassifier" << endl;
  outStream << version_ << endl;

  // Store the simple variables first.
  outStream << version() << " " << alpha_ << " " << actValueAlpha_ << " "
            << maxSteps_ << " " << maxBucketIdx_ << " " << maxInputIdx_ << " "
            << verbosity_ << " " << endl;

  // V1 additions.
  outStream << recordNumHistory_.size() << " ";
  for (const auto &elem : recordNumHistory_) {
    outStream << elem << " ";
  }
  outStream << endl;

  // Store the different prediction steps.
  outStream << steps_.size() << " ";
  for (auto &elem : steps_) {
    outStream << elem << " ";
  }
  outStream << endl;

  // Store the pattern history.
  outStream << patternNZHistory_.size() << " ";
  for (auto &pattern : patternNZHistory_) {
    outStream << pattern.size() << " ";
    for (auto &pattern_j : pattern) {
      outStream << pattern_j << " ";
    }
  }
  outStream << endl;

  // Store weight matrix
  outStream << weightMatrix_.size() << " ";
  for (const auto &elem : weightMatrix_) { // elem = Matrix
    outStream << elem.first << " ";
    const auto map2d =  elem.second; //2d map: map<int<map<int, double>>
    for(UInt i=0; i < maxInputIdx_; i++) {
      for(UInt j=0; j< maxBucketIdx_; j++) {
	const Real64 val = get_(map2d, i, j); //store dense map because indices i,j have to match in load()
        outStream << val << " "; //map2d[i][j]
      }
    }
  }
  outStream << endl;

  // Store the actual values for each bucket.
  outStream << actualValues_.size() << " ";
  for (UInt i = 0; i < actualValues_.size(); ++i) {
    outStream << actualValues_[i] << " ";
    outStream << actualValuesSet_[i] << " ";
  }
  outStream << endl;

  // Write an ending marker.
  outStream << "~SDRClassifier" << endl;
}


void SDRClassifier::load(istream &inStream) {
  // Clean up the existing data structures before loading
  steps_.clear();
  recordNumHistory_.clear();
  patternNZHistory_.clear();
  actualValues_.clear();
  actualValuesSet_.clear();
  weightMatrix_.clear();

  // Check the starting marker.
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "SDRClassifier");

  // Check the version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version == 2);

  // Load the simple variables.
  inStream >> version_ >> alpha_ >> actValueAlpha_ >> maxSteps_ >>
      maxBucketIdx_ >> maxInputIdx_ >> verbosity_;

  UInt recordNumHistory;
  UInt curRecordNum;
  inStream >> recordNumHistory;
  for (UInt i = 0; i < recordNumHistory; ++i) {
      inStream >> curRecordNum;
      recordNumHistory_.push_back(curRecordNum);
  }

  // Load the prediction steps.
  UInt size;
  UInt step;
  inStream >> size;
  for (UInt i = 0; i < size; ++i) {
    inStream >> step;
    steps_.push_back(step);
  }

  // Load the input pattern history.
  inStream >> size;
  UInt vSize;
  for (UInt i = 0; i < size; ++i) {
    inStream >> vSize;
    patternNZHistory_.emplace_back(vSize);
    for (UInt j = 0; j < vSize; ++j) {
      inStream >> patternNZHistory_[i][j];
    }
  }

  // Load weight matrix.
  UInt numSteps;
  inStream >> numSteps;
  for (UInt s = 0; s < numSteps; ++s) {
    inStream >> step;
    // Insert the step to initialize the weight matrix
    auto m = Matrix();
    for (UInt i = 0; i < maxInputIdx_; i++) {
      for (UInt j = 0; j < maxBucketIdx_; j++) {
	Real64 val;
        inStream >> val;
	if(val == 0.0) continue; //load sparse map only
	m[i][j] = val;
      }
    }
    weightMatrix_[step] = m;
  }

  // Load the actual values for each bucket.
  UInt numBuckets;
  Real64 actualValue;
  bool actualValueSet;
  inStream >> numBuckets;
  for (UInt i = 0; i < numBuckets; ++i) {
    inStream >> actualValue;
    actualValues_.push_back(actualValue);
    inStream >> actualValueSet;
    actualValuesSet_.push_back(actualValueSet);
  }

  // Check for the end marker.
  inStream >> marker;
  NTA_CHECK(marker == "~SDRClassifier");

  // Update the version number.
  version_ = sdrClassifierVersion;
}



bool SDRClassifier::operator==(const SDRClassifier &other) const {
  if (steps_.size() != other.steps_.size()) {
    return false;
  }
  for (UInt i = 0; i < steps_.size(); i++) {
    if (steps_.at(i) != other.steps_.at(i)) {
      return false;
    }
  }

  if (fabs(alpha_ - other.alpha_) > 0.000001 ||
      fabs(actValueAlpha_ - other.actValueAlpha_) > 0.000001 ||
      maxSteps_ != other.maxSteps_) {
    return false;
  }

  if (patternNZHistory_.size() != other.patternNZHistory_.size()) {
    return false;
  }
  for (UInt i = 0; i < patternNZHistory_.size(); i++) {
    if (patternNZHistory_.at(i).size() !=
        other.patternNZHistory_.at(i).size()) {
      return false;
    }
    for (UInt j = 0; j < patternNZHistory_.at(i).size(); j++) {
      if (patternNZHistory_.at(i).at(j) !=
          other.patternNZHistory_.at(i).at(j)) {
        return false;
      }
    }
  }

  if (recordNumHistory_.size() != other.recordNumHistory_.size()) {
    return false;
  }
  for (UInt i = 0; i < recordNumHistory_.size(); i++) {
    if (recordNumHistory_.at(i) != other.recordNumHistory_.at(i)) {
      return false;
    }
  }

  if (maxBucketIdx_ != other.maxBucketIdx_) {
    return false;
  }

  if (maxInputIdx_ != other.maxInputIdx_) {
    return false;
  }

  if (weightMatrix_.size() != other.weightMatrix_.size()) {
    return false;
  }
  for (auto it = weightMatrix_.begin(); it != weightMatrix_.end(); it++) {
    const Matrix thisWeights = it->second;
    const Matrix otherWeights = other.weightMatrix_.at(it->first);
    for (UInt i = 0; i <= maxInputIdx_; ++i) {
      for (UInt j = 0; j <= maxBucketIdx_; ++j) {
        if (get_(thisWeights, i, j) != get_(otherWeights, i, j)) {
          return false;
        }
      }
    }
  }

  if (actualValues_.size() != other.actualValues_.size() ||
      actualValuesSet_.size() != other.actualValuesSet_.size()) {
    return false;
  }
  for (UInt i = 0; i < actualValues_.size(); i++) {
    if (fabs(actualValues_[i] - other.actualValues_[i]) > 0.000001 ||
        actualValuesSet_[i] != other.actualValuesSet_[i]) {
      return false;
    }
  }

  if (version_ != other.version_ || verbosity_ != other.verbosity_) {
    return false;
  }

  return true;
}


} // namespace sdr_classifier
} // namespace algorithms
} // namespace nupic
