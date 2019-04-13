/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.
 *               2019, David McDougall
 *
 * Unless you have an agreement with Numenta, Inc., for a separate license for
 * this software code, the following terms and conditions apply:
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
 * --------------------------------------------------------------------- */

// TODO: Remove unused includes, string?
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
#include <algorithm> // sort

#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;
using namespace nupic::algorithms::sdr_classifier;
using namespace std;


Classifier::Classifier(Real64 alpha)
  { initialize( alpha ); }

void Classifier::initialize(Real64 alpha)
{
  NTA_ASSERT(alpha > 0.0f);
  this->alpha = alpha;
  dimensions_.clear();
  numCategories_ = 0u;
  weights_.clear();
}


void Classifier::infer(const SDR & pattern, PDF & probabilities)
{
  // Check input dimensions, or if this is the first time the Classifier has
  // been used then initialize it with the given SDR's dimensions.
  if( dimensions_.empty() ) {
    dimensions_ = pattern.dimensions;
    while( weights_.size() < pattern.size ) {
      weights_.push_back( vector<Real>( numCategories_, 0.0f ));
    }
  } else {
    NTA_CHECK( pattern.dimensions == dimensions_ );
  }

  // Accumulate feed forward input.
  probabilities.assign( numCategories_, 0.0f );

  for( const auto& bit : pattern.getSparse() ) {
    for( size_t i = 0; i < numCategories_; i++ ) {
      probabilities[i] += weights_[bit][i];
    }
  }
  // Convert from accumulated votes to probability density function.
  softmax( probabilities.begin(), probabilities.end() );
}


UInt Classifier::inferCategory(const SDR & pattern) {
  PDF data;
  infer( pattern, data );
  return max_element( data.begin(), data.end() ) - data.begin();
}


void Classifier::learn(const SDR &pattern, const vector<UInt> &categoryIdxList)
{
  // Check if this is a new category & resize the weights table to hold it.
  const auto maxCategoryIdx = *max_element(categoryIdxList.begin(), categoryIdxList.end());
  if( maxCategoryIdx >= numCategories_ ) {
    numCategories_ = maxCategoryIdx + 1;
    for( auto & vec : weights_ ) {
      while( vec.size() < numCategories_ ) {
        vec.push_back( 0.0f );
      }
    }
  }

  // Compute errors and update weights.
  const vector<Real64> error = calculateError_(categoryIdxList, pattern);
  for( const auto& bit : pattern.getSparse() ) {
    for(size_t i = 0; i < numCategories_; i++) {
      weights_[bit][i] += alpha * error[i];
    }
  }
}


// Helper function to compute the error signal in learning.
std::vector<Real64> Classifier::calculateError_(
                    const std::vector<UInt> &categoryIdxList, const SDR &pattern)
{
  // compute predicted likelihoods
  vector<Real64> likelihoods;
  infer(pattern, likelihoods);

  // Compute target likelihoods
  vector<Real64> targetDistribution(numCategories_ + 1, 0.0);
  for( size_t i = 0; i < categoryIdxList.size(); i++ ) {
    targetDistribution[categoryIdxList[i]] = 1.0 / categoryIdxList.size();
  }

  for( size_t i = 0; i < likelihoods.size(); i++ ) {
    likelihoods[i] = targetDistribution[i] - likelihoods[i];
  }
  return likelihoods;
}


void nupic::algorithms::sdr_classifier::softmax(PDF::iterator begin, PDF::iterator end) {
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


void Classifier::save(std::ostream &outStream) const {}


void Classifier::load(std::istream &inStream) {}


/******************************************************************************/

Predictor::Predictor(const vector<UInt> &steps, Real64 alpha, Real64 actValueAlpha)
  { initialize(steps, alpha, actValueAlpha); }

void Predictor::initialize(const vector<UInt> &steps, Real64 alpha,
                             Real64 actValueAlpha) {
  steps_ = steps;
  actValueAlpha_ = actValueAlpha;
  maxInputIdx_ = 0;
  maxBucketIdx_ = 0; 
  actualValues_ = {0.0};
  actualValuesSet_ = {false}; 

  sort(steps_.begin(), steps_.end());

  if (steps_.size() > 0) {
    maxSteps_ = steps_.at(steps_.size() - 1) + 1;
  } else {
    maxSteps_ = 1;
  }

  for (const auto step : steps_) {
    classifiers_.emplace( step, alpha );
  }
}

Predictor::~Predictor() {}

void Predictor::compute(UInt recordNum, const SDR & pattern,
                            const vector<UInt> &bucketIdxList,
                            const vector<Real64> &actValueList, bool category,
                            bool learn, bool infer, ClassifierResult &result) {
  // Ensure that recordNum increases monotonically.
  UInt lastRecordNum = -1;
  if (recordNumHistory_.size() > 0) {
    lastRecordNum = recordNumHistory_.back();
    if (recordNum < lastRecordNum) {
      NTA_THROW << "the record number has to increase monotonically";
    }
  }

  // Update pattern history if this is a new record.
  if (recordNumHistory_.size() == 0 || recordNum > lastRecordNum) {
    const auto &patternNZ = pattern.getSparse();
    patternNZHistory_.emplace_back(patternNZ.begin(), patternNZ.end());
    recordNumHistory_.push_back(recordNum);
    if (patternNZHistory_.size() > maxSteps_) {
      patternNZHistory_.pop_front();
      recordNumHistory_.pop_front();
    }
  }

  // If in inference mode, compute likelihood and update return value
  if (infer) {
    infer_(pattern, actValueList, result);
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
        classifiers_[nSteps].learn( pattern, bucketIdxList );
      }
    }
  }
}


void Predictor::infer_(const SDR &pattern,
                       const vector<Real64> &actValue,
                       ClassifierResult &result)
{
  // Add the actual values to the return value. For buckets that haven't
  // been seen yet, the actual value doesn't matter since it will have
  // zero likelihood.
  vector<Real64> &actValueVector = result[ACTUAL_VALUES];
  actValueVector.reserve( actualValues_.size() );

  for( size_t i = 0; i < actualValues_.size(); ++i ) {
    if (actualValuesSet_[i]) {
      actValueVector.push_back( actualValues_[i] );
    }
    else {
      // if doing 0-step ahead prediction, we shouldn't use any
      // knowledge of the classification input during inference
      if( steps_.at(0) == 0 ) {
        actValueVector.push_back( 0.0f );
      }
      else {
        actValueVector.push_back( actValue[0] );
      }
    }
  }

  for( const auto step : steps_ ) {
    classifiers_[step].infer( pattern, result[ step ] );
  }
}


void Predictor::save(ostream &outStream) const {
 //  // Write a starting marker and version.
 //  outStream << "Predictor" << endl;

 //  // Store the simple variables first.
 //  outStream << actValueAlpha_ << " "
 //            << maxSteps_ << " " << maxBucketIdx_ << " " << maxInputIdx_ << " "
 //            << " " << endl;

 //  // V1 additions.
 //  outStream << recordNumHistory_.size() << " ";
 //  for (const auto &elem : recordNumHistory_) {
 //    outStream << elem << " ";
 //  }
 //  outStream << endl;

 //  // Store the different prediction steps.
 //  outStream << steps_.size() << " ";
 //  for (auto &elem : steps_) {
 //    outStream << elem << " ";
 //  }
 //  outStream << endl;

 //  // Store the pattern history.
 //  outStream << patternNZHistory_.size() << " ";
 //  for (auto &pattern : patternNZHistory_) {
 //    outStream << pattern.size() << " ";
 //    for (auto &pattern_j : pattern) {
 //      outStream << pattern_j << " ";
 //    }
 //  }
 //  outStream << endl;

 //  // Store weight matrix
 //  outStream << weightMatrix_.size() << " ";
 //  for (const auto &elem : weightMatrix_) { // elem = Matrix
 //    outStream << elem.first << " ";
 //    const auto map2d =  elem.second; //2d map: map<int<map<int, double>>
 //    for(UInt i=0; i < maxInputIdx_; i++) {
 //      for(UInt j=0; j< maxBucketIdx_; j++) {
	// const Real64 val = get_(map2d, i, j); //store dense map because indices i,j have to match in load()
 //        outStream << val << " "; //map2d[i][j]
 //      }
 //    }
 //  }
 //  outStream << endl;

 //  // Store the actual values for each bucket.
 //  outStream << actualValues_.size() << " ";
 //  for (UInt i = 0; i < actualValues_.size(); ++i) {
 //    outStream << actualValues_[i] << " ";
 //    outStream << actualValuesSet_[i] << " ";
 //  }
 //  outStream << endl;

 //  // Write an ending marker.
 //  outStream << "~Predictor" << endl;
}

void Predictor::load(istream &inStream) {
 //  // Clean up the existing data structures before loading
 //  steps_.clear();
 //  recordNumHistory_.clear();
 //  patternNZHistory_.clear();
 //  actualValues_.clear();
 //  actualValuesSet_.clear();
 //  weightMatrix_.clear();

 //  // Check the starting marker.
 //  string marker;
 //  inStream >> marker;
 //  NTA_CHECK(marker == "Predictor");

 //  // Check the version.
 //  UInt version;
 //  inStream >> version;
 //  NTA_CHECK(version == 2);

 //  // Load the simple variables.
 //  inStream >> actValueAlpha_ >> maxSteps_ >>
 //      maxBucketIdx_ >> maxInputIdx_;

 //  UInt recordNumHistory;
 //  UInt curRecordNum;
 //  inStream >> recordNumHistory;
 //  for (UInt i = 0; i < recordNumHistory; ++i) {
 //      inStream >> curRecordNum;
 //      recordNumHistory_.push_back(curRecordNum);
 //  }

 //  // Load the prediction steps.
 //  UInt size;
 //  UInt step;
 //  inStream >> size;
 //  for (UInt i = 0; i < size; ++i) {
 //    inStream >> step;
 //    steps_.push_back(step);
 //  }

 //  // Load the input pattern history.
 //  inStream >> size;
 //  UInt vSize;
 //  for (UInt i = 0; i < size; ++i) {
 //    inStream >> vSize;
 //    patternNZHistory_.emplace_back(vSize);
 //    for (UInt j = 0; j < vSize; ++j) {
 //      inStream >> patternNZHistory_[i][j];
 //    }
 //  }

 //  // Load weight matrix.
 //  UInt numSteps;
 //  inStream >> numSteps;
 //  for (UInt s = 0; s < numSteps; ++s) {
 //    inStream >> step;
 //    // Insert the step to initialize the weight matrix
 //    auto m = Matrix();
 //    for (UInt i = 0; i < maxInputIdx_; i++) {
 //      for (UInt j = 0; j < maxBucketIdx_; j++) {
	// Real64 val;
 //        inStream >> val;
	// if(val == 0.0) continue; //load sparse map only
	// m[i][j] = val;
 //      }
 //    }
 //    weightMatrix_[step] = m;
 //  }

 //  // Load the actual values for each bucket.
 //  UInt numBuckets;
 //  Real64 actualValue;
 //  bool actualValueSet;
 //  inStream >> numBuckets;
 //  for (UInt i = 0; i < numBuckets; ++i) {
 //    inStream >> actualValue;
 //    actualValues_.push_back(actualValue);
 //    inStream >> actualValueSet;
 //    actualValuesSet_.push_back(actualValueSet);
 //  }

 //  // Check for the end marker.
 //  inStream >> marker;
 //  NTA_CHECK(marker == "~Predictor");

 //  // Update the version number.
 //  version_ = PredictorVersion;
}

bool Predictor::operator==(const Predictor &other) const {
  // if (steps_.size() != other.steps_.size()) {
  //   return false;
  // }
  // for (UInt i = 0; i < steps_.size(); i++) {
  //   if (steps_.at(i) != other.steps_.at(i)) {
  //     return false;
  //   }
  // }

  // if (fabs(alpha_ - other.alpha_) > 0.000001 ||
  //     fabs(actValueAlpha_ - other.actValueAlpha_) > 0.000001 ||
  //     maxSteps_ != other.maxSteps_) {
  //   return false;
  // }

  // if (patternNZHistory_.size() != other.patternNZHistory_.size()) {
  //   return false;
  // }
  // for (UInt i = 0; i < patternNZHistory_.size(); i++) {
  //   if (patternNZHistory_.at(i).size() !=
  //       other.patternNZHistory_.at(i).size()) {
  //     return false;
  //   }
  //   for (UInt j = 0; j < patternNZHistory_.at(i).size(); j++) {
  //     if (patternNZHistory_.at(i).at(j) !=
  //         other.patternNZHistory_.at(i).at(j)) {
  //       return false;
  //     }
  //   }
  // }

  // if (recordNumHistory_.size() != other.recordNumHistory_.size()) {
  //   return false;
  // }
  // for (UInt i = 0; i < recordNumHistory_.size(); i++) {
  //   if (recordNumHistory_.at(i) != other.recordNumHistory_.at(i)) {
  //     return false;
  //   }
  // }

  // if (maxBucketIdx_ != other.maxBucketIdx_) {
  //   return false;
  // }

  // if (maxInputIdx_ != other.maxInputIdx_) {
  //   return false;
  // }

  // if (weightMatrix_.size() != other.weightMatrix_.size()) {
  //   return false;
  // }
  // for (auto it = weightMatrix_.begin(); it != weightMatrix_.end(); it++) {
  //   const Matrix thisWeights = it->second;
  //   const Matrix otherWeights = other.weightMatrix_.at(it->first);
  //   for (UInt i = 0; i <= maxInputIdx_; ++i) {
  //     for (UInt j = 0; j <= maxBucketIdx_; ++j) {
  //       if (get_(thisWeights, i, j) != get_(otherWeights, i, j)) {
  //         return false;
  //       }
  //     }
  //   }
  // }

  // if (actualValues_.size() != other.actualValues_.size() ||
  //     actualValuesSet_.size() != other.actualValuesSet_.size()) {
  //   return false;
  // }
  // for (UInt i = 0; i < actualValues_.size(); i++) {
  //   if (fabs(actualValues_[i] - other.actualValues_[i]) > 0.000001 ||
  //       actualValuesSet_[i] != other.actualValuesSet_[i]) {
  //     return false;
  //   }
  // }

  // if (version_ != other.version_ || verbosity_ != other.verbosity_) {
  //   return false;
  // }

  return true;
}
