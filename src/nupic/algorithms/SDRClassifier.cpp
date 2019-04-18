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

#include <cmath> // exp
#include <numeric> // accumulate

#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;
using nupic::sdr::SDR;
using namespace nupic::algorithms::sdr_classifier;
using namespace std;

UInt nupic::algorithms::sdr_classifier::argmax( const PDF & data )
  { return max_element( data.begin(), data.end() ) - data.begin(); }


/******************************************************************************/

Classifier::Classifier(const Real alpha)
  { initialize( alpha ); }

void Classifier::initialize(const Real alpha)
{
  NTA_CHECK(alpha > 0.0f);
  alpha_ = alpha;
  dimensions_.clear();
  numCategories_ = 0u;
  weights_.clear();
}


PDF Classifier::infer(const SDR & pattern)
{
  // Check input dimensions, or if this is the first time the Classifier has
  // been used then initialize it with the given SDR's dimensions.
  if( dimensions_.empty() ) {
    dimensions_ = pattern.dimensions;
    while( weights_.size() < pattern.size ) {
      weights_.push_back( vector<Real>( numCategories_, 0.0f ));
    }
  } else if( pattern.dimensions != dimensions_ ) {
      stringstream err_msg;
      err_msg << "Classifier input SDR.dimensions mismatch: previously given SDR with dimensions ( ";
      for( auto dim : dimensions_ )
        { err_msg << dim << " "; }
      err_msg << "), now given SDR with dimensions ( ";
      for( auto dim : pattern.dimensions )
        { err_msg << dim << " "; }
      err_msg << ").";
      NTA_THROW << err_msg.str();
  }

  // Accumulate feed forward input.
  PDF probabilities( numCategories_, 0.0f );
  for( const auto bit : pattern.getSparse() ) {
    for( size_t i = 0; i < numCategories_; i++ ) {
      probabilities[i] += weights_[bit][i];
    }
  }

  // Convert from accumulated votes to probability density function.
  softmax( probabilities.begin(), probabilities.end() );
  return probabilities;
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
  const vector<Real> error = calculateError_(categoryIdxList, pattern);
  for( const auto& bit : pattern.getSparse() ) {
    for(size_t i = 0; i < numCategories_; i++) {
      weights_[bit][i] += alpha_ * error[i];
    }
  }
}


// Helper function to compute the error signal in learning.
std::vector<Real> Classifier::calculateError_(
                    const std::vector<UInt> &categoryIdxList, const SDR &pattern)
{
  // compute predicted likelihoods
  auto likelihoods = infer(pattern);

  // Compute target likelihoods
  PDF targetDistribution(numCategories_ + 1, 0.0);
  for( size_t i = 0; i < categoryIdxList.size(); i++ ) {
    targetDistribution[categoryIdxList[i]] = 1.0 / categoryIdxList.size();
  }

  for( size_t i = 0; i < likelihoods.size(); i++ ) {
    likelihoods[i] = targetDistribution[i] - likelihoods[i];
  }
  return likelihoods;
}


void nupic::algorithms::sdr_classifier::softmax(PDF::iterator begin, PDF::iterator end) {
  if( begin == end ) {
    return;
  }
  const auto maxVal = *max_element(begin, end);
  for (auto itr = begin; itr != end; ++itr) {
    *itr = std::exp(*itr - maxVal); // x[i] = e ^ (x[i] - maxVal)
  }
  // Sum of all elements raised to exp(elem) each.
  const Real sum = std::accumulate(begin, end, 0.0);
  NTA_ASSERT(sum > 0.0);
  for (auto itr = begin; itr != end; ++itr) {
    *itr /= sum;
  }
}


/******************************************************************************/


Predictor::Predictor(const vector<UInt> &steps, const Real alpha)
  { initialize(steps, alpha); }

void Predictor::initialize(const vector<UInt> &steps, const Real alpha)
{
  NTA_CHECK( not steps.empty() ) << "Required argument steps is empty!";
  steps_ = steps;
  sort(steps_.begin(), steps_.end());

  for( const auto step : steps_ ) {
    classifiers_.emplace( step, alpha );
  }

  reset();
}


void Predictor::reset() {
  patternHistory_.clear();
  recordNumHistory_.clear();
}


Predictions Predictor::infer(const UInt recordNum, const SDR &pattern)
{
  updateHistory_( recordNum, pattern );

  Predictions result;
  for( const auto step : steps_ ) {
    result[step] = classifiers_[step].infer( pattern );
  }
  return result;
}


void Predictor::learn(const UInt recordNum, const SDR &pattern,
                      const std::vector<UInt> &bucketIdxList)
{
  updateHistory_( recordNum, pattern );

  // Iterate through all recently given inputs, starting from the furthest in the past.
  auto pastPattern   = patternHistory_.begin();
  auto pastRecordNum = recordNumHistory_.begin();
  for( ; pastRecordNum != recordNumHistory_.end(); pastPattern++, pastRecordNum++ )
  {
    const UInt nSteps = recordNum - *pastRecordNum;

    // Update weights.
    if( binary_search( steps_.begin(), steps_.end(), nSteps )) {
      classifiers_[nSteps].learn( *pastPattern, bucketIdxList );
    }
  }
}


void Predictor::updateHistory_(const UInt recordNum, const SDR & pattern)
{
  // Ensure that recordNum increases monotonically.
  UInt lastRecordNum = -1;
  if( not recordNumHistory_.empty() ) {
    lastRecordNum = recordNumHistory_.back();
    if (recordNum < lastRecordNum) {
      NTA_THROW << "The record number must increase monotonically.";
    }
  }

  // Update pattern history if this is a new record.
  if (recordNumHistory_.size() == 0 || recordNum > lastRecordNum) {
    patternHistory_.emplace_back( pattern );
    recordNumHistory_.push_back(recordNum);
    if (patternHistory_.size() > steps_.back() + 1) {
      patternHistory_.pop_front();
      recordNumHistory_.pop_front();
    }
  } 
}
