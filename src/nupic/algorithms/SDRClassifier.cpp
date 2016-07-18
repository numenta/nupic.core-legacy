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

#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/proto/SdrClassifier.capnp.h>
#include <nupic/math/ArrayAlgo.hpp>
#include <nupic/utils/Log.hpp>

using namespace std;

namespace nupic
{
	namespace algorithms 
	{
	  namespace sdr_classifier
	  {

	    SDRClassifier::SDRClassifier(
	        const vector<UInt>& steps, Real64 alpha, Real64 actValueAlpha,
	        UInt verbosity) : alpha_(alpha), actValueAlpha_(actValueAlpha),
	        learnIteration_(0), recordNumMinusLearnIteration_(0),
	        maxBucketIdx_(0), version_(Version), verbosity_(verbosity),
	        maxInputIdx_(0)
	    {
	      for (const auto& step : steps)
	      {
	        steps_.push_back(step);
	      }
	      recordNumMinusLearnIterationSet_ = false;
	      maxSteps_ = 0;
	      for (auto& elem : steps_)
	      {
	        UInt current = elem + 1;
	        if (current > maxSteps_)
	        {
	          maxSteps_ = current;
	        }
	      }
	      actualValues_.push_back(0.0);
	      actualValuesSet_.push_back(false);

	      // TODO: insert maxBucketIdx / maxInputIdx hint as parameter?
	      // There can be great overhead reallocating the array every time a new
	      // input is seen, especially if we start at (0, 0). The client will
	      // usually know what is the final maxInputIdx (typically the number
	      // of columns?), and we can have heuristics using the encoder's 
	      // settings to get an good approximate of the maxBucketIdx, thus having
	      // to reallocate this matrix only a few times, even never if we use
	      // lower bounds
	      for (const auto& step : steps_)
	      {
	      	Matrix weights = Matrix(maxInputIdx_ + 1, maxBucketIdx_ + 1)
	      	weightMatrix_.insert(pair<UInt, Matrix>(step, matrix))
	      }
	    }

	    SDRClassifier::~SDRClassifier()
	    {
	    }

	    void SDRClassifier::compute(
	      UInt recordNum, const vector<UInt>& patternNZ, UInt bucketIdx,
	      Real64 actValue, bool category, bool learn, bool infer,
	      ClassifierResult* result)
	    {
	    	// Save the offset between recordNum and learnIteration_ if this
	    	// was not set (first call to compute)
	    	if (!recordNumMinusLearnIterationSet_)
	    	{
	    		recordNumMinusLearnIteration_ = recordNum - learnIteration_;
	    		recordNumMinusLearnIterationSet_ = true;
	    	}

	    	// Update learnIteration_
	    	learnIteration_ = recordNum - recordNumMinusLearnIteration_;

	    	// Update pattern history
        patternNZHistory_.emplace_front(patternNZ.begin(), patternNZ.end());
        iterationNumHistory_.push_front(learnIteration_);
        if (patternNZHistory_.size() > maxSteps_)
        {
          patternNZHistory_.pop_back();
          iterationNumHistory_.pop_back();
        }

        // If input pattern has greater index than previously seen, update 
        // maxInputIdx and augment weight matrix with zero padding
        maxInputIdx = max_element(patternNZ.begin(), patternNZ.end();
        if (maxInputIdx) > maxBucketIdx_)
        {
        	maxInputIdx_ = maxInputIdx;
        	for (const auto& step : steps_)
        	{	
	        	weightMatrix_[*step].resize(maxInputIdx_ + 1, maxBucketIdx_ + 1);
          }
        }

        // If in inference mode, compute likelihood and update return value
        if (infer)
        {
        		infer_(patternNZ, bucketIdx, actValue, result);
        }

        // Update weights if in learning mode
        if (learn)
        {
        	// If bucket is greater, updated maxBucketIdx_ and augment weight
        	// matrix with zero-padding
        	if (bucketIdx > maxBucketIdx_) 
        	{
        		maxBucketIdx_ = bucketIdx;
        		for (const auto& step : steps_)
        		{
		        	weightMatrix_[*step].resize(maxInputIdx_ + 1, maxBucketIdx_ + 1);
        		}
        	}

        	// Update rolling averages of values
        	while (actualValues_.size() <= maxBucketIdx_)
          {
            actualValues_.push_back(0.0);
            actualValuesSet_.push_back(false);
          }
          if (!actualValuesSet_[bucketIdx] || category)
          {
            actualValues_[bucketIdx] = actValue;
            actualValuesSet_[bucketIdx] = true;
          } else {
            actualValues_[bucketIdx] =
                ((1.0 - actValueAlpha_) * actualValues_[bucketIdx]) +
                (actValueAlpha_ * actValue);
          }

          for (deque<UInt>::const_iterator learnIteration =
               iterationNumHistory_.begin();
               learnIteration !=iterationNumHistory_.end();
               learnIteration++, patternIteration++)
          {
            const vector<UInt> learnPatternNZ = *patternIteration;
            UInt nSteps = learnIteration_ - *learnIteration;

            // Update weights
            if (binary_search(steps_.begin(), steps_.end(), nSteps))
            {
            	Real64 error = calculateError_(bucketIdx, actValue, nSteps);
              for (auto& bit : learnPatternNZ)
              {
              	for (int bucket = 0; bucket < maxBucketIdx_; ++i)
              	{
              		weightMatrix_[*nSteps].at(*bit, bucket) += alpha_ * error;
              	}
              }
            }

          }

        }
	    }  // end of compute method

      UInt FastCLAClassifier::persistentSize() const
      {
        // TODO: this won't scale!
        stringstream s;
        s.flags(ios::scientific);
        s.precision(numeric_limits<double>::digits10 + 1);
        save(s);
        return s.str().size();
      }

	    void SdrClassifier::infer_(vector<UInt>& patternNZ, UInt, bucketIdx, 
	    	Real64 actValue, ClassifierResult* result)
	    {
    	  // Add the actual values to the return value. For buckets that haven't
        // been seen yet, the actual value doesn't matter since it will have
        // zero likelihood.
        vector<Real64>* actValueVector = result->createVector(
            -1, actualValues_.size(), 0.0);
        for (UInt i = 0; i < actualValues_.size(); ++i)
        {
          if (actualValuesSet_[i])
          {
            (*actValueVector)[i] = actualValues_[i];
          } else {
            // if doing 0-step ahead prediction, we shouldn't use any
            // knowledge of the classification input during inference
            if (steps_.at(0) == 0)
            {
              (*actValueVector)[i] = 0;
            } else {
              (*actValueVector)[i] = actValue;
            }
          }
        }

        for (auto nSteps = steps_.begin(), steps!=steps_.end(), ++steps)
        {
        	vector<Real64>* likelihoods = result->createVector(*nSteps, 
        		maxBucketIdx_ + 1, 1.0 / actualValues_.size());
        	for (auto& bit : patternNZ)
        	{
        		Matrix weights = weightMatrix_[*nSteps];
        		add(likelihoods.begin(), likelihoods.end(), weights.begin(bit),
        			weights.begin(bit + 1)); // ???
        	}
        	(*likelihoods) = exp(*likelihoods); 
        	(*likelihoods) = normalize(likelihoods.begin(), likelihoods.end(),
        		1.0, 1.0);
        }

	    }



	  } 	 
	}
}

