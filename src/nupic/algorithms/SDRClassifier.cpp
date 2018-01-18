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
          UInt verbosity) : steps_(steps), alpha_(alpha),
          actValueAlpha_(actValueAlpha), maxInputIdx_(0), maxBucketIdx_(0),
          actualValues_({0.0}), actualValuesSet_({false}),
          version_(sdrClassifierVersion), verbosity_(verbosity)
      {
        sort(steps_.begin(), steps_.end());
        if (steps_.size() > 0)
        {
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
        for (const auto& step : steps_)
        {
          weightMatrix_.emplace(step, Matrix(maxInputIdx_ + 1,
            maxBucketIdx_ + 1));
        }
      }

      SDRClassifier::~SDRClassifier()
      {
      }

      void SDRClassifier::compute(
        UInt recordNum, const vector<UInt>& patternNZ, const vector<UInt>& bucketIdxList,
        const vector<Real64>& actValueList, bool category, bool learn, bool infer,
        ClassifierResult* result)
      {
        // ensures that recordNum increases monotonically
        UInt lastRecordNum = -1;
        if (recordNumHistory_.size() > 0)
        {
          lastRecordNum = recordNumHistory_[recordNumHistory_.size()-1];
          if (recordNum < lastRecordNum)
            NTA_THROW << "the record number has to increase monotonically";
        }

        // update pattern history if this is a new record
        if (recordNumHistory_.size() == 0 || recordNum > lastRecordNum)
        {
          patternNZHistory_.emplace_back(patternNZ.begin(), patternNZ.end());
          recordNumHistory_.push_back(recordNum);
          if (patternNZHistory_.size() > maxSteps_)
          {
            patternNZHistory_.pop_front();
            recordNumHistory_.pop_front();
          }
        }

        // if input pattern has greater index than previously seen, update 
        // maxInputIdx and augment weight matrix with zero padding
        if (patternNZ.size() > 0)
        {
          UInt maxInputIdx = *max_element(patternNZ.begin(), patternNZ.end());
          if (maxInputIdx > maxInputIdx_)
          {
            maxInputIdx_ = maxInputIdx;
            for (const auto& step : steps_)
            {
              Matrix& weights = weightMatrix_.at(step);
              weights.resize(maxInputIdx_ + 1, maxBucketIdx_ + 1);
            }
          }
        }

        // if in inference mode, compute likelihood and update return value
        if (infer)
        {
          infer_(patternNZ, actValueList, result);
        }

        // update weights if in learning mode
        if (learn)
        {
          for(size_t categoryI=0; categoryI < bucketIdxList.size(); categoryI++)
          {
            UInt bucketIdx = bucketIdxList[categoryI];
            Real64 actValue = actValueList[categoryI];
            // if bucket is greater, update maxBucketIdx_ and augment weight
            // matrix with zero-padding
            if (bucketIdx > maxBucketIdx_) 
            {
              maxBucketIdx_ = bucketIdx;
              for (const auto& step : steps_)
              {
                Matrix& weights = weightMatrix_.at(step);
                weights.resize(maxInputIdx_ + 1, maxBucketIdx_ + 1);
              }
            }

            // update rolling averages of bucket values
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
          }

          // compute errors and update weights
          auto patternIteration = patternNZHistory_.begin();
          for (auto learnRecord = recordNumHistory_.begin();
               learnRecord != recordNumHistory_.end();
               learnRecord++, patternIteration++)
          {
            const vector<UInt> learnPatternNZ = *patternIteration;
            const UInt nSteps = recordNum - *learnRecord;

            // update weights
            if (binary_search(steps_.begin(), steps_.end(), nSteps))
            {
              vector<Real64> error = calculateError_(bucketIdxList, 
                learnPatternNZ, nSteps);
              Matrix& weights = weightMatrix_.at(nSteps);
              for (auto& bit : learnPatternNZ)
              {
                weights.axby(bit, 1.0, alpha_, error);
              }
            }
          }
        }

      }

      UInt SDRClassifier::persistentSize() const
      {
        stringstream s;
        s.flags(ios::scientific);
        s.precision(numeric_limits<double>::digits10 + 1);
        save(s);
        return s.str().size();
      }

      void SDRClassifier::infer_(const vector<UInt>& patternNZ, 
        const vector<Real64>&  actValue, ClassifierResult* result)
      {
        // add the actual values to the return value. For buckets that haven't
        // been seen yet, the actual value doesn't matter since it will have
        // zero likelihood.
        vector<Real64>* actValueVector = result->createVector(-1,
          actualValues_.size(), 0.0);
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
              (*actValueVector)[i] = actValue[0];
            }
          }
        }

        for (auto nSteps = steps_.begin(); nSteps!=steps_.end(); ++nSteps)
        {
          vector<Real64>* likelihoods = result->createVector(*nSteps, 
            maxBucketIdx_ + 1, 0.0);
          for (auto& bit : patternNZ)
          {
            const Matrix& weights = weightMatrix_.at(*nSteps);
            add(likelihoods->begin(), likelihoods->end(), weights.begin(bit),
              weights.begin(bit + 1));
          }
          // compute softmax of raw scores
          // TODO: fix potential overflow problem by shifting scores by their
          // maximal value across buckets
          range_exp(1.0, *likelihoods);
          normalize(*likelihoods, 1.0, 1.0);
        }
      }

      vector<Real64> SDRClassifier::calculateError_(const vector<UInt>& bucketIdxList, 
        const vector<UInt> patternNZ, UInt step)
      {
        // compute predicted likelihoods
        vector<Real64> likelihoods (maxBucketIdx_ + 1, 0);

        for (auto& bit : patternNZ)
        {
          const Matrix& weights = weightMatrix_.at(step);
          add(likelihoods.begin(), likelihoods.end(), weights.begin(bit),
            weights.begin(bit + 1));
        }
        range_exp(1.0, likelihoods);
        normalize(likelihoods, 1.0, 1.0);

        // compute target likelihoods
        vector<Real64> targetDistribution (maxBucketIdx_ + 1, 0.0);
        Real64 numCategories = (Real64)bucketIdxList.size();
        for(size_t i=0; i<bucketIdxList.size(); i++)
          targetDistribution[bucketIdxList[i]] = 1.0 / numCategories;

        axby(-1.0, likelihoods, 1.0, targetDistribution);
        return likelihoods;
      }

      UInt SDRClassifier::version() const
      {
        return version_;
      }

      UInt SDRClassifier::getVerbosity() const
      {
        return verbosity_;
      }

      void SDRClassifier::setVerbosity(UInt verbosity)
      {
        verbosity_ = verbosity;
      }

      UInt SDRClassifier::getAlpha() const
      {
        return alpha_;
      }

      void SDRClassifier::save(ostream& outStream) const
      {
        // Write a starting marker and version.
        outStream << "SDRClassifier" << endl;
        outStream << version_ << endl;

        // Store the simple variables first.
        outStream << version() << " "
                  << alpha_ << " "
                  << actValueAlpha_ << " "
                  << maxSteps_ << " "
                  << maxBucketIdx_ << " "
                  << maxInputIdx_ << " "
                  << verbosity_ << " "
                  << endl;

        // V1 additions.
        outStream << recordNumHistory_.size() << " ";
        for (const auto& elem : recordNumHistory_)
        {
          outStream << elem << " ";
        }
        outStream << endl;

        // Store the different prediction steps.
        outStream << steps_.size() << " ";
        for (auto& elem : steps_)
        {
          outStream << elem << " ";
        }
        outStream << endl;

        // Store the pattern history.
        outStream << patternNZHistory_.size() << " ";
        for (auto& pattern : patternNZHistory_)
        {
          outStream << pattern.size() << " ";
          for (auto& pattern_j : pattern)
          {
            outStream << pattern_j << " ";
          }
        }
        outStream << endl;

        // Store weight matrix
        outStream << weightMatrix_.size() << " ";
        for (const auto& elem : weightMatrix_)
        {
          outStream << elem.first << " ";
          outStream << elem.second;
        }
        outStream << endl;

        // Store the actual values for each bucket.
        outStream << actualValues_.size() << " ";
        for (UInt i = 0; i < actualValues_.size(); ++i)
        {
          outStream << actualValues_[i] << " ";
          outStream << actualValuesSet_[i] << " ";
        }
        outStream << endl;

        // Write an ending marker.
        outStream << "~SDRClassifier" << endl;
      }

      void SDRClassifier::load(istream& inStream)
      {
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
        NTA_CHECK(version <= 1);

        // Load the simple variables.
        inStream >> version_
                 >> alpha_
                 >> actValueAlpha_
                 >> maxSteps_
                 >> maxBucketIdx_
                 >> maxInputIdx_
                 >> verbosity_;

        UInt recordNumHistory;
        UInt curRecordNum;
        if (version == 1)
        {
          inStream >> recordNumHistory;
          for (UInt i = 0; i < recordNumHistory; ++i)
          {
            inStream >> curRecordNum;
            recordNumHistory_.push_back(curRecordNum);
          }
        }

        // Load the prediction steps.
        UInt size;
        UInt step;
        inStream >> size;
        for (UInt i = 0; i < size; ++i)
        {
          inStream >> step;
          steps_.push_back(step);
        }

        // Load the input pattern history.
        inStream >> size;
        UInt vSize;
        for (UInt i = 0; i < size; ++i)
        {
          inStream >> vSize;
          patternNZHistory_.emplace_back(vSize);
          for (UInt j = 0; j < vSize; ++j)
          {
            inStream >> patternNZHistory_[i][j];
          }
        }

        // Load weight matrix.
        UInt numSteps;
        inStream >> numSteps;
        for (UInt s = 0; s < numSteps; ++s)
        {
          inStream >> step;
          // Insert the step to initialize the weight matrix
          weightMatrix_[step] = Matrix(maxInputIdx_ + 1, maxBucketIdx_ + 1);
          for (UInt i = 0; i <= maxInputIdx_; ++i)
          {
            for (UInt j = 0; j <= maxBucketIdx_; ++j)
            {
              inStream >> weightMatrix_[step].at(i, j);
            }
          }
        }

        // Load the actual values for each bucket.
        UInt numBuckets;
        Real64 actualValue;
        bool actualValueSet;
        inStream >> numBuckets;
        for (UInt i = 0; i < numBuckets; ++i)
        {
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

      void SDRClassifier::write(SdrClassifierProto::Builder& proto) const 
      {
        auto stepsProto = proto.initSteps(steps_.size());
        for (UInt i = 0; i < steps_.size(); i++)
        {
          stepsProto.set(i, steps_[i]);
        }

        proto.setAlpha(alpha_);
        proto.setActValueAlpha(actValueAlpha_);
        proto.setMaxSteps(maxSteps_);

        auto patternNZHistoryProto =
          proto.initPatternNZHistory(patternNZHistory_.size());
        for (UInt i = 0; i < patternNZHistory_.size(); i++)
        {
          const auto& pattern = patternNZHistory_[i];
          auto patternProto = patternNZHistoryProto.init(i, pattern.size());
          for (UInt j = 0; j < pattern.size(); j++)
          {
            patternProto.set(j, pattern[j]);
          }
        }

        auto recordNumHistoryProto =
          proto.initRecordNumHistory(recordNumHistory_.size());
        for (UInt i = 0; i < recordNumHistory_.size(); i++)
        {
          recordNumHistoryProto.set(i, recordNumHistory_[i]);
        }

        proto.setMaxBucketIdx(maxBucketIdx_);
        proto.setMaxInputIdx(maxInputIdx_);

        auto weightMatrixProtos =
          proto.initWeightMatrix(weightMatrix_.size());
        UInt k = 0;
        for (const auto& stepWeightMatrix : weightMatrix_)
        {
          auto stepWeightMatrixProto = weightMatrixProtos[k];
          stepWeightMatrixProto.setSteps(stepWeightMatrix.first);
          auto weightProto = stepWeightMatrixProto.initWeight(
            (maxInputIdx_ + 1) * (maxBucketIdx_ + 1)
            );
          // flatten weight matrix, serialized as a list of floats
          UInt idx = 0;
          for (UInt i = 0; i <= maxInputIdx_; ++i)
          {
            for (UInt j = 0; j <= maxBucketIdx_; ++j)
            {
              weightProto.set(idx, stepWeightMatrix.second.at(i, j));
              idx++;
            }
          }
          k++;
        }

        auto actualValuesProto = proto.initActualValues(actualValues_.size());
        for (UInt i = 0; i < actualValues_.size(); i++)
        {
          actualValuesProto.set(i, actualValues_[i]);
        }

        auto actualValuesSetProto =
          proto.initActualValuesSet(actualValuesSet_.size());
        for (UInt i = 0; i < actualValuesSet_.size(); i++)
        {
          actualValuesSetProto.set(i, actualValuesSet_[i]);
        }

        proto.setVersion(version_);
        proto.setVerbosity(verbosity_);
      }

      void SDRClassifier::read(SdrClassifierProto::Reader& proto)
      {
        // Clean up the existing data structures before loading
        steps_.clear();
        recordNumHistory_.clear();
        patternNZHistory_.clear();
        actualValues_.clear();
        actualValuesSet_.clear();
        weightMatrix_.clear();

        for (auto step : proto.getSteps())
        {
          steps_.push_back(step);
        }

        alpha_ = proto.getAlpha();
        actValueAlpha_ = proto.getActValueAlpha();
        maxSteps_ = proto.getMaxSteps();

        auto patternNZHistoryProto = proto.getPatternNZHistory();
        for (UInt i = 0; i < patternNZHistoryProto.size(); i++)
        {
          patternNZHistory_.emplace_back(patternNZHistoryProto[i].size());
          for (UInt j = 0; j < patternNZHistoryProto[i].size(); j++)
          {
            patternNZHistory_[i][j] = patternNZHistoryProto[i][j];
          }
        }

        auto recordNumHistoryProto = proto.getRecordNumHistory();
        for (UInt i = 0; i < recordNumHistoryProto.size(); i++)
        {
          recordNumHistory_.push_back(recordNumHistoryProto[i]);
        }

        maxBucketIdx_ = proto.getMaxBucketIdx();
        maxInputIdx_ = proto.getMaxInputIdx();

        auto weightMatrixProto = proto.getWeightMatrix();
        for (UInt i = 0; i < weightMatrixProto.size(); ++i)
        {
          auto stepWeightMatrix = weightMatrixProto[i];
          UInt steps = stepWeightMatrix.getSteps();
          weightMatrix_[steps] = Matrix(maxInputIdx_ + 1, maxBucketIdx_ + 1);
          auto weights = stepWeightMatrix.getWeight();
          UInt j = 0;
          // un-flatten weight matrix, serialized as a list of floats
          for (UInt row = 0; row <= maxInputIdx_; ++row)
          {
            for (UInt col = 0; col <= maxBucketIdx_; ++col)
            {
              weightMatrix_[steps].at(row, col) = weights[j];
              j++;
            }
          }
        }

        for (auto actValue : proto.getActualValues())
        {
          actualValues_.push_back(actValue);
        }

        for (auto actValueSet : proto.getActualValuesSet())
        {
          actualValuesSet_.push_back(actValueSet);
        }

        version_ = proto.getVersion();
        verbosity_ = proto.getVerbosity();
      }

      bool SDRClassifier::operator==(const SDRClassifier& other) const
      {
        if (steps_.size() != other.steps_.size())
        {
          return false;
        }
        for (UInt i = 0; i < steps_.size(); i++)
        {
          if (steps_.at(i) != other.steps_.at(i))
          {
            return false;
          }
        }

        if (fabs(alpha_ - other.alpha_) > 0.000001 ||
            fabs(actValueAlpha_ - other.actValueAlpha_) > 0.000001 ||
            maxSteps_ != other.maxSteps_)
        {
          return false;
        }

        if (patternNZHistory_.size() != other.patternNZHistory_.size())
        {
          return false;
        }
        for (UInt i = 0; i < patternNZHistory_.size(); i++)
        {
          if (patternNZHistory_.at(i).size() !=
              other.patternNZHistory_.at(i).size())
          {
            return false;
          }
          for (UInt j = 0; j < patternNZHistory_.at(i).size(); j++)
          {
            if (patternNZHistory_.at(i).at(j) !=
                other.patternNZHistory_.at(i).at(j))
            {
              return false;
            }
          }
        }

        if (recordNumHistory_.size() !=
            other.recordNumHistory_.size())
        {
          return false;
        }
        for (UInt i = 0; i < recordNumHistory_.size(); i++)
        {
          if (recordNumHistory_.at(i) !=
              other.recordNumHistory_.at(i))
          {
            return false;
          }
        }

        if (maxBucketIdx_ != other.maxBucketIdx_)
        {
          return false;
        }

        if (maxInputIdx_ != other.maxInputIdx_)
        {
          return false;
        }

        if (weightMatrix_.size() != other.weightMatrix_.size())
        {
          return false;
        }
        for (auto it = weightMatrix_.begin(); it != weightMatrix_.end(); it++)
        {
          Matrix thisWeights = it->second;
          Matrix otherWeights = other.weightMatrix_.at(it->first);
          for (UInt i = 0; i <= maxInputIdx_; ++i)
          {
            for (UInt j = 0; j <= maxBucketIdx_; ++j)
            {
              if (thisWeights.at(i, j) != otherWeights.at(i, j))
              {
                return false;
              }
            }
          }
        }

        if (actualValues_.size() != other.actualValues_.size() ||
            actualValuesSet_.size() != other.actualValuesSet_.size())
        {
          return false;
        }
        for (UInt i = 0; i < actualValues_.size(); i++)
        {
          if (fabs(actualValues_[i] - other.actualValues_[i]) > 0.000001 ||
              actualValuesSet_[i] != other.actualValuesSet_[i])
          {
            return false;
          }
        }

        if (version_ != other.version_ ||
            verbosity_ != other.verbosity_)
        {
          return false;
        }

        return true;
      }

    }    
  }
}
