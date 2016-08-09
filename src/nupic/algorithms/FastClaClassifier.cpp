/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <nupic/algorithms/BitHistory.hpp>
#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/FastClaClassifier.hpp>
#include <nupic/proto/ClaClassifier.capnp.h>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

using namespace std;

namespace nupic
{
  namespace algorithms
  {
    namespace cla_classifier
    {

      FastCLAClassifier::FastCLAClassifier(
          const vector<UInt>& steps, Real64 alpha, Real64 actValueAlpha,
          UInt verbosity) : alpha_(alpha), actValueAlpha_(actValueAlpha),
          learnIteration_(0), recordNumMinusLearnIteration_(0),
          maxBucketIdx_(0), version_(claClassifierVersion),
          verbosity_(verbosity)
      {
        for (const auto & step : steps)
        {
          steps_.push_back(step);
        }
        recordNumMinusLearnIterationSet_ = false;
        maxSteps_ = 0;
        for (auto & elem : steps_)
        {
          UInt current = elem + 1;
          if (current > maxSteps_)
          {
            maxSteps_ = current;
          }
        }
        actualValues_.push_back(0.0);
        actualValuesSet_.push_back(false);
      }

      FastCLAClassifier::~FastCLAClassifier()
      {
      }

      void FastCLAClassifier::fastCompute(
          UInt recordNum, const vector<UInt>& patternNZ, UInt bucketIdx,
          Real64 actValue, bool category, bool learn, bool infer,
          ClassifierResult* result)
      {
        // Save the offset between recordNum and learnIteration_ if this is the
        // first compute.
        if (!recordNumMinusLearnIterationSet_)
        {
          recordNumMinusLearnIteration_ = recordNum - learnIteration_;
          recordNumMinusLearnIterationSet_ = true;
        }

        // Update the learn iteration.
        learnIteration_ = recordNum - recordNumMinusLearnIteration_;

        // Update the input pattern history.
        patternNZHistory_.emplace_front(patternNZ.begin(), patternNZ.end());

        iterationNumHistory_.push_front(learnIteration_);
        if (patternNZHistory_.size() > maxSteps_)
        {
          patternNZHistory_.pop_back();
          iterationNumHistory_.pop_back();
        }

        // If inference is enabled, compute the likelihoods and add them to the
        // return value.
        if (infer)
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

          // Generate the predictions for each steps-ahead value
          for (auto step = steps_.begin(); step != steps_.end(); ++step)
          {
            // Skip if we don't have data yet.
            if (activeBitHistory_.find(*step) == activeBitHistory_.end())
            {
              // This call creates the vector with specified default values.
              result->createVector(
                  *step, actualValues_.size(), 1.0 / actualValues_.size());
              continue;
            }

            vector<Real64>* likelihoods = result->createVector(
                *step, maxBucketIdx_ + 1, 0.0);
            vector<Real64> bitVotes(maxBucketIdx_ + 1, 0.0);

            for (const auto & elem : patternNZ)
            {
              if (activeBitHistory_[*step].find(elem) !=
                  activeBitHistory_[*step].end())
              {
                BitHistory& history =
                    activeBitHistory_[*step].find(elem)->second;
                for (auto & bitVote : bitVotes) {
                  bitVote = 0.0;
                }
                history.infer(learnIteration_, &bitVotes);
                for (UInt i = 0; i < bitVotes.size(); ++i) {
                  (*likelihoods)[i] += bitVotes[i];
                }
              }
            }
            Real64 total = 0.0;
            for (auto & likelihood : *likelihoods)
            {
              total += likelihood;
            }
            for (auto & likelihood : *likelihoods)
            {
              if (total > 0.0)
              {
                likelihood = likelihood / total;
              } else {
                likelihood = 1.0 / likelihoods->size();
              }
            }
          }
        }

        // If learning is enabled, update the bit histories.
        if (learn)
        {
          // Update the predicted actual values for each bucket.
          if (bucketIdx > maxBucketIdx_)
          {
            maxBucketIdx_ = bucketIdx;
          }
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

          for (auto & elem : steps_)
          {
            UInt step = elem;

            // Check if there is a pattern that should be assigned to this
            // classification in our history. If not, skip it.
            bool found = false;
            deque<vector<UInt>>::const_iterator patternIteration =
                patternNZHistory_.begin();
            for (deque<UInt>::const_iterator learnIteration =
                 iterationNumHistory_.begin();
                 learnIteration !=iterationNumHistory_.end();
                 ++learnIteration, ++patternIteration)
            {
              if (*learnIteration == (learnIteration_ - step))
              {
                found = true;
                break;
              }
            }
            if (!found)
            {
              continue;
            }

            // Store classification info for each active bit from the pattern
            // that we got step time steps ago.
            const vector<UInt> learnPatternNZ = *patternIteration;
            for (auto & learnPatternNZ_j : learnPatternNZ)
            {
              UInt bit = learnPatternNZ_j;
              // This will implicitly insert the key "step" into the map if it
              // doesn't exist yet.
              auto it = activeBitHistory_[step].find(bit);
              if (it == activeBitHistory_[step].end())
              {
                activeBitHistory_[step][bit] =
                    BitHistory(bit, step, alpha_, verbosity_);
              }
              activeBitHistory_[step][bit].store(learnIteration_, bucketIdx);
            }
          }
        }
      }

      UInt FastCLAClassifier::persistentSize() const
      {
        // TODO: this won't scale!
        stringstream s;
        s.flags(ios::scientific);
        s.precision(numeric_limits<double>::digits10 + 1);
        save(s);
        return s.str().size();
      }

      void FastCLAClassifier::save(ostream& outStream) const
      {
        // Write a starting marker and version.
        outStream << "FastCLAClassifier" << endl;
        outStream << version_ << endl;

        // Store the simple variables first.
        outStream << version() << " "
                  << alpha_ << " "
                  << actValueAlpha_ << " "
                  << learnIteration_ << " "
                  << maxSteps_ << " "
                  << maxBucketIdx_ << " "
                  << verbosity_ << " "
                  << endl;

        // V1 additions.
        outStream << recordNumMinusLearnIteration_ << " "
                  << recordNumMinusLearnIterationSet_ << " ";
        outStream << iterationNumHistory_.size() << " ";
        for (const auto & elem : iterationNumHistory_)
        {
          outStream << elem << " ";
        }
        outStream << endl;

        // Store the different prediction steps.
        outStream << steps_.size() << " ";
        for (auto & elem : steps_)
        {
          outStream << elem << " ";
        }
        outStream << endl;

        // Store the input pattern history.
        outStream << patternNZHistory_.size() << " ";
        for (auto & pattern : patternNZHistory_)
        {
          outStream << pattern.size() << " ";
          for (auto & pattern_j : pattern)
          {
            outStream << pattern_j << " ";
          }
        }
        outStream << endl;

        // Store the bucket duty cycles.
        outStream << activeBitHistory_.size() << " ";
        for (const auto & elem : activeBitHistory_)
        {
          outStream << elem.first << " ";
          outStream << elem.second.size() << " ";
          for (auto it2 = elem.second.begin(); it2 != elem.second.end(); ++it2)
          {
            outStream << it2->first << " ";
            it2->second.save(outStream);
          }
        }

        // Store the actual values for each bucket.
        outStream << actualValues_.size() << " ";
        for (UInt i = 0; i < actualValues_.size(); ++i)
        {
          outStream << actualValues_[i] << " ";
          outStream << actualValuesSet_[i] << " ";
        }
        outStream << endl;

        // Write an ending marker.
        outStream << "~FastCLAClassifier" << endl;

      }

      void FastCLAClassifier::load(istream& inStream)
      {
        // Clean up the existing data structures before loading
        steps_.clear();
        iterationNumHistory_.clear();
        patternNZHistory_.clear();
        actualValues_.clear();
        actualValuesSet_.clear();
        activeBitHistory_.clear();

        // Check the starting marker.
        string marker;
        inStream >> marker;
        NTA_CHECK(marker == "FastCLAClassifier");

        // Check the version.
        UInt version;
        inStream >> version;
        NTA_CHECK(version <= 1);

        // Load the simple variables.
        inStream >> version_
                 >> alpha_
                 >> actValueAlpha_
                 >> learnIteration_
                 >> maxSteps_
                 >> maxBucketIdx_
                 >> verbosity_;

        // V1 additions.
        UInt numIterationHistory;
        UInt curIterationNum;
        if (version == 1)
        {
          inStream >> recordNumMinusLearnIteration_
                   >> recordNumMinusLearnIterationSet_;
          inStream >> numIterationHistory;
          for (UInt i = 0; i < numIterationHistory; ++i)
          {
            inStream >> curIterationNum;
            iterationNumHistory_.push_back(curIterationNum);
          }
        } else {
          recordNumMinusLearnIterationSet_ = false;
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
          if (version == 0)
          {
            iterationNumHistory_.push_back(
                learnIteration_ - (size - i));
          }
        }

        // Load the bucket duty cycles.
        UInt numSteps;
        UInt numInputBits;
        UInt inputBit;
        inStream >> numSteps;
        for (UInt i = 0; i < numSteps; ++i)
        {
          inStream >> step;
          // Insert the step to initialize the BitHistory
          activeBitHistory_[step];
          inStream >> numInputBits;
          for (UInt j = 0; j < numInputBits; ++j)
          {
            inStream >> inputBit;
            activeBitHistory_[step][inputBit].load(inStream);
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
        NTA_CHECK(marker == "~FastCLAClassifier");

        // Update the version number.
        version_ = claClassifierVersion;
      }

      void FastCLAClassifier::write(ClaClassifierProto::Builder& proto) const
      {
        auto stepsProto = proto.initSteps(steps_.size());
        for (UInt i = 0; i < steps_.size(); i++)
        {
          stepsProto.set(i, steps_[i]);
        }

        proto.setAlpha(alpha_);
        proto.setActValueAlpha(actValueAlpha_);
        proto.setLearnIteration(learnIteration_);
        proto.setRecordNumMinusLearnIteration(recordNumMinusLearnIteration_);
        proto.setRecordNumMinusLearnIterationSet(
            recordNumMinusLearnIterationSet_);
        proto.setMaxSteps(maxSteps_);

        auto patternNZHistoryProto =
          proto.initPatternNZHistory(patternNZHistory_.size());
        for (UInt i = 0; i < patternNZHistory_.size(); i++)
        {
          const auto & pattern = patternNZHistory_[i];
          auto patternProto = patternNZHistoryProto.init(i, pattern.size());
          for (UInt j = 0; j < pattern.size(); j++)
          {
            patternProto.set(j, pattern[j]);
          }
        }

        auto iterationNumHistoryProto =
          proto.initIterationNumHistory(iterationNumHistory_.size());
        for (UInt i = 0; i < iterationNumHistory_.size(); i++)
        {
          iterationNumHistoryProto.set(i, iterationNumHistory_[i]);
        }

        auto activeBitHistoryProtos =
          proto.initActiveBitHistory(activeBitHistory_.size());
        UInt i = 0;
        for (const auto & stepBitHistory : activeBitHistory_)
        {
          auto stepBitHistoryProto = activeBitHistoryProtos[i];
          stepBitHistoryProto.setSteps(stepBitHistory.first);
          auto indexBitHistoryListProto =
            stepBitHistoryProto.initBitHistories(stepBitHistory.second.size());
          UInt j = 0;
          for (const auto & indexBitHistory : stepBitHistory.second)
          {
            auto indexBitHistoryProto = indexBitHistoryListProto[j];
            indexBitHistoryProto.setIndex(indexBitHistory.first);
            auto bitHistoryProto = indexBitHistoryProto.initHistory();
            indexBitHistory.second.write(bitHistoryProto);
            j++;
          }
          i++;
        }

        proto.setMaxBucketIdx(maxBucketIdx_);

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

      void FastCLAClassifier::read(ClaClassifierProto::Reader& proto)
      {
        // Clean up the existing data structures before loading
        steps_.clear();
        iterationNumHistory_.clear();
        patternNZHistory_.clear();
        actualValues_.clear();
        actualValuesSet_.clear();
        activeBitHistory_.clear();

        for (auto step : proto.getSteps())
        {
          steps_.push_back(step);
        }

        alpha_ = proto.getAlpha();
        actValueAlpha_ = proto.getActValueAlpha();
        learnIteration_ = proto.getLearnIteration();
        recordNumMinusLearnIteration_ = proto.getRecordNumMinusLearnIteration();
        recordNumMinusLearnIterationSet_ =
          proto.getRecordNumMinusLearnIterationSet();
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

        auto iterationNumHistoryProto = proto.getIterationNumHistory();
        for (UInt i = 0; i < iterationNumHistoryProto.size(); i++)
        {
          iterationNumHistory_.push_back(iterationNumHistoryProto[i]);
        }

        auto activeBitHistoryProto = proto.getActiveBitHistory();
        for (UInt i = 0; i < activeBitHistoryProto.size(); i++)
        {
          auto stepBitHistories = activeBitHistoryProto[i];
          UInt steps = stepBitHistories.getSteps();
          for (auto indexBitHistoryProto : stepBitHistories.getBitHistories())
          {
            BitHistory& bitHistory =
              activeBitHistory_[steps][indexBitHistoryProto.getIndex()];
            auto bitHistoryProto = indexBitHistoryProto.getHistory();
            bitHistory.read(bitHistoryProto);
          }
        }

        maxBucketIdx_ = proto.getMaxBucketIdx();

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

      bool FastCLAClassifier::operator==(const FastCLAClassifier& other) const
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
            learnIteration_ != other.learnIteration_ ||
            recordNumMinusLearnIteration_ !=
                other.recordNumMinusLearnIteration_  ||
            recordNumMinusLearnIterationSet_ !=
                other.recordNumMinusLearnIterationSet_  ||
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

        if (iterationNumHistory_.size() !=
            other.iterationNumHistory_.size())
        {
          return false;
        }
        for (UInt i = 0; i < iterationNumHistory_.size(); i++)
        {
          if (iterationNumHistory_.at(i) !=
              other.iterationNumHistory_.at(i))
          {
            return false;
          }
        }

        if (activeBitHistory_.size() != other.activeBitHistory_.size())
        {
          return false;
        }
        for (auto it1 = activeBitHistory_.begin();
             it1 != activeBitHistory_.end(); it1++)
        {
          auto thisInnerMap = it1->second;
          auto otherInnerMap = other.activeBitHistory_.at(it1->first);
          if (thisInnerMap.size() != otherInnerMap.size())
          {
            return false;
          }
          for (auto it2 = thisInnerMap.begin(); it2 != thisInnerMap.end();
               it2++)
          {
            auto thisBitHistory = it2->second;
            auto otherBitHistory = otherInnerMap.at(it2->first);
            if (thisBitHistory != otherBitHistory)
            {
              return false;
            }
          }
        }

        if (maxBucketIdx_ != other.maxBucketIdx_)
        {
          return false;
        }

        if (actualValues_.size() != other.actualValues_.size() ||
            actualValuesSet_.size() != other.actualValuesSet_.size())
        {
          return false;
        }
        for (UInt i = 0; i < actualValues_.size(); i++)
        {
          if (fabs(actualValues_[i] - other.actualValues_[i]) > 0.000001 ||
              fabs(actualValuesSet_[i] - other.actualValuesSet_[i]) > 0.00001)
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

    } // end namespace cla_classifier
  } // end namespace algorithms
} // end namespace nupic
