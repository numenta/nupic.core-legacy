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

/** @file
 * Definitions for the SDRClassifier.
 */

#ifndef NTA_SDR_CLASSIFIER_HPP
#define NTA_SDR_CLASSIFIER_HPP

#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <nupic/proto/SdrClassifier.capnp.h>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/math/DenseMatrix.hpp>

namespace nupic 
{
  namespace algorithms
  {

    typedef cla_classifier::ClassifierResult ClassifierResult;

    namespace sdr_classifier
    {

      const UInt sdrClassifierVersion = 1;

      typedef Dense<UInt, Real64> Matrix;

      class SDRClassifier : public Serializable<SdrClassifierProto>
      {
        public:
          /**
           * Constructor for use when deserializing.
           */
          SDRClassifier() {}

          /**
           * Constructor.
           *
           * @param steps The different number of steps to learn and predict.
           * @param alpha The alpha to use when decaying the duty cycles.
           * @param actValueAlpha The alpha to use when decaying the actual
           *                      values for each bucket.
           * @param verbosity The logging verbosity.
           */
          SDRClassifier(
            const vector<UInt>& steps, Real64 alpha, Real64 actValueAlpha,
            UInt verbosity);

          /**
           * Destructor.
           */
          virtual ~SDRClassifier();

          /**
           * Compute the likelihoods for each bucket.
           *
           * @param recordNum An incrementing integer for each record. Gaps in
           *                  numbers correspond to missing records.
           * @param patternNZ The active input bit indices.
           * @param bucketIdx The current value bucket index.
           * @param actValue The current scalar value.
           * @param category Whether the actual values represent categories.
           * @param learn Whether or not to perform learning.
           * @param infer Whether or not to perform inference.
           * @param result A mapping from prediction step to a vector of
           *               likelihoods where the value at an index corresponds
           *               to the bucket with the same index. In addition, the
           *               values for key 0 correspond to the actual values to
           *               used when predicting each bucket.
           */
          virtual void compute(
            UInt recordNum, const vector<UInt>& patternNZ, const vector<UInt>& bucketIdxList,
            const vector<Real64>& actValueList, bool category, bool learn, bool infer,
            ClassifierResult* result);

          /**
           * Gets the version number
           */
          UInt version() const;

          /**
           * Getter and setter for verbosity level.
           */
          UInt getVerbosity() const;
          void setVerbosity(UInt verbosity);

          /**
           * Gets the learning rate
           */
          UInt getAlpha() const;

          /**
           * Get the size of the string needed for the serialized state.
           */
          UInt persistentSize() const;

          /**
           * Save the state to the ostream.
           */
          void save(std::ostream& outStream) const;

          /**
           * Load state from istream.
           */
          void load(std::istream& inStream);

          /**
           * Save the state to the builder.
           */
          void write(SdrClassifierProto::Builder& proto) const override;

          /**
           * Save the state to the stream.
           */
          using Serializable::write;

          /**
           * Load state from reader.
           */
          void read(SdrClassifierProto::Reader& proto) override;

          /**
           * Load state from stream.
           */
          using Serializable::read;

          /**
           * Compare the other instance to this one.
           *
           * @param other Another instance of SDRClassifier to compare to.
           * @returns true iff other is identical to this instance.
           */
          virtual bool operator==(const SDRClassifier& other) const;

        private:
          // Helper function for inference mode
          void infer_(const vector<UInt>& patternNZ,
            const vector<Real64>& actValue, ClassifierResult* result);

          // Helper function to compute the error signal in learning mode
          vector<Real64> calculateError_(const vector<UInt>& bucketIdxList, 
            const vector<UInt> patternNZ, UInt step);

          // The list of prediction steps to learn and infer.
          vector<UInt> steps_;

          // The alpha used to decay the duty cycles in the BitHistorys.
          Real64 alpha_;

          // The alpha used to decay the actual values used for each bucket.
          Real64 actValueAlpha_;

          // The maximum number of the prediction steps.
          UInt maxSteps_;

          // Stores the input pattern history, starting with the previous input
          // and containing _maxSteps total input patterns.
          deque< vector<UInt> > patternNZHistory_;
          deque<UInt> recordNumHistory_;

          // Weight matrices for the classifier (one per prediction step)
          map<UInt, Matrix> weightMatrix_;

          // The highest input bit that the classifier has seen so far.
          UInt maxInputIdx_;

          // The highest bucket index that the classifier has been seen so far.
          UInt maxBucketIdx_;

          // The current actual values used for each bucket index. The index of
          // the actual value matches the index of the bucket.
          vector<Real64> actualValues_;

          // A boolean that distinguishes between actual values that have been
          // seen and those that have not.
          vector<bool> actualValuesSet_;

          // Version and verbosity.
          UInt version_;
          UInt verbosity_;
      };  // end of SDRClassifier class

    }  // end of namespace sdr_classifier
  }  // end of namespace algorithms
}  // end of name space nupic

#endif 
