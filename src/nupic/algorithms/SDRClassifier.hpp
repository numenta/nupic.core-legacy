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

#include <nupic/types/Types.hpp>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/Serializable.hpp>

namespace nupic {
namespace algorithms {
namespace sdr_classifier {


// TODO: Document that categories should be autoincrement style.  start at 0,
// holes are generally bad for this, because they will be represented regardless
// of if they're used.

/**
 * PDF - Probability Distribution Function, distribution of likelihood of values
 *       for each category.
 *
 * See also:  https://en.wikipedia.org/wiki/Probability_distribution
 */
using PDF = std::vector<Real64>;

class Classifier : public Serializable
{
public:
  /**
   * Constructor.
   *
   * @param alpha The alpha to use when decaying the duty cycles.
   */
  Classifier(Real64 alpha);

  Real64 alpha;

  /**
   * Constructor for use when deserializing.
   */
  Classifier() {}
  void initialize(Real64 alpha);

  /**
   * Compute the likelihoods for each category / bucket.
   *
   * @param pattern:       The active input bit SDR.
   * @param probabilities: The Probablility Density Function (PDF) of the categories.
   */
  void infer(const sdr::SDR & pattern, PDF & probabilities);

  /**
   * Convenience method to find the classification with the greatest probablility.
   */
  UInt inferCategory(const sdr::SDR & pattern);

  /**
   * Learn from example data.
   *
   * @param pattern:  The active input bit SDR.
   * @param categoryIdxList:  The current categories or bucket indices.
   */
  void learn(const sdr::SDR & pattern, const std::vector<UInt> & categoryIdxList);

  void save(std::ostream &outStream) const override;
  void load(std::istream &inStream) override;

private:
  std::vector<UInt> dimensions_;
  UInt numCategories_;

  /**
   * 2D map used to store the data.
   * Use as: weights_[ input-bit ][ category-index ]
   */
  std::vector<std::vector<Real>> weights_;

  // Helper function to compute the error signal for learning.
  std::vector<Real64> calculateError_(const std::vector<UInt> &bucketIdxList,
                                      const sdr::SDR &pattern);
};

/**
 * Helper function for Classifier::infer.  Converts the raw data accumulators
 * into a PDF.
 */
void softmax(PDF::iterator begin, PDF::iterator end);


////////////////////////////////////////////////////////////////////////////////


/**
 * The key is the step, for predicting multiple time steps into the future.
 * The key ACTUAL_VALUES contains an estimate of the actual values.
 * The value is a PDF(probability density function, list of probabilities of outcomes) 
 * of the result being in each bucket.
 */
const Int ACTUAL_VALUES = -1;
using ClassifierResult = std::map<Int, PDF>;


using SDR = nupic::sdr::SDR;

class Predictor : public Serializable
{
  // Make test class friend so it can unit test private members directly
  friend class SDRClassifierTest;

public:
  /**
   * Constructor for use when deserializing.
   */
  Predictor() {}
  void initialize(const std::vector<UInt> &steps, Real64 alpha, Real64 actValueAlpha);

  /**
   * Constructor.
   *
   * @param steps The different number of steps to learn and predict.
   * @param alpha The alpha to use when decaying the duty cycles.
   * @param actValueAlpha The alpha to use when decaying the actual
   *                      values for each bucket.
   * @param verbosity The logging verbosity.
   */
  Predictor(const std::vector<UInt> &steps, Real64 alpha, Real64 actValueAlpha);

  /**
   * Destructor.
   */
  virtual ~Predictor();

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
  virtual void compute(UInt recordNum, const SDR &patternNZ,
                       const std::vector<UInt> &bucketIdxList,
                       const std::vector<Real64> &actValueList, bool category,
                       bool learn, bool infer, ClassifierResult &result);

  /**
   * Save the state to the ostream.
   */
  void save(std::ostream &outStream) const override;

  /**
   * Load state from istream.
   */
  void load(std::istream &inStream) override;

  /**
   * Compare the other instance to this one.
   *
   * @param other Another instance of Predictor to compare to.
   * @returns true iff other is identical to this instance.
   */
  virtual bool operator==(const Predictor &other) const;

private:
  // Helper function for inference mode
  void infer_(const SDR &pattern, const std::vector<Real64> &actValue,
              ClassifierResult &result);

  // The list of prediction steps to learn and infer.
  std::vector<UInt> steps_;

  // The alpha used to decay the actual values used for each bucket.
  Real64 actValueAlpha_;

  // The maximum number of the prediction steps.
  UInt maxSteps_;

  // Stores the input pattern history, starting with the previous input
  // and containing _maxSteps total input patterns.
  std::deque<std::vector<UInt>> patternNZHistory_;
  std::deque<UInt> recordNumHistory_;

  // One per prediction step
  std::map<UInt, Classifier> classifiers_;

  // The highest input bit that the classifier has seen so far.
  UInt maxInputIdx_;

  // The highest bucket index that the classifier has been seen so far.
  UInt maxBucketIdx_;

  // The current actual values used for each bucket index. The index of
  // the actual value matches the index of the bucket.
  std::vector<Real64> actualValues_;

  // A boolean that distinguishes between actual values that have been
  // seen and those that have not.
  std::vector<bool> actualValuesSet_;

}; // end of SDRClassifier class

} // end of namespace sdr_classifier
} // end of namespace algorithms
} // namespace nupic

#endif
