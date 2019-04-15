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
 * Definitions for the SDR Classifier & Predictor.
 */

#ifndef NTA_SDR_CLASSIFIER_HPP
#define NTA_SDR_CLASSIFIER_HPP

#include <deque>
#include <map>
#include <vector>

#include <nupic/types/Types.hpp>
#include <nupic/types/Sdr.hpp>

#include <nupic/types/Serializable.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/map.hpp>

namespace nupic {
namespace algorithms {
namespace sdr_classifier {


/**
 * PDF - Probability Distribution Function, distribution of likelihood of values
 *       for each category.
 *
 * See also:  https://en.wikipedia.org/wiki/Probability_distribution
 */
using PDF = std::vector<Real>;

/**
 * Returns the class with the single greatest probablility.
 */
UInt argmax( const PDF & data );


/**
 * The SDR Classifier takes the form of a single layer classification network
 * that takes SDRs as input and outputs a predicted distribution of classes.
 *
 * The SDR Classifier accepts an SDR input pattern from the level below (the
 * “pattern”) and information from the sensor and encoders (the
 * “classification”) describing the true (target) input.
 *
 * The SDR classifier maps input patterns to class labels. There are as many
 * output units as the maximum class label or bucket (in the case of scalar
 * encoders). The output is a probabilistic distribution over all class labels.
 *
 * During inference, the output is calculated by first doing a weighted
 * summation of all the inputs, and then perform a softmax nonlinear function to
 * get the predicted distribution of class labels
 *
 * During learning, the connection weights between input units and output units
 * are adjusted to maximize the likelihood of the model
 *
 * Example Usage: TODO
 *
 * References:
 *  - Alex Graves. Supervised Sequence Labeling with Recurrent Neural Networks,
 *    PhD Thesis, 2008
 *  - J. S. Bridle. Probabilistic interpretation of feedforward classification
 *    network outputs, with relationships to statistical pattern recognition
 *  - In F. Fogleman-Soulie and J.Herault, editors, Neurocomputing: Algorithms,
 *    Architectures and Applications, pp 227-236, Springer-Verlag, 1990
 */
class Classifier : public Serializable
{
public:
  /**
   * Constructor.
   *
   * @param alpha - The alpha used to adapt the weight matrix during learning. A
   *                larger alpha results in faster adaptation to the data.
   */
  Classifier(Real alpha);

  /**
   * Constructor for use when deserializing.
   */
  Classifier() {}
  void initialize(Real alpha);

  /**
   * Compute the likelihoods for each category / bucket.
   *
   * @param pattern: The active input bit SDR.
   * @returns: The Probablility Density Function (PDF) of the categories.
   */
  PDF infer(const sdr::SDR & pattern);

  /**
   * Learn from example data.
   *
   * @param pattern:  The active input bit SDR.
   * @param categoryIdxList:  The current categories or bucket indices.
   */
  void learn(const sdr::SDR & pattern, const std::vector<UInt> & categoryIdxList);

  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const
  {
    ar(cereal::make_nvp("alpha",         alpha_),
       cereal::make_nvp("dimensions",    dimensions_),
       cereal::make_nvp("numCategories", numCategories_),
       cereal::make_nvp("weights",       weights_));
  }

  template<class Archive>
  void load_ar(Archive & ar)
    { ar( alpha_, dimensions_, numCategories_, weights_ ); }

private:
  Real alpha_;
  std::vector<UInt> dimensions_;
  UInt numCategories_;

  /**
   * 2D map used to store the data.
   * Use as: weights_[ input-bit ][ category-index ]
   */
  std::vector<std::vector<Real>> weights_;

  // Helper function to compute the error signal for learning.
  std::vector<Real> calculateError_(const std::vector<UInt> &bucketIdxList,
                                    const sdr::SDR &pattern);
};

/**
 * Helper function for Classifier::infer.  Converts the raw data accumulators
 * into a PDF.
 */
void softmax(PDF::iterator begin, PDF::iterator end);


/******************************************************************************/

/**
 * The key is the step, for predicting multiple time steps into the future.
 * The value is a PDF (probability distribution function, list of probabilities
 * of outcomes) of the result being in each bucket.
 */
using Predictions = std::map<Int, PDF>;


/**
 * The Predictor class does N-Step ahead predictions.
 *
 * Internally, this class uses Classifiers to associate SDRs with future values.
 * This class handles missing datapoints.
 *
 * Compatibility Note:  This class is the replacement for the old SDRClassifier.
 * It no longer provides estimates of the actual value.
 */
class Predictor : public Serializable
{
public:
  /**
   * Constructor.
   *
   * @param steps - The different number of steps to learn and predict.
   * @param alpha - The alpha used to adapt the weight matrix during learning. A
   *                larger alpha results in faster adaptation to the data.
   */
  Predictor(const std::vector<UInt> &steps, Real alpha);

  /**
   * Constructor for use when deserializing.
   */
  Predictor() {}
  void initialize(const std::vector<UInt> &steps, Real alpha);

  /**
   * For use with time series datasets.
   */
  void reset();

  /**
   * Compute the likelihoods for each bucket.
   *
   * @param recordNum: An incrementing integer for each record. Gaps in
   *                   numbers correspond to missing records.
   *
   * @param pattern: The active input SDR.
   *
   * @returns: A mapping from prediction step to a vector of likelihoods where
   *           the value at an index corresponds to the bucket with the same
   *           index.
   */
  Predictions infer(UInt recordNum, const sdr::SDR &pattern);

  /**
   * Learn from example data.
   *
   * @param recordNum: An incrementing integer for each record. Gaps in
   *                   numbers correspond to missing records.
   * @param pattern: The active input SDR.
   * @param bucketIdxList: Vector of the current value bucket indices or categories.
   */
  void learn(UInt recordNum, const sdr::SDR &pattern,
             const std::vector<UInt> &bucketIdxList);

  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const
  {
    ar(cereal::make_nvp("steps",            steps_),
       cereal::make_nvp("patternHistory",   patternHistory_),
       cereal::make_nvp("recordNumHistory", recordNumHistory_),
       cereal::make_nvp("classifiers",      classifiers_));
  }

  template<class Archive>
  void load_ar(Archive & ar)
    { ar( steps_, patternHistory_, recordNumHistory_, classifiers_ ); }

private:
  // The list of prediction steps to learn and infer.
  std::vector<UInt> steps_;

  // Stores the input pattern history, starting with the previous input
  // and containing _maxSteps total input patterns.
  std::deque<sdr::SDR> patternHistory_;
  std::deque<UInt>     recordNumHistory_;
  void updateHistory_(UInt recordNum, const sdr::SDR & pattern);

  // One per prediction step
  std::map<UInt, Classifier> classifiers_;

};      // End of Predictor class

}       // End of namespace sdr_classifier
}       // End of namespace algorithms
}       // End of namespace nupic
#endif  // End of ifdef NTA_SDR_CLASSIFIER_HPP
