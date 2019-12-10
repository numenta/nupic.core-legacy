/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
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
 * --------------------------------------------------------------------- */

/** @file
 * Definitions for the SDR Classifier & Predictor.
 * 
 * `Classifier` learns mapping from SDR->input value (encoder's output). 
 * This is used when you need to "explain" the HTM network back to real-world, 
 * ie. mapping SDRs back to digits in MNIST digit classification task. 
 *
 * `Predictor` has similar functionality for time-sequences 
 * where you want to "predict" N-steps ahead and then return real-world value. 
 * Internally it uses (several) Classifiers, and in nupic.core this used to be 
 * a part for SDRClassifier, for `htm.core` this is a separate class `Predictor`. 
 *
 */

#ifndef NTA_SDR_CLASSIFIER_HPP
#define NTA_SDR_CLASSIFIER_HPP

#include <deque>
#include <unordered_map>
#include <vector>

#include <htm/types/Types.hpp>
#include <htm/types/Sdr.hpp>
#include <htm/types/Serializable.hpp>

namespace htm {


/**
 * PDF: Probability Distribution Function.  Each index in this vector is a
 *      category label, and each value is the likelihood of the that category.
 *
 * See also:  https://en.wikipedia.org/wiki/Probability_distribution
 */
using PDF = std::vector<Real64>; //Real64 (not Real/float) must be used here, 
// ... otherwise precision is lost and Predictor never reaches sufficient results.

/**
 * Returns the category with the greatest probablility.
 */
UInt argmax( const PDF & data );

/**
 * The SDR Classifier takes the form of a single layer classification network.
 * It accepts SDRs as input and outputs a predicted distribution of categories.
 *
 * Categories are labeled using unsigned integers.  Other data types must be
 * enumerated or transformed into postitive integers.  There are as many output
 * units as the maximum category label.
 *
 * Example Usage:
 *
 *    // Make a random SDR and associate it with the category B.
 *    SDR inputData({ 1000 });
 *        inputData.randomize( 0.02 );
 *    enum Category { A, B, C, D };
 *    Classifier clsr;
 *    clsr.learn( inputData, { Category::B } );
 *    argmax( clsr.infer( inputData ) )  ->  Category::B
 *
 *    // Estimate a scalar value.  The Classifier only accepts categories, so
 *    // put real valued inputs into bins (AKA buckets) by subtracting the
 *    // minimum value and dividing by a resolution.
 *    double scalar = 567.8;
 *    double minimum = 500;
 *    double resolution = 10;
 *    clsr.learn( inputData, { (scalar - minimum) / resolution } );
 *    argmax( clsr.infer( inputData ) ) * resolution + minimum  ->  560
 *
 * During inference, the output is calculated by first doing a weighted
 * summation of all the inputs, and then perform a softmax nonlinear function to
 * get the predicted distribution of category labels.
 *
 * During learning, the connection weights between input units and output units
 * are adjusted to maximize the likelihood of the model.
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
   * @param alpha - The alpha used to adapt the weight matrix during learning. 
   *                A larger alpha results in faster adaptation to the data.
   *                Note: when SDRs are formed correctly, the classification task 
   *                for this class is quite easy, so you likely will never need to 
   *                optimize this parameter. 
   */
  Classifier(Real alpha = 0.001f );

  /**
   * For use when deserializing.
   */
  void initialize(Real alpha);

  /**
   * Compute the likelihoods for each category / bucket.
   *
   * @param pattern: The SDR containing the active input bits.
   * @returns: The Probablility Distribution Function (PDF) of the categories.
   *           This is indexed by the category label.
   *           Or empty array ([]) if Classifier hasn't called learn() before. 
   */
  PDF infer(const SDR & pattern) const;

  /**
   * Learn from example data.
   *
   * @param pattern:  The active input bit SDR.
   * @param categoryIdxList:  The current categories or bucket indices.
   */
  void learn(const SDR & pattern, const std::vector<UInt> & categoryIdxList);

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
  void load_ar(Archive & ar) {
    ar(cereal::make_nvp("alpha", alpha_), 
       cereal::make_nvp("dimensions", dimensions_),
       cereal::make_nvp("numCategories", numCategories_), 
       cereal::make_nvp("weights", weights_));
  }

  bool operator==(const Classifier &other) const;
  bool operator!=(const Classifier &other) const { return !operator==(other); }

private:
  Real alpha_;
  UInt dimensions_;
  UInt numCategories_;

  /**
   * 2D map used to store the data.
   * Use as: weights_[ input-bit ][ category-index ]
   * Real64 (not just Real) so the computations do not lose precision.
   */
  std::vector<std::vector<Real64>> weights_;

  // Helper function to compute the error signal for learning.
  std::vector<Real64> calculateError_(const std::vector<UInt> &bucketIdxList,
                                      const SDR &pattern) const;
};

/**
 * Helper function for Classifier::infer.  Converts the raw data accumulators
 * into a PDF.
 */
void softmax(PDF::iterator begin, PDF::iterator end);


/******************************************************************************/


/**
 * The key is the step, for predicting multiple time steps into the future.
 * The value is a PDF (probability distribution function, of the result being in
 * each bucket or category).
 */
using Predictions = std::unordered_map<UInt, PDF>;

/**
 * The Predictor class does N-Step ahead predictions.
 *
 * Internally, this class uses Classifiers to associate SDRs with future values.
 * This class handles missing datapoints.
 *
 * Compatibility Note:  This class is the replacement for the old SDRClassifier.
 * It no longer provides estimates of the actual value. Instead, users can get a rough estimate
 * from bucket-index. If more precision is needed, use more buckets in the encoder. 
 *
 * Example Usage:
 *   ```
 *    // Predict 1 and 2 time steps into the future.
 *    // Make a sequence of 4 random SDRs. Each SDR has 1000 bits and 2% sparsity.
 *    vector<SDR> sequence( 4, { 1000 } );
 *    for( SDR & inputData : sequence )
 *        inputData.randomize( 0.02 );
 *
 *    // Make category labels for the sequence.
 *    vector<UInt> labels = { 4, 5, 6, 7 };
 *
 *    // Make a Predictor and train it.
 *    Predictor pred( vector<UInt>{ 1, 2 } );
 *    pred.learn( 0, sequence[0], { labels[0] } );
 *    pred.learn( 1, sequence[1], { labels[1] } );
 *    pred.learn( 2, sequence[2], { labels[2] } );
 *    pred.learn( 3, sequence[3], { labels[3] } );
 *
 *    // Give the predictor partial information, and make predictions
 *    // about the future.
 *    pred.reset();
 *    Predictions A = pred.infer( sequence[0] );
 *    argmax( A[1] )  ->  labels[1]
 *    argmax( A[2] )  ->  labels[2]
 *
 *    Predictions B = pred.infer( sequence[1] );
 *    argmax( B[1] )  ->  labels[2]
 *    argmax( B[2] )  ->  labels[3]
 *    ```
 */
class Predictor : public Serializable
{
public:
  /**
   * Constructor.
   *
   * @param steps - The number of steps into the future to learn and predict.
   * @param alpha - The alpha used to adapt the weight matrix during learning. 
   *                A larger alpha results in faster adaptation to the data.
   *                (The default value will likely be OK in most cases.)
   */
  Predictor(const std::vector<UInt> &steps, Real alpha = 0.001f );

  /**
   * Constructor for use when deserializing.
   */
  Predictor() {}
  void initialize(const std::vector<UInt> &steps, Real alpha = 0.001f );

  /**
   * For use with time series datasets.
   */
  void reset();

  /**
   * Compute the likelihoods.
   *
   * @param pattern: The active input SDR.
   *
   * @returns: A mapping from prediction step to PDF.
   */
  Predictions infer(const SDR &pattern) const;

  /**
   * Learn from example data.
   *
   * @param recordNum: An incrementing integer for each record. Gaps in
   *                   numbers correspond to missing records.
   * @param pattern: The active input SDR.
   * @param bucketIdxList: Vector of the current value bucket indices or categories.
   */
  void learn(const UInt recordNum, 
	     const SDR &pattern,
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

  // Stores the input pattern history, starting with the previous input.
  std::deque<SDR>  patternHistory_;
  std::deque<UInt> recordNumHistory_;
  void checkMonotonic_(UInt recordNum) const;

  // One per prediction step
  std::unordered_map<UInt, Classifier> classifiers_;

};      // End of Predictor class

}       // End of namespace htm
#endif  // End of ifdef NTA_SDR_CLASSIFIER_HPP
