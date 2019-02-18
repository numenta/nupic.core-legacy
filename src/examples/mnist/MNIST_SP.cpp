/* ---------------------------------------------------------------------
 * Copyright (C) 2018, David McDougall.
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
 * ----------------------------------------------------------------------
 */

/**
 * Solving the MNIST dataset with Spatial Pooler.
 *
 * This consists of a simple black & white image encoder, a spatial pool, and an
 * SDR classifier.  The task is to recognise images of hand written numbers 0-9.
 * This should score at least 95%.
 */

#include <algorithm>
#include <cstdint> //uint8_t
#include <iostream>
#include <vector>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/utils/SdrMetrics.hpp>

#include <mnist/mnist_reader_less.hpp> // MNIST data itself + read methods, namespace mnist::

namespace examples {

using namespace std;
using namespace nupic;

using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::sdr_classifier::SDRClassifier;
using nupic::algorithms::cla_classifier::ClassifierResult;

class MNIST {

  private:
    SpatialPooler sp;
    SDR input;
    SDR columns;
    SDRClassifier clsr;
    mnist::MNIST_dataset<uint8_t, uint8_t> dataset;

  public:
    UInt verbosity = 1;
    const UInt train_dataset_iterations = 1u;


void setup() {

  input.initialize({28, 28, 1});
  sp.initialize(
    /* numInputs */                    input.dimensions,
    /* numColumns */                   {7, 7, 8}
    );
  sp.setGlobalInhibition(false);
  // mapping to input
  sp.setPotentialRadius(5); //receptive field (hyper-cube) how each col maps to input
  sp.setPotentialPct(0.5); //of the recept. field above, how much % of inputs individual col connects to?
  sp.setStimulusThreshold(0); //FIXME no learning if this > 0
  sp.setSynPermActiveInc(0.2);
  sp.setSynPermInactiveDec(0.02);

  sp.setSpVerbosity(verbosity);
  // on columnar level
  sp.setLocalAreaDensity(.05); // 5% sparsity
  sp.setMinPctOverlapDutyCycles(0.4);
  //boost
  sp.setBoostStrength(1.8);
  sp.setDutyCyclePeriod(1000);

  sp.setWrapAround(false);


  columns.initialize({sp.getNumColumns()});

  clsr.initialize(
    /* steps */         {0},
    /* alpha */         .001,
    /* actValueAlpha */ .3,
                        verbosity);

  dataset = mnist::read_dataset<uint8_t, uint8_t>(string(MNIST_DATA_LOCATION)); //from CMake
}

void train() {
  // Train

  if(verbosity)
    cout << "Training for " << (train_dataset_iterations * dataset.training_labels.size())
         << " cycles ..." << endl;
  size_t i = 0;
  SDR_Metrics columnStats(columns, 1402);

  for(auto epoch = 0u; epoch < train_dataset_iterations; epoch++) {
    NTA_WARN << "epoch " << epoch;
    // Shuffle the training data.
    vector<UInt> index( dataset.training_labels.size() );
    index.assign(dataset.training_labels.cbegin(), dataset.training_labels.cend());
    Random().shuffle( index.begin(), index.end() );

    for(const auto idx : index) { // index = order of label (shuffeled)
      // Get the input & label
      const auto image = dataset.training_images.at(idx);
      const UInt label  = dataset.training_labels.at(idx);

      // Compute & Train
      input.setDense( image );
      sp.compute(input, true, columns);
      ClassifierResult result;
      clsr.compute(sp.getIterationNum(), columns.getFlatSparse(),
        /* bucketIdxList */   {label},
        /* actValueList */    {(Real)label},
        /* category */        true,
        /* learn */           true,
        /* infer */           false,
                              &result);
      if( verbosity && (++i % 1000 == 0) ) cout << "." << flush;
  }
  if( verbosity ) cout << endl;
  cout << "epoch ended" << endl;
  cout << columnStats << endl;
  }
}

void test() {
  // Test
  Real score = 0;
  UInt n_samples = 0;
  if(verbosity)
    cout << "Testing for " << dataset.test_labels.size() << " cycles ..." << endl;
  for(UInt i = 0; i < dataset.test_labels.size(); i++) {
    // Get the input & label
    const auto image  = dataset.test_images.at(i);
    const UInt label  = dataset.test_labels.at(i);

    // Compute
    input.setDense( image );
    sp.compute(input, false, columns);
    ClassifierResult result;
    clsr.compute(sp.getIterationNum(), columns.getFlatSparse(),
      /* bucketIdxList */   {},
      /* actValueList */    {},
      /* category */        true,
      /* learn */           false,
      /* infer */           true,
                            &result);
    // Check results
    for(auto iter : result) {
      if( iter.first == 0 ) {
          const auto *pdf = iter.second;
          const auto max  = std::max_element(pdf->cbegin(), pdf->cend());
          const UInt cls  = max - pdf->cbegin();
          if(cls == label)
            score += 1;
          n_samples += 1;
      }
    }
    if( verbosity && i % 1000 == 0 ) cout << "." << flush;
  }
  if( verbosity ) cout << endl;
  cout << "Score: " << 100.0 * score / n_samples << "% " << endl;
}
};
}

int main(int argc, char **argv) {
  examples::MNIST m;
  m.setup();
  m.train();
  m.test();

  return 0;
}

