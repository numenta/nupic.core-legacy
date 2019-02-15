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
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/utils/SdrMetrics.hpp>

#include <mnist/mnist_reader_less.hpp>

namespace examples {

using namespace std;
using namespace nupic;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::sdr_classifier::SDRClassifier;
using nupic::algorithms::cla_classifier::ClassifierResult;

typedef vector<UInt> image_t;
/*
const vector<UInt> read_mnist_labels(string path) {
    ifstream file(path);
    NTA_CHECK( file.is_open() )  << "ERROR: Failed to open file " << path;

    int magic_number     = 0;
    int number_of_labels = 0;
    file.read( (char*) &magic_number,     4);
    file.read( (char*) &number_of_labels, 4);
    if(magic_number != 0x00000801) {
        std::reverse((char*) &magic_number,      (char*) &magic_number + 4);
        std::reverse((char*) &number_of_labels,  (char*) &number_of_labels + 4);
    }
    NTA_CHECK(magic_number == 0x00000801)  << "ERROR: MNIST data is compressed or corrupt";

    vector<UInt> retval;
    for(int i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read( (char*) &label, 1);
        retval.push_back((UInt) label);
    }
    return retval;
}


const vector<image_t> read_mnist_images(string path) {
    ifstream file(path);
    NTA_CHECK(file.is_open() )  << "ERROR: Failed to open file " << path;

    int magic_number     = 0;
    int number_of_images = 0;
    int n_rows           = 0;
    int n_cols           = 0;
    file.read( (char*) &magic_number,     4);
    file.read( (char*) &number_of_images, 4);
    file.read( (char*) &n_rows,           4);
    file.read( (char*) &n_cols,           4);
    if(magic_number != 0x00000803) {
        std::reverse((char*) &magic_number,      (char*) &magic_number + 4);
        std::reverse((char*) &number_of_images,  (char*) &number_of_images + 4);
        std::reverse((char*) &n_rows,            (char*) &n_rows + 4);
        std::reverse((char*) &n_cols,            (char*) &n_cols + 4);
    }
    NTA_CHECK(magic_number == 0x00000803)  << "ERROR: MNIST data is compressed or corrupt";
    NTA_ASSERT(n_rows == 28);
    NTA_ASSERT(n_cols == 28);

    const UInt img_size = n_rows * n_cols;
    vector<image_t > retval;
    for(int i = 0; i < number_of_images; ++i) {
        vector<unsigned char> data_raw(img_size);
        file.read( (char*) data_raw.data(), img_size);
        // Copy the data into an array of UInt's
        image_t data(img_size);
        // Apply a threshold to the image, yielding a B & W image.
        for(const auto pixel: data_raw) {
            data.push_back(pixel >= 128 ? 1u : 0u);
        }
        retval.push_back(data);
    }
    return retval;
}
*/
class MNIST {

  private:
    SpatialPooler sp;
    SDR input;
    SDR columns;
    SDRClassifier clsr;

    UInt verbosity = 1;
    const UInt train_dataset_iterations = 1u;

  public:

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
  sp.setSynPermConnected(0.4);
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
}

void train() {
  // Train
  const auto train_images = read_mnist_images("./mnist_data/train-images-idx3-ubyte");
  const auto train_labels = read_mnist_labels("./mnist_data/train-labels-idx1-ubyte");

  if(verbosity)
    cout << "Training for " << (train_dataset_iterations * train_labels.size())
         << " cycles ..." << endl;
  size_t i = 0;
  SDR_Metrics columnStats(columns, 1402);

  for(auto epoch = 0u; epoch < train_dataset_iterations; epoch++) {
    // Shuffle the training data.
    NTA_WARN << "epoch " << epoch;
    vector<UInt> index( train_labels.size() );
    index.assign(train_labels.cbegin(), train_labels.cend());
    Random().shuffle( index.begin(), index.end() );

    for(const auto idx : index) { // index = order of label (shuffeled)
      // Get the input & label
      const image_t image = train_images.at(idx);
      const UInt label  = train_labels.at(idx);

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
  auto test_images  = read_mnist_images("./mnist_data/t10k-images-idx3-ubyte");
  auto test_labels  = read_mnist_labels("./mnist_data/t10k-labels-idx1-ubyte");
  Real score = 0;
  UInt n_samples = 0;
  if(verbosity)
    cout << "Testing for " << test_labels.size() << " cycles ..." << endl;
  for(UInt i = 0; i < test_labels.size(); i++) {
    // Get the input & label
    const image_t image = test_images.at(i);
    const UInt label  = test_labels.at(i);

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

