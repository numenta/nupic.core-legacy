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
#include <random>

#include "nupic/algorithms/SpatialPooler.hpp"
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/algorithms/ClassifierResult.hpp>
#include "nupic/utils/SdrMetrics.hpp"

using namespace std;
using namespace nupic;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::sdr_classifier::SDRClassifier;
using nupic::algorithms::cla_classifier::ClassifierResult;


vector<UInt> read_mnist_labels(string path) {
    ifstream file(path);
    if( !file.is_open() ) {
        cerr << "ERROR: Failed to open file " << path << endl;
        exit(1);
    }
    int magic_number     = 0;
    int number_of_labels = 0;
    file.read( (char*) &magic_number,     4);
    file.read( (char*) &number_of_labels, 4);
    if(magic_number != 0x00000801) {
        std::reverse((char*) &magic_number,      (char*) &magic_number + 4);
        std::reverse((char*) &number_of_labels,  (char*) &number_of_labels + 4);
    }
    if(magic_number != 0x00000801) {
        cerr << "ERROR: MNIST data is compressed or corrupt" << endl;
        exit(1);
    }
    vector<UInt> retval;
    for(int i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read( (char*) &label, 1);
        retval.push_back((UInt) label);
    }
    return retval;
}


vector<UInt*> read_mnist_images(string path) {
    ifstream file(path);
    if( !file.is_open() ) {
        cerr << "ERROR: Failed to open file " << path << endl;
        exit(1);
    }
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
    if(magic_number != 0x00000803) {
        cerr << "ERROR: MNIST data is compressed or corrupt" << endl;
        exit(1);
    }
    NTA_ASSERT(n_rows == 28);
    NTA_ASSERT(n_cols == 28);
    UInt img_size = n_rows * n_cols;
    vector<UInt*> retval;
    for(int i = 0; i < number_of_images; ++i) {
        auto data_raw = new unsigned char[img_size];
        file.read( (char*) data_raw, img_size);
        // Copy the data into an array of UInt's
        auto data = new UInt[img_size];
        // auto data = new UInt[2 * img_size];
        // Apply a threshold to the image, yielding a B & W image.
        for(UInt pixel = 0; pixel < img_size; pixel++) {
            data[pixel] = data_raw[pixel] >= 128 ? 1 : 0;
            // data[2 * pixel] = data_raw[pixel] >= 128 ? 1 : 0;
            // data[2 * pixel + 1] = 1 - data[2 * pixel];
        }
        retval.push_back(data);
        delete[] data_raw;
    }
    return retval;
}


int main(int argc, char **argv) {
  UInt verbosity = 1;
  auto train_dataset_iterations = 1u;
  int opt;
  while ( (opt = getopt(argc, argv, "tv")) != -1 ) {  // for each option...
    switch ( opt ) {
      case 't':
          train_dataset_iterations += 1;
        break;
      case 'v':
          verbosity = 1;
        break;
      case '?':
          cerr << "Unknown option: '" << char(optopt) << "'!" << endl;
        break;
    }
  }
  UInt train_time = train_dataset_iterations * 60000;

  SDR input({28, 28, 2});
  SpatialPooler sp(
    /* numInputs */                    input.dimensions,
    /* numColumns */                   {10, 10, 120},
    /* potentialRadius */              0,  // hardcoded elsewhere
    /* potentialPct */                 .0000001, // hardcoded elsewhere
    /* globalInhibition */             true,
    /* localAreaDensity */             .015,
    /* numActiveColumnsPerInhArea */   -1,
    /* stimulusThreshold */             28,
    /* synPermInactiveDec */           .00928,
    /* synPermActiveInc */             .032,
    /* synPermConnected */             .422,
    /* minPctOverlapDutyCycles */      0.,
    /* dutyCyclePeriod */              1402,
    /* boostStrength */                0,
    /* CPP SP seed */                  0,
    /* spVerbosity */                  verbosity,
    /* wrapAround */                   0 // discarded
    );


  SDR columns({sp.getNumColumns()});
  SDR_Metrics columnStats(columns, 1402);

  SDRClassifier clsr(
    /* steps */         {0},
    /* alpha */         .001,
    /* actValueAlpha */ .3,
                        verbosity);

  // Train
  auto train_images = read_mnist_images("./mnist_data/train-images-idx3-ubyte");
  auto train_labels = read_mnist_labels("./mnist_data/train-labels-idx1-ubyte");
  if(verbosity)
    cout << "Training for " << (train_dataset_iterations * train_labels.size())
         << " cycles ..." << endl;
  for(auto i = 0u; i < train_dataset_iterations; i++) {
    // Shuffle the training data.
    vector<UInt> index( train_labels.size() );
    for(auto s = 0u; s < train_labels.size(); s++)
        index[s] = s;
    Random().shuffle( index.begin(), index.end() );

    for(auto s = 0u; s < train_labels.size(); s++) {
      // Get the input & label
      UInt *image = train_images[ index[s] ];
      UInt label  = train_labels[ index[s] ];

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
      if( verbosity and i % 100 == 0 )
        cout << "." << flush;
  }
  if( verbosity ) cout << endl;
  }
  cout << columnStats << endl;

  // Test
  auto test_images  = read_mnist_images("./mnist_data/t10k-images-idx3-ubyte");
  auto test_labels  = read_mnist_labels("./mnist_data/t10k-labels-idx1-ubyte");
  Real score = 0;
  UInt n_samples = 0;
  if(verbosity)
    cout << "Testing for " << test_labels.size() << " cycles ..." << endl;
  for(UInt i = 0; i < test_labels.size(); i++) {
    // Get the input & label
    UInt *image = test_images[i];
    UInt label  = test_labels[i];

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
          auto *pdf = iter.second;
          auto max  = std::max_element(pdf->begin(), pdf->end());
          UInt cls  = max - pdf->begin();
          if(cls == label)
            score += 1;
          n_samples += 1;
      }
    }
    if( verbosity and i % 100 == 0 )
      cout << "." << flush;
  }
  if( verbosity ) cout << endl;
  cout << "Score: " << score / n_samples << endl;
}
