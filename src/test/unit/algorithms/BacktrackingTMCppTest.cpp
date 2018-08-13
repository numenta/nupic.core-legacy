/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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



 * Author: David Keeney, April, 2018
 * ---------------------------------------------------------------------
 */

/*---------------------------------------------------------------------
 * This is a test of the backtrackingTMCpp C++ implimentation (no Python).
 *
 * For those not familiar with GTest:
 * ASSERT_TRUE(value)   -- Fatal assertion that the value is true. Test
 *                         terminates if false.
 * ASSERT_FALSE(value)   -- Fatal assertion that the value
 *                         is false. Test terminates if true.
 * ASSERT_STREQ(str1, str2)   -- Fatal assertion that the strings are equal.
 *                         Test terminates if false.
 * EXPECT_TRUE(value)   -- Nonfatal assertion that the value is true. Test
 *                         continues if false.
 * EXPECT_FALSE(value)   -- Nonfatal assertion that the value is false.
 *                          Test continues if true.
 * EXPECT_STREQ(str1, str2) -- Nonfatal assertion that the strings are equal.
 *                          Test continues if false.
 * EXPECT_THROW(statement, exception_type) -- nonfatal exception, cought,
 *                         reported and continues.
 *---------------------------------------------------------------------
 */

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <nupic/algorithms/BacktrackingTMCpp.hpp>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/types/Exception.hpp>


#include <gtest/gtest.h>

#define VERBOSITY 0
#define VERBOSE if (verbose) std::cerr << "[          ] "
static bool verbose =  true; // turn this on to print extra stuff for debugging the test.
#define SEED 12

using namespace nupic;
using namespace nupic::algorithms::backtracking_tm;
namespace testing {

////////////////////////////////////////////////////////////////////////////////
//     helper routines
////////////////////////////////////////////////////////////////////////////////
typedef shared_ptr<Real32> Pattern_t;


// Generate a single test pattern of random bits with given parameters.
//
//    Parameters :
//    @numCols -Number of columns in each pattern.
//    @minOnes -The minimum number of 1's in each pattern.
//    @maxOnes -The maximum number of 1's in each pattern.

static Pattern_t generatePattern(Size numCols = 100, Size minOnes = 21,
                                 Size maxOnes = 25) {
  NTA_ASSERT(minOnes <= maxOnes);
  NTA_ASSERT(maxOnes < numCols);
  Pattern_t p(new Real32[numCols], std::default_delete<Real32[]>());
  memset(p.get(), 0, numCols);

  int numOnes = (int)minOnes + std::rand() % (maxOnes - minOnes + 1);
  std::uniform_int_distribution<> distr(0, (int)numCols - 1); // define the range (requires C++11)
  for (int n = 0; n < numOnes; n++) {
    int idx = std::rand() % numCols;
    p.get()[idx] = 1.0f;
  }
  return p;
}

// Generates a sequence of n random patterns.
static std::vector<Pattern_t> generateSequence(Size n = 10, Size numCols = 100,
                                               Size minOnes = 21,
                                               Size maxOnes = 25) {
  std::vector<Pattern_t> seq;
  Pattern_t p;  // an empty pattern which means reset
  seq.push_back(p);

  for (Size i = 0; i < n; i++) {
    Pattern_t p = generatePattern(numCols, minOnes, maxOnes);
    seq.push_back(p);
  }
  return seq;
}


//  Static function nonzero()
// returns an array of the indexes of the non-zero elements.
// The first element is the number of elements in the rest of the array.
// Dynamically allocated, caller must delete.
template <typename T>
static UInt32 *nonzero(const T *dence_buffer, Size len) {
  UInt32 *nz = new UInt32[len + 1];
  nz[0] = 1;
  for (Size n = 0; n < len; n++) {
    if (dence_buffer[n] != (T)0)
      nz[nz[0]++] = (UInt32)n;
  }
  return nz;
}



struct param_t {
  UInt32 numberOfCols;
  UInt32 cellsPerColumn;
  Real32 initialPerm;
  Real32 connectedPerm;
  UInt32 minThreshold;
  UInt32 newSynapseCount;
  Real32 permanenceInc;
  Real32 permanenceDec;
  Real32 permanenceMax;
  Real32 globalDecay;
  UInt32 activationThreshold;
  bool doPooling;
  UInt32 segUpdateValidDuration;
  UInt32 burnIn;
  bool collectStats;
  Int32 seed;
  Int32 verbosity;
  bool checkSynapseConsistency;
  UInt32 pamLength;
  UInt32 maxInfBacktrack;
  UInt32 maxLrnBacktrack;
  UInt32 maxAge;
  UInt32 maxSeqLength;
  Int32 maxSegmentsPerCell;
  Int32 maxSynapsesPerSegment;
  char outputType[25];
};

void initializeParameters(struct param_t &param) {
  // same as default settings
  param.numberOfCols = 10;
  param.cellsPerColumn = 3;
  param.initialPerm = 0.11f;
  param.connectedPerm = 0.5f;
  param.minThreshold = 8;
  param.newSynapseCount = 15;
  param.permanenceInc = 0.1f;
  param.permanenceDec = 0.1f;
  param.permanenceMax = 1.0f;
  param.globalDecay = 0.1f;
  param.activationThreshold = 12;
  param.doPooling = false;
  param.segUpdateValidDuration = 5;
  param.burnIn = 2;
  param.collectStats = false;
  param.seed = 42;
  param.verbosity = VERBOSITY;
  param.checkSynapseConsistency = false;
  param.pamLength = 1;
  param.maxInfBacktrack = 10;
  param.maxLrnBacktrack = 5;
  param.maxAge = 100000;
  param.maxSeqLength = 32;
  param.maxSegmentsPerCell = -1;
  param.maxSynapsesPerSegment = -1;
  strcpy(param.outputType, "normal");
}

//////////////////////////////////////////////////////////////////
//////////    Tests
//////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// Run a test of the basic TM class ported from backtracking_tm_test.py
TEST(BacktrackingTMTest, testCheckpointLearned) {

  struct param_t param;
  initializeParameters(param);
  param.numberOfCols = 100;
  param.cellsPerColumn = 12;
  param.verbosity = VERBOSITY;

  try {
    // Create TM object
    BacktrackingTMCpp tm1(
        param.numberOfCols, param.cellsPerColumn, param.initialPerm,
        param.connectedPerm, param.minThreshold, param.newSynapseCount,
        param.permanenceInc, param.permanenceDec, param.permanenceMax,
        param.globalDecay, param.activationThreshold, param.doPooling,
        param.segUpdateValidDuration, param.burnIn, param.collectStats,
        param.seed, param.verbosity, param.checkSynapseConsistency,
        param.pamLength, param.maxInfBacktrack, param.maxLrnBacktrack,
        param.maxAge, param.maxSeqLength, param.maxSegmentsPerCell,
        param.maxSynapsesPerSegment, param.outputType);

    // generate 5 sets of sequences
    // The first pattern in each sequence is empty (a reset)
    std::vector<std::vector<Pattern_t>> sequences;
    for (Size s = 0; s < 5; s++) {
      sequences.push_back(generateSequence());
    }
    // train with the first 3 sets of sequences.
    std::vector<Pattern_t> train;
    for (Size s = 0; s < 3; s++) {
      for (Size i = 0; i < sequences[s].size();i++) {
        train.push_back(sequences[s][i]);
      }
    }

    // process the patterns that are in the training set.
    for (Size t = 0; t < train.size(); t++) {
      if (!train[t]) {
        tm1.reset();
      } else {
        Real *bottomUpInput = train[t].get();
        tm1.compute(bottomUpInput, true, true);
      }
    }
    // Serialize and deserialized the TM.
    Directory::create("TestOutputDir", false, true);
    std::string checkpointPath = "TestOutputDir/tm.save";
    tm1.saveToFile(checkpointPath);

    BacktrackingTMCpp tm2;
    tm2.loadFromFile(checkpointPath);

    // Check that the TMs are the same.
    ASSERT_TRUE(BacktrackingTMCpp::tmDiff2(tm1, tm2, std::cout, 2));

    // Feed remaining data into the models.
    train.clear();
    for (Size s = 3; s < sequences.size(); s++) {
      for (Size i = 0; i < sequences[s].size(); i++) {
        train.push_back(sequences[s][i]);
      }
    }

    for (Size t = 0; t < train.size(); t++) {
      if (!train[t]) {
        tm1.reset();
        tm2.reset();
      } else {
        Real *bottomUpInput = train[t].get();
        Real *result1 = tm1.compute(bottomUpInput, true, true);
        Real *result2 = tm2.compute(bottomUpInput, true, true);

        ASSERT_TRUE(tm1 == tm2);
        ASSERT_TRUE(tm1.getOutputBufferSize() == tm2.getOutputBufferSize());
        for (Size i = 0; i < tm1.getOutputBufferSize(); i++) {
          ASSERT_TRUE(result1[i] == result2[i]);
        }
      }
    }
  }
  catch (nupic::Exception &ex) {
    FAIL() << "Failure: Exception: " << ex.getFilename() << "("
           << ex.getLineNumber() << ") " << ex.getMessage() << "" << std::endl;
  }
  catch (std::exception &e) {
    FAIL() << "Failure: Exception: " << e.what() << "" << std::endl;
  }

    // cleanup .
  Directory::removeTree("TestOutputDir");

}

TEST(BacktrackingTMTest, testCheckpointMiddleOfSequence)
{
    // Create a model and give it some inputs to learn.
    BacktrackingTMCpp tm1(100, 12);
    tm1.setVerbosity((Int32)VERBOSITY);


    // generate 5 sets of sequences
    std::vector<std::vector<Pattern_t>> sequences;
    for (Size s = 0; s < 5; s++) {
      sequences.push_back(generateSequence());
    }
    // train with the first 3-1/2 sets of sequences.
    std::vector<Pattern_t> train;
    for (Size s = 0; s < 3; s++) {
      for (Size i = 0; i < sequences[s].size(); i++) {
        train.push_back(sequences[s][i]);
      }
    }
    for (Size i = 0; i < 5; i++) {
      train.push_back(sequences[3][i]);
    }

    // compute each of the patterns in train
    for (Size t = 0; t < train.size(); t++) {
      Real *bottomUpInput1 = train[t].get();
      if (!train[t]) {
        tm1.reset();
      } else {
        Real *bottomUpInput = train[t].get();
        tm1.compute(bottomUpInput, true, true);
      }
    }

    // Serialize and TM.
    Directory::create("TestOutputDir", false, true);
    std::string checkpointPath = "TestOutputDir/tm.save";
    tm1.saveToFile(checkpointPath);

    // Prepair the remaining data.
    train.clear();
    for (Size i = 5; i < sequences[3].size(); i++) {
      train.push_back(sequences[3][i]);
    }
    for (Size s = 4; s < sequences.size(); s++) {
      for (Size i = 0; i < sequences[s].size(); i++) {
        train.push_back(sequences[s][i]);
      }
    }

    // process the remaining patterns in train with the first TM.
    for (Size t = 0; t < train.size(); t++) {
      if (!train[t]) {
        tm1.reset();
      } else {
        Real *bottomUpInput = train[t].get();
        Real *result1 = tm1.compute(bottomUpInput, true, true);
      }
    }

    // Restore the saved TM into tm2.
    // Note that this resets the random generator to the same
    // point that the first TM used when it processed that second set
    // of patterns.
    BacktrackingTMCpp tm2;
    tm2.loadFromFile(checkpointPath);

    // process the same remaining patterns in the train with the second TM.
    for (Size t = 0; t < train.size(); t++) {
      if (!train[t]) {
        tm2.reset();
      } else {
        Real *bottomUpInput = train[t].get();
        Real *result1 = tm2.compute(bottomUpInput, true, true);
      }
    }

    // Check that the TMs are the same.
    ASSERT_TRUE(tm1 == tm2);

    // cleanup if successful.
    Directory::removeTree("TestOutputDir");

 }

////////////////////////////////////////////////////////////////////////////////
// Run a test of the TM class ported from backtracking_tm_cpp2_test.py
TEST(BacktrackingTMTest, basicTest) {
    // Create a model and give it some inputs to learn.
    BacktrackingTMCpp tm1(10, 3, 0.2f, 0.8f, 2, 5, 0.10f, 0.05f, 1.0f, 0.05f, 4,
                          false, 5, 2, false, SEED, (Int32)VERBOSITY /* rest are defaults */);
    tm1.setRetrieveLearningStates(true);
    Size nCols = tm1.getnumCol();

    // Serialize and deserialized the TM.
    Directory::create("TestOutputDir", false, true);
    std::string checkpointPath = "TestOutputDir/tm.save";
    tm1.saveToFile(checkpointPath);

    BacktrackingTMCpp tm2;
    tm2.loadFromFile(checkpointPath);

    // Check that the TMs are the same.
    ASSERT_TRUE(BacktrackingTMCpp::tmDiff2(tm1, tm2, std::cout, 2));

    // generate some test data and NT patterns from it.
    std::vector<Pattern_t> data = generateSequence(10, nCols, 2, 2);
    std::vector<const UInt32 *> nzData;
    for (Size i = 0; i < data.size(); i++) {
      if (data[i]) { // skip reset patterns
        UInt32 *indexes = nonzero<Real>(data[i].get(), nCols);
        nzData.push_back(indexes);
      }
    }


    // Learn
    for (Size i = 0; i < 5; i++) {
      if (data[i])
        tm1.learn(data[i].get());
    }
    tm1.reset();

    // save and reload again after learning.
    tm1.saveToFile(checkpointPath);
    BacktrackingTMCpp tm3;
    tm3.loadFromFile(checkpointPath);
    // Check that the TMs are the same.
    ASSERT_TRUE(BacktrackingTMCpp::tmDiff2(tm1, tm3, std::cout, 2));

    // Infer
    for (Size i = 0; i < 10; i++) {
      if (data[i]) {
        tm1.infer(data[i].get());
        if (i > 0)
          tm1._checkPrediction(nzData);
      }
    }

    // cleanup if successful.
    for (Size i = 0; i < nzData.size(); i++)
      delete[] nzData[i];
    Directory::removeTree("TestOutputDir");
}

} // namespace testing
