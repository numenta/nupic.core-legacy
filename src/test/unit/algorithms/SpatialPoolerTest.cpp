/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * Implementation of unit tests for SpatialPooler
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdio.h>
#include <numeric>

#include "gtest/gtest.h"
#include <htm/algorithms/SpatialPooler.hpp>

#include <htm/utils/StlIo.hpp>
#include <htm/types/Types.hpp>
#include <htm/utils/Log.hpp>
#include <htm/os/Timer.hpp>

namespace testing {

using namespace std;
using namespace htm;

UInt countNonzero(const vector<UInt> &vec) {
  UInt count = 0;

  for (UInt x : vec) {
    if (x > 0) {
      count++;
    }
  }

  return count;
}

bool almost_eq(Real a, Real b) {
  Real diff = a - b;
  return (diff > -1e-5 && diff < 1e-5);
}

bool check_vector_eq(UInt arr[], vector<UInt> vec) {  //TODO replace with ArrayBase, VectorHelpers or teplates
  for (UInt i = 0; i < vec.size(); i++) {
    if (arr[i] != vec[i]) {
      return false;
    }
  }
  return true;
}

bool check_vector_eq(UInt arr[], vector<SynapseIdx> vec) {
  for(UInt i = 0; i < vec.size(); i++) {
    if(arr[i] != (UInt)vec[i]) return false;
  }
  return true;
}

bool check_vector_eq(Real arr[], vector<Real> vec) {
  for (UInt i = 0; i < vec.size(); i++) {
    if (!almost_eq(arr[i], vec[i])) {
      return false;
    }
  }
  return true;
}

bool check_vector_eq(UInt arr1[], UInt arr2[], UInt n) {
  for (UInt i = 0; i < n; i++) {
    if (arr1[i] != arr2[i]) {
      return false;
    }
  }
  return true;
}

bool check_vector_eq(Real arr1[], Real arr2[], UInt n) {
  for (UInt i = 0; i < n; i++) {
    if (!almost_eq(arr1[i], arr2[i])) {
      return false;
    }
  }
  return true;
}

bool check_vector_eq(vector<UInt> vec1, vector<UInt> vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  for (UInt i = 0; i < vec1.size(); i++) {
    if (vec1[i] != vec2[i]) {
      return false;
    }
  }
  return true;
}

bool check_vector_eq(vector<Real> vec1, vector<Real> vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  for (UInt i = 0; i < vec1.size(); i++) {
    if (!almost_eq(vec1[i], vec2[i])) {
      return false;
    }
  }
  return true;
}

void check_spatial_eq(const SpatialPooler& sp1, const SpatialPooler& sp2) {
  UInt numColumns = sp1.getNumColumns();
  UInt numInputs = sp2.getNumInputs();

  ASSERT_TRUE(sp1.getNumColumns() == sp2.getNumColumns());
  ASSERT_TRUE(sp1.getNumInputs() == sp2.getNumInputs());
  ASSERT_TRUE(sp1.getPotentialRadius() == sp2.getPotentialRadius());
  ASSERT_TRUE(sp1.getPotentialPct() == sp2.getPotentialPct());
  ASSERT_TRUE(sp1.getGlobalInhibition() == sp2.getGlobalInhibition());
  ASSERT_TRUE(almost_eq(sp1.getLocalAreaDensity(), sp2.getLocalAreaDensity()));
  ASSERT_TRUE(sp1.getStimulusThreshold() == sp2.getStimulusThreshold());
  ASSERT_TRUE(sp1.getDutyCyclePeriod() == sp2.getDutyCyclePeriod());
  ASSERT_TRUE(almost_eq(sp1.getBoostStrength(), sp2.getBoostStrength()));
  ASSERT_TRUE(sp1.getIterationNum() == sp2.getIterationNum());
  ASSERT_TRUE(sp1.getIterationLearnNum() == sp2.getIterationLearnNum());
  ASSERT_TRUE(sp1.getSpVerbosity() == sp2.getSpVerbosity());
  ASSERT_TRUE(sp1.getWrapAround() == sp2.getWrapAround());
  ASSERT_TRUE(sp1.getUpdatePeriod() == sp2.getUpdatePeriod());
  cout << "check: " << sp1.getSynPermActiveInc() << " "
       << sp2.getSynPermActiveInc() << endl;
  ASSERT_TRUE(almost_eq(sp1.getSynPermActiveInc(), sp2.getSynPermActiveInc()));
  ASSERT_TRUE(
      almost_eq(sp1.getSynPermInactiveDec(), sp2.getSynPermInactiveDec()));
  ASSERT_TRUE(almost_eq(sp1.getSynPermBelowStimulusInc(),
                        sp2.getSynPermBelowStimulusInc()));
  ASSERT_TRUE(almost_eq(sp1.getSynPermConnected(), sp2.getSynPermConnected()));
  ASSERT_TRUE(almost_eq(sp1.getMinPctOverlapDutyCycles(),
                        sp2.getMinPctOverlapDutyCycles()));

  auto boostFactors1 = new Real[numColumns];
  auto boostFactors2 = new Real[numColumns];
  sp1.getBoostFactors(boostFactors1);
  sp2.getBoostFactors(boostFactors2);
  ASSERT_TRUE(check_vector_eq(boostFactors1, boostFactors2, numColumns));
  delete[] boostFactors1;
  delete[] boostFactors2;

  auto overlapDutyCycles1 = new Real[numColumns];
  auto overlapDutyCycles2 = new Real[numColumns];
  sp1.getOverlapDutyCycles(overlapDutyCycles1);
  sp2.getOverlapDutyCycles(overlapDutyCycles2);
  ASSERT_TRUE(
      check_vector_eq(overlapDutyCycles1, overlapDutyCycles2, numColumns));
  delete[] overlapDutyCycles1;
  delete[] overlapDutyCycles2;

  auto activeDutyCycles1 = new Real[numColumns];
  auto activeDutyCycles2 = new Real[numColumns];
  sp1.getActiveDutyCycles(activeDutyCycles1);
  sp2.getActiveDutyCycles(activeDutyCycles2);
  ASSERT_TRUE(
      check_vector_eq(activeDutyCycles1, activeDutyCycles2, numColumns));
  delete[] activeDutyCycles1;
  delete[] activeDutyCycles2;

  auto minOverlapDutyCycles1 = new Real[numColumns];
  auto minOverlapDutyCycles2 = new Real[numColumns];
  sp1.getMinOverlapDutyCycles(minOverlapDutyCycles1);
  sp2.getMinOverlapDutyCycles(minOverlapDutyCycles2);
  ASSERT_TRUE(check_vector_eq(minOverlapDutyCycles1, minOverlapDutyCycles2,
                              numColumns));
  delete[] minOverlapDutyCycles1;
  delete[] minOverlapDutyCycles2;

  for (UInt i = 0; i < numColumns; i++) {
    auto potential1 = new UInt[numInputs];
    auto potential2 = new UInt[numInputs];
    sp1.getPotential(i, potential1);
    sp2.getPotential(i, potential2);
    ASSERT_TRUE(check_vector_eq(potential1, potential2, numInputs));
    delete[] potential1;
    delete[] potential2;
  }

  for (UInt i = 0; i < numColumns; i++) {
    auto perm1 = new Real[numInputs];
    auto perm2 = new Real[numInputs];
    sp1.getPermanence(i, perm1);
    sp2.getPermanence(i, perm2);
    ASSERT_TRUE(check_vector_eq(perm1, perm2, numInputs));
    delete[] perm1;
    delete[] perm2;
  }

  for (UInt i = 0; i < numColumns; i++) {
    auto con1 = new UInt[numInputs];
    auto con2 = new UInt[numInputs];
    sp1.getConnectedSynapses(i, con1);
    sp2.getConnectedSynapses(i, con2);
    ASSERT_TRUE(check_vector_eq(con1, con2, numInputs));
    delete[] con1;
    delete[] con2;
  }

  auto conCounts1 = new UInt[numColumns];
  auto conCounts2 = new UInt[numColumns];
  sp1.getConnectedCounts(conCounts1);
  sp2.getConnectedCounts(conCounts2);
  ASSERT_TRUE(check_vector_eq(conCounts1, conCounts2, numColumns));
  delete[] conCounts1;
  delete[] conCounts2;
}

void setup(SpatialPooler &sp, vector<UInt> inputDim, vector<UInt> columnDim, Real sparsity = 0.5f) {
  //we are interested in the sparsity, should make it artificially high.
  //As we added SP check that sparsity*numColumns > 0, which is correct requirement.
  //But many tests have very small (artificial) number of columns (for convenient results),
  //therefore the check is failing -> we must set high sparsity at initialization. 
  EXPECT_NO_THROW(sp.initialize(inputDim, columnDim, 16u, 0.5f, true, sparsity));
}
void setup(SpatialPooler& sp, UInt numIn, UInt numCols, Real sparsity = 0.5f) {
  setup(sp, vector<UInt>{numIn}, vector<UInt>{numCols}, sparsity); 
}

TEST(SpatialPoolerTest, testUpdateInhibitionRadius) {
  SpatialPooler sp;
  vector<UInt> colDim, inputDim;
  colDim.push_back(57);
  colDim.push_back(31);
  colDim.push_back(2);
  inputDim.push_back(1);
  inputDim.push_back(1);
  inputDim.push_back(1);

  EXPECT_NO_THROW(sp.initialize(inputDim, colDim));
  sp.setGlobalInhibition(true);
  ASSERT_EQ(sp.getInhibitionRadius(), 57u);

  //test 2 - local inhibition radius
  colDim.clear();
  inputDim.clear();
  // avgColumnsPerInput = 4
  // avgConnectedSpanForColumn = 3
  UInt numInputs = 3;
  UInt numCols = 12;
  setup(sp, numInputs, numCols);
  sp.setGlobalInhibition(false);
  sp.setInhibitionRadius(10); //must be < numColumns, otherwise this resorts to global inh

  for (UInt i = 0; i < numCols; i++) {
    sp.setPotential(i, vector<UInt>(numInputs, 1).data());
    Real permArr[] = {1, 1, 1};
    sp.setPermanence(i, permArr);
  }
  UInt trueInhibitionRadius = 6;
  // ((3 * 4) - 1)/2 => round up
  sp.updateInhibitionRadius_();
  ASSERT_EQ(trueInhibitionRadius, sp.getInhibitionRadius());

  //test 3
  colDim.clear();
  inputDim.clear();
  // avgColumnsPerInput = 1.2
  // avgConnectedSpanForColumn = 0.5
  numInputs = 5;
  numCols = 6;
  setup(sp, numInputs, numCols);
  sp.setGlobalInhibition(false);

  for (UInt i = 0; i < numCols; i++) {
    sp.setPotential(i, vector<UInt>(numInputs, 1).data());
    Real permArr[] = {1, 0, 0, 0, 0};
    if (i % 2 == 0) {
      permArr[0] = 0;
    }
    sp.setPermanence(i, permArr);
  }
  trueInhibitionRadius = 1;
  sp.updateInhibitionRadius_();
  ASSERT_EQ(trueInhibitionRadius, sp.getInhibitionRadius());


  //test 4
  colDim.clear();
  inputDim.clear();
  // avgColumnsPerInput = 2.4
  // avgConnectedSpanForColumn = 2
  numInputs = 5;
  numCols = 12;
  setup(sp, numInputs, numCols);
  sp.setGlobalInhibition(false);

  for (UInt i = 0; i < numCols; i++) {
    sp.setPotential(i, vector<UInt>(numInputs, 1).data());
    Real permArr[] = {1, 1, 0, 0, 0};
    sp.setPermanence(i, permArr);
  }
  trueInhibitionRadius = 2;
  // ((2.4 * 2) - 1)/2 => round up
  sp.updateInhibitionRadius_();
  ASSERT_EQ(trueInhibitionRadius, sp.getInhibitionRadius());
}

TEST(SpatialPoolerTest, testUpdateMinDutyCycles) {
  SpatialPooler sp;
  UInt numColumns = 10;
  UInt numInputs = 5;
  setup(sp, numInputs, numColumns);
  sp.setMinPctOverlapDutyCycles(0.01f);

  Real initOverlapDuty[10] = {0.0100f, 0.001f, 0.020f, 0.3000f, 0.012f,
                              0.0512f, 0.054f, 0.221f, 0.0873f, 0.309f};

  Real initActiveDuty[10] = {0.0100f, 0.045f, 0.812f, 0.091f, 0.001f,
                             0.0003f, 0.433f, 0.136f, 0.211f, 0.129f};

  sp.setOverlapDutyCycles(initOverlapDuty);
  sp.setActiveDutyCycles(initActiveDuty);
  sp.setGlobalInhibition(true);
  sp.setInhibitionRadius(2);
  sp.updateMinDutyCycles_();
  Real resultMinOverlap[10];
  sp.getMinOverlapDutyCycles(resultMinOverlap);

  sp.updateMinDutyCyclesGlobal_();
  Real resultMinOverlapGlobal[10];
  sp.getMinOverlapDutyCycles(resultMinOverlapGlobal);

  sp.updateMinDutyCyclesLocal_();
  Real resultMinOverlapLocal[10];
  sp.getMinOverlapDutyCycles(resultMinOverlapLocal);

  ASSERT_TRUE(
      check_vector_eq(resultMinOverlap, resultMinOverlapGlobal, numColumns));

  sp.setGlobalInhibition(false);
  sp.updateMinDutyCycles_();
  sp.getMinOverlapDutyCycles(resultMinOverlap);

  ASSERT_TRUE(
      !check_vector_eq(resultMinOverlap, resultMinOverlapGlobal, numColumns));
}


TEST(SpatialPoolerTest, testUpdateMinDutyCyclesGlobal) {
  SpatialPooler sp;
  UInt numColumns = 5;
  UInt numInputs = 5;
  setup(sp, numInputs, numColumns);
  Real minPctOverlap;

  minPctOverlap = 0.01f;

  sp.setMinPctOverlapDutyCycles(minPctOverlap);

  Real overlapArr1[] = {0.06f, 1.00f, 3.00f, 6.00f, 0.50f};
  Real activeArr1[]  = {0.60f, 0.07f, 0.50f, 0.40f, 0.30f};

  sp.setOverlapDutyCycles(overlapArr1);
  sp.setActiveDutyCycles(activeArr1);

  Real trueMinOverlap1 = 0.01f * 6;

  sp.updateMinDutyCyclesGlobal_();
  Real resultOverlap1[5];
  sp.getMinOverlapDutyCycles(resultOverlap1);
  for (UInt i = 0; i < numColumns; i++) {
    ASSERT_EQ(resultOverlap1[i], trueMinOverlap1);
  }

  minPctOverlap = 0.015f;

  sp.setMinPctOverlapDutyCycles(minPctOverlap);

  Real overlapArr2[] = {0.86f, 2.40f, 0.03f, 1.60f, 1.50f};
  Real activeArr2[]  = {0.16f, 0.007f,0.15f, 0.54f, 0.13f};

  sp.setOverlapDutyCycles(overlapArr2);
  sp.setActiveDutyCycles(activeArr2);

  Real trueMinOverlap2 = 0.015f * 2.4f;

  sp.updateMinDutyCyclesGlobal_();
  Real resultOverlap2[5];
  sp.getMinOverlapDutyCycles(resultOverlap2);
  for (UInt i = 0; i < numColumns; i++) {
    ASSERT_TRUE(almost_eq(resultOverlap2[i], trueMinOverlap2));
  }

  minPctOverlap = 0.015f;

  sp.setMinPctOverlapDutyCycles(minPctOverlap);

  Real overlapArr3[] = {0, 0, 0, 0, 0};
  Real activeArr3[]  = {0, 0, 0, 0, 0};

  sp.setOverlapDutyCycles(overlapArr3);
  sp.setActiveDutyCycles(activeArr3);

  Real trueMinOverlap3 = 0;

  sp.updateMinDutyCyclesGlobal_();
  Real resultOverlap3[5];
  sp.getMinOverlapDutyCycles(resultOverlap3);
  for (UInt i = 0; i < numColumns; i++) {
    ASSERT_TRUE(almost_eq(resultOverlap3[i], trueMinOverlap3));
  }
}


TEST(SpatialPoolerTest, testUpdateMinDutyCyclesLocal) {
  // wrapAround=false
  {
    UInt numColumns = 8;
    SpatialPooler sp(
        /*inputDimensions*/ {5},
        /*columnDimensions*/ {numColumns},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5f,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ 0.2f,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008f,
        /*synPermActiveInc*/ 0.05f,
        /*synPermConnected*/ 0.1f,
        /*minPctOverlapDutyCycles*/ 0.001f,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 0.0f,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ false);

    sp.setInhibitionRadius(1);

    Real activeDutyArr[] = {0.9f, 0.3f, 0.5f, 0.7f, 0.1f, 0.01f, 0.08f, 0.12f};
    sp.setActiveDutyCycles(activeDutyArr);

    Real overlapDutyArr[] = {0.7f, 0.1f, 0.5f, 0.01f, 0.78f, 0.55f, 0.1f, 0.001f};
    sp.setOverlapDutyCycles(overlapDutyArr);

    sp.setMinPctOverlapDutyCycles(0.2f);

    sp.updateMinDutyCyclesLocal_();

    Real trueOverlapArr[] = {0.2f * 0.70f, 0.2f * 0.70f, 0.2f * 0.50f, 0.2f * 0.78f,
                             0.2f * 0.78f, 0.2f * 0.78f, 0.2f * 0.55f, 0.2f * 0.10f};
    Real resultMinOverlapArr[8];
    sp.getMinOverlapDutyCycles(resultMinOverlapArr);
    ASSERT_TRUE(
        check_vector_eq(resultMinOverlapArr, trueOverlapArr, numColumns));
  }

  // wrapAround=true
  {
    UInt numColumns = 8;
    SpatialPooler sp(
        /*inputDimensions*/ {5},
        /*columnDimensions*/ {numColumns},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5f,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ 0.2f,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008f,
        /*synPermActiveInc*/ 0.05f,
        /*synPermConnected*/ 0.1f,
        /*minPctOverlapDutyCycles*/ 0.001f,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 10.0f,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ true);

    sp.setInhibitionRadius(1);

    Real activeDutyArr[] = {0.9f, 0.3f, 0.5f, 0.7f, 0.1f, 0.01f, 0.08f, 0.12f};
    sp.setActiveDutyCycles(activeDutyArr);

    Real overlapDutyArr[] = {0.7f, 0.1f, 0.5f, 0.01f, 0.78f, 0.55f, 0.1f, 0.001f};
    sp.setOverlapDutyCycles(overlapDutyArr);

    sp.setMinPctOverlapDutyCycles(0.2f);

    sp.updateMinDutyCyclesLocal_();

    Real trueOverlapArr[] = {0.2f * 0.70f, 0.2f * 0.70f, 0.2f * 0.50f, 0.2f * 0.78f,
                             0.2f * 0.78f, 0.2f * 0.78f, 0.2f * 0.55f, 0.2f * 0.70f};
    Real resultMinOverlapArr[8];
    sp.getMinOverlapDutyCycles(resultMinOverlapArr);
    ASSERT_TRUE(
      check_vector_eq(resultMinOverlapArr, trueOverlapArr, numColumns));
  }
}


TEST(SpatialPoolerTest, testUpdateDutyCycles) {
  SpatialPooler sp;
  UInt numInputs = 5;
  UInt numColumns = 5;
  setup(sp, numInputs, numColumns);
  vector<SynapseIdx> overlaps;
  SDR active({numColumns});

  Real initOverlapArr1[] = {1, 1, 1, 1, 1};
  sp.setOverlapDutyCycles(initOverlapArr1);
  UInt overlapNewVal1[] = {1, 5, 7, 0, 0};
  overlaps.assign(overlapNewVal1, overlapNewVal1 + numColumns);
  active.setDense(vector<Byte>({0, 0, 0, 0, 0}));

  sp.setIterationNum(2);
  sp.updateDutyCycles_(overlaps, active);

  Real resultOverlapArr1[5];
  sp.getOverlapDutyCycles(resultOverlapArr1);

  Real trueOverlapArr1[] = {1.0f, 1.0f, 1.0f, 0.5f, 0.5f};
  ASSERT_TRUE(check_vector_eq(resultOverlapArr1, trueOverlapArr1, numColumns));

  sp.setOverlapDutyCycles(initOverlapArr1);
  sp.setIterationNum(2000);
  sp.setUpdatePeriod(1000);
  sp.updateDutyCycles_(overlaps, active);

  Real resultOverlapArr2[5];
  sp.getOverlapDutyCycles(resultOverlapArr2);
  Real trueOverlapArr2[] = {1, 1, 1, 0.999f, 0.999f};

  ASSERT_TRUE(check_vector_eq(resultOverlapArr2, trueOverlapArr2, numColumns));
}


TEST(SpatialPoolerTest, testAvgColumnsPerInput) {
  SpatialPooler sp;
  vector<UInt> inputDim, colDim;

  UInt colDim1[4] = {2, 2, 2, 2};
  UInt inputDim1[4] = {4, 4, 4, 4};
  Real trueAvgColumnPerInput1 = 0.5f;

  inputDim.assign(inputDim1, inputDim1 + 4);
  colDim.assign(colDim1, colDim1 + 4);
  setup(sp, inputDim, colDim);

  Real result = sp.avgColumnsPerInput_();
  ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput1);

  UInt colDim2[4] = {2, 2, 2, 2};
  UInt inputDim2[4] = {7, 5, 1, 3};
  Real trueAvgColumnPerInput2 = (2.0f / 7 + 2.0f / 5 + 2.0f / 1 + 2.0f / 3) / 4;


  inputDim.assign(inputDim2, inputDim2 + 4);
  colDim.assign(colDim2, colDim2 + 4);
  setup(sp, inputDim, colDim);

  result = sp.avgColumnsPerInput_();
  ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput2);


  UInt colDim3[2] = {3, 3};
  UInt inputDim3[2] = {3, 3};
  Real trueAvgColumnPerInput3 = 1;

  inputDim.assign(inputDim3, inputDim3 + 2);
  colDim.assign(colDim3, colDim3 + 2);
  setup(sp, inputDim, colDim);
  result = sp.avgColumnsPerInput_();
  ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput3);


  UInt colDim4[1] = {25};
  UInt inputDim4[1] = {5};
  Real trueAvgColumnPerInput4 = 5;

  inputDim.assign(inputDim4, inputDim4 + 1);
  colDim.assign(colDim4, colDim4 + 1);
  setup(sp, inputDim, colDim);
  result = sp.avgColumnsPerInput_();
  ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput4);


  UInt colDim5[7] = {3, 5, 6};
  UInt inputDim5[7] = {3, 5, 6};
  Real trueAvgColumnPerInput5 = 1;

  inputDim.assign(inputDim5, inputDim5 + 3);
  colDim.assign(colDim5, colDim5 + 3);
  setup(sp, inputDim, colDim);
  result = sp.avgColumnsPerInput_();
  ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput5);


  UInt colDim6[4] = {2, 4, 6, 8};
  UInt inputDim6[4] = {2, 2, 2, 2};
  //  1  2  3  4
  Real trueAvgColumnPerInput6 = 2.5;

  inputDim.assign(inputDim6, inputDim6 + 4);
  colDim.assign(colDim6, colDim6 + 4);
  setup(sp, inputDim, colDim);
  result = sp.avgColumnsPerInput_();
  ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput6);
}

TEST(SpatialPoolerTest, testAvgConnectedSpanForColumn1D) {

  SpatialPooler sp;
  UInt numColumns = 9;
  UInt numInputs = 8;
  setup(sp, numInputs, numColumns);

  vector<UInt> potential(numInputs, 1);
  Real permArr[9][8] = {{0, 1, 0, 1, 0, 1, 0, 1}, {0, 0, 0, 1, 0, 0, 0, 1},
                        {0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 1, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0},
                        {0, 0, 1, 1, 1, 0, 0, 0}, {0, 0, 1, 0, 1, 0, 0, 0},
                        {1, 1, 1, 1, 1, 1, 1, 1}};

  UInt trueAvgConnectedSpan[9] = {7, 5, 1, 5, 0, 2, 3, 3, 8};

  for (UInt i = 0; i < numColumns; i++) {
    sp.setPotential(i, potential.data());
    sp.setPermanence(i, permArr[i]);
    UInt result = (UInt)floor(sp.avgConnectedSpanForColumnND_(i));
    ASSERT_TRUE(result == trueAvgConnectedSpan[i]);
  }
}


TEST(SpatialPoolerTest, testAvgConnectedSpanForColumn2D) {
  SpatialPooler sp;

  UInt numColumns = 7;
  UInt numInputs = 20;

  vector<UInt> colDim, inputDim;
  vector<UInt> potential1(numInputs, 1);
  Real permArr1[7][20] = {
      {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
      // rowspan = 3, colspan = 3, avg = 3

      {1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      // rowspan = 2 colspan = 4, avg = 3

      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      // row span = 5, colspan = 4, avg = 4.5

      {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
      // rowspan = 5, colspan = 1, avg = 3

      {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      // rowspan = 1, colspan = 4, avg = 2.5

      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
      // rowspan = 2, colspan = 2, avg = 2

      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      // rowspan = 0, colspan = 0, avg = 0
  };
  inputDim.push_back(5);
  inputDim.push_back(4);
  colDim.push_back(10);
  colDim.push_back(1);
  setup(sp, inputDim, colDim);

  UInt trueAvgConnectedSpan1[7] = {3, 3, 4, 3, 2, 2, 0};

  for (UInt i = 0; i < numColumns; i++) {
    sp.setPotential(i, potential1.data());
    sp.setPermanence(i, permArr1[i]);
    UInt result = (UInt)floor(sp.avgConnectedSpanForColumnND_(i));
    ASSERT_EQ(result,  trueAvgConnectedSpan1[i]);
  }

  // 1D tests repeated
  numColumns = 9;
  numInputs = 8;

  colDim.clear();
  inputDim.clear();
  inputDim.push_back(numInputs);
  inputDim.push_back(1);
  colDim.push_back(numColumns);
  colDim.push_back(1);

  setup(sp, inputDim, colDim);

  vector<UInt> potential2(numInputs, 1);
  Real permArr2[9][8] = {{0, 1, 0, 1, 0, 1, 0, 1}, {0, 0, 0, 1, 0, 0, 0, 1},
                         {0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 1, 0, 0, 0, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0},
                         {0, 0, 1, 1, 1, 0, 0, 0}, {0, 0, 1, 0, 1, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1}};

  UInt trueAvgConnectedSpan2[9] = {8, 5, 1, 5, 0, 2, 3, 3, 8};

  for (UInt i = 0; i < numColumns; i++) {
    sp.setPotential(i, potential2.data());
    sp.setPermanence(i, permArr2[i]);
    UInt result = (UInt)floor(sp.avgConnectedSpanForColumnND_(i));
    ASSERT_EQ(result, static_cast<UInt>((trueAvgConnectedSpan2[i] + 1) / 2));
  }
}


TEST(SpatialPoolerTest, testAvgConnectedSpanForColumnND) {
  SpatialPooler sp;
  vector<UInt> inputDim, colDim;
  inputDim.push_back(4);
  inputDim.push_back(4);
  inputDim.push_back(2);
  inputDim.push_back(5);
  colDim.push_back(5);
  colDim.push_back(1);
  colDim.push_back(1);
  colDim.push_back(1);

  setup(sp, inputDim, colDim);

  UInt numInputs = 160;
  UInt numColumns = 5;

  // All potential synapses exist.
  vector<UInt> ones(numInputs, 1);
  sp.setPotential(0, (UInt *)ones.data());
  sp.setPotential(1, (UInt *)ones.data());
  sp.setPotential(2, (UInt *)ones.data());
  sp.setPotential(3, (UInt *)ones.data());
  sp.setPotential(4, (UInt *)ones.data());

  Real permArr0[4][4][2][5];
  Real permArr1[4][4][2][5];
  Real permArr2[4][4][2][5];
  Real permArr3[4][4][2][5];
  Real permArr4[4][4][2][5];

  for (UInt i = 0; i < numInputs; i++) {
    ((Real *)permArr0)[i] = 0;
    ((Real *)permArr1)[i] = 0;
    ((Real *)permArr2)[i] = 0;
    ((Real *)permArr3)[i] = 0;
    ((Real *)permArr4)[i] = 0;
  }

  permArr0[1][0][1][0] = 1;
  permArr0[1][0][1][1] = 1;
  permArr0[3][2][1][0] = 1;
  permArr0[3][0][1][0] = 1;
  permArr0[1][0][1][3] = 1;
  permArr0[2][2][1][0] = 1;

  permArr1[2][0][1][0] = 1;
  permArr1[2][0][0][0] = 1;
  permArr1[3][0][0][0] = 1;
  permArr1[3][0][1][0] = 1;

  permArr2[0][0][1][4] = 1;
  permArr2[0][0][0][3] = 1;
  permArr2[0][0][0][1] = 1;
  permArr2[1][0][0][2] = 1;
  permArr2[0][0][1][1] = 1;
  permArr2[3][3][1][1] = 1;

  permArr3[3][3][1][4] = 1;
  permArr3[0][0][0][0] = 1;

  sp.setPermanence(0, (Real *)permArr0);
  sp.setPermanence(1, (Real *)permArr1);
  sp.setPermanence(2, (Real *)permArr2);
  sp.setPermanence(3, (Real *)permArr3);
  sp.setPermanence(4, (Real *)permArr4);

  Real trueAvgConnectedSpan[5] = {11.0f / 4, 6.0f / 4, 14.0f / 4, 15.0f / 4, 0};

  for (UInt i = 0; i < numColumns; i++) {
    Real result = sp.avgConnectedSpanForColumnND_(i);
    ASSERT_EQ(result, trueAvgConnectedSpan[i]);
  }
}


TEST(SpatialPoolerTest, testAdaptSynapses) {
  SpatialPooler sp;
  UInt numColumns = 4;
  UInt numInputs = 8;
  setup(sp, numInputs, numColumns);

  SDR activeColumns({numColumns});
  vector<UInt> inputVector;

  UInt potentialArr1[4][8] = {{1, 1, 1, 1, 0, 0, 0, 0},
                              {1, 0, 0, 0, 1, 1, 0, 1},
                              {0, 0, 1, 0, 0, 0, 1, 0},
                              {1, 0, 0, 0, 0, 0, 1, 0}};

  Real permanencesArr1[4][8] = {
      {0.200f, 0.120f, 0.090f, 0.060f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.150f, 0.000f, 0.000f, 0.000f, 0.180f, 0.120f, 0.000f, 0.450f},
      {0.000f, 0.000f, 0.014f, 0.000f, 0.000f, 0.000f, 0.110f, 0.000f},
      {0.070f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.178f, 0.000f}};

  Real truePermanences1[4][8] = {
      {0.300f, 0.110f, 0.080f, 0.160f, 0.000f, 0.000f, 0.000f, 0.000f},
      // Inc     Dec    Dec     Inc      -       -       -       -
      {0.250f, 0.000f, 0.000f, 0.000f, 0.280f, 0.110f, 0.000f, 0.440f},
      // Inc      -      -       -       Inc     Dec     -       Dec
      {0.000f, 0.000f, 0.004f, 0.000f, 0.000f, 0.000f, 0.210f, 0.000f},
      //  -       -      -       -       -       -       Inc     -
      {0.070f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.178f, 0.000f}};
      //  -       -      -       -       -       -        -      -

  SDR input1({8});
  input1.setDense(SDR_dense_t{1, 0, 0, 1, 1, 0, 1, 0});
  activeColumns.setSparse(SDR_sparse_t({ 0, 1, 2 }));

  for (UInt column = 0; column < numColumns; column++) {
    sp.setPotential(column, potentialArr1[column]);
    sp.setPermanence(column, permanencesArr1[column]);
  }

  sp.adaptSynapses_(input1, activeColumns);
  for (UInt column = 0; column < numColumns; column++) {
    auto permArr = new Real[numInputs];
    sp.getPermanence(column, permArr);
    ASSERT_TRUE(check_vector_eq(truePermanences1[column], permArr, numInputs));
    delete[] permArr;
  }

  UInt potentialArr2[4][8] = {{1, 1, 1, 0, 0, 0, 0, 0},
                              {0, 1, 1, 1, 0, 0, 0, 0},
                              {0, 0, 1, 1, 1, 0, 0, 0},
                              {1, 0, 0, 0, 0, 0, 1, 0}};

  Real permanencesArr2[4][8] = {
      {0.200f, 0.120f, 0.090f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.000f, 0.017f, 0.232f, 0.400f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.000f, 0.000f, 0.014f, 0.051f, 0.730f, 0.000f, 0.000f, 0.000f},
      {0.170f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.380f, 0.000f}};

  Real truePermanences2[4][8] = {
      {0.300f, 0.110f, 0.080f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f},
      // Inc     Dec     Dec      -       -       -       -       -
      {0.000f, 0.007f, 0.222f, 0.500f, 0.000f, 0.000f, 0.000f, 0.000f},
      //  -     -        Dec     Inc      -       -       -       -
      {0.000f, 0.000f, 0.004f, 0.151f, 0.830f, 0.000f, 0.000f, 0.000f},
      //  -       -       -      Inc     Inc      -       -       -
      {0.170f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.380f, 0.000f}};
      //  -       -       -       -       -       -       -       -

  SDR input2({8});
  input2.setDense(SDR_dense_t{1, 0, 0, 1, 1, 0, 1, 0});
  UInt activeColumnsArr2[3] = {0, 1, 2};

  for (UInt column = 0; column < numColumns; column++) {
    sp.setPotential(column, potentialArr2[column]);
    sp.setPermanence(column, permanencesArr2[column]);
  }

  activeColumns.setSparse(activeColumnsArr2, 3);

  sp.adaptSynapses_(input2, activeColumns);
  for (UInt column = 0; column < numColumns; column++) {
    auto permArr = new Real[numInputs];
    sp.getPermanence(column, permArr);
    ASSERT_TRUE(check_vector_eq(truePermanences2[column], permArr, numInputs));
    delete[] permArr;
  }
}


TEST(SpatialPoolerTest, testBumpUpWeakColumns) {
  SpatialPooler sp;
  UInt numInputs = 8;
  UInt numColumns = 5;
  setup(sp, numInputs, numColumns);
  sp.setSynPermBelowStimulusInc(0.01f);
  Real overlapDutyCyclesArr[] = {0.000f, 0.009f, 0.100f, 0.001f, 0.002f};
  sp.setOverlapDutyCycles(overlapDutyCyclesArr);
  Real minOverlapDutyCyclesArr[] = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
  sp.setMinOverlapDutyCycles(minOverlapDutyCyclesArr);

  UInt potentialArr[5][8] = {{1, 1, 1, 1, 0, 0, 0, 0},
                             {1, 0, 0, 0, 1, 1, 0, 1},
                             {0, 0, 1, 0, 1, 1, 1, 0},
                             {1, 1, 1, 0, 0, 0, 1, 0},
                             {1, 1, 1, 1, 1, 1, 1, 1}};

  Real permArr[5][8] = {
      {0.200f, 0.120f, 0.090f, 0.040f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.150f, 0.000f, 0.000f, 0.000f, 0.180f, 0.120f, 0.000f, 0.450f},
      {0.000f, 0.000f, 0.074f, 0.000f, 0.062f, 0.054f, 0.110f, 0.000f},
      {0.051f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.178f, 0.000f},
      {0.100f, 0.738f, 0.085f, 0.002f, 0.052f, 0.008f, 0.208f, 0.034f}};

  Real truePermArr[5][8] = {
      {0.210f, 0.130f, 0.100f, 0.050f, 0.000f, 0.000f, 0.000f, 0.000f},
      // Inc     Inc     Inc     Inc      -       -       -       -
      {0.160f, 0.000f, 0.000f, 0.000f, 0.190f, 0.130f, 0.000f, 0.460f},
      // Inc      -       -       -      Inc     Inc      -      Inc
      {0.000f, 0.000f, 0.074f, 0.000f, 0.062f, 0.054f, 0.110f, 0.000f}, // unchanged
      //  -       -       -       -       -       -       -       -
      {0.061f, 0.010f, 0.010f, 0.000f, 0.000f, 0.000f, 0.188f, 0.000f},
      // Inc     Inc     Inc      -       -       -      Inc      -
      {0.110f, 0.748f, 0.095f, 0.012f, 0.062f, 0.018f, 0.218f, 0.044f}};
      //  -       -       -       -       -       -       -       -

  for (UInt i = 0; i < numColumns; i++) {
    sp.setPotential(i, potentialArr[i]);
    sp.setPermanence(i, permArr[i]);
  }

  sp.bumpUpWeakColumns_();

  for (UInt i = 0; i < numColumns; i++) {
    Real perm[8];
    sp.getPermanence(i, perm);
    for(UInt z = 0; z < numInputs; z++)
      ASSERT_FLOAT_EQ( truePermArr[i][z], perm[z] );
  }
}


TEST(SpatialPoolerTest, testUpdateDutyCyclesHelper) {
  SpatialPooler sp;
  vector<Real> dutyCycles;
  SDR newValues({5});
  UInt period;

  Real dutyCyclesArr1[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  Real newValues1[] = {0, 0, 0, 0, 0};
  period = 1000;
  Real trueDutyCycles1[] = {0.999f, 0.999f, 0.999f, 0.999f, 0.999f};
  dutyCycles.assign(dutyCyclesArr1, dutyCyclesArr1 + 5);
  newValues.setDense(newValues1);
  sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
  ASSERT_TRUE(check_vector_eq(trueDutyCycles1, dutyCycles));

  Real dutyCyclesArr2[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  Real newValues2[] = {1, 1, 1, 1, 1};
  period = 1000;
  Real trueDutyCycles2[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  dutyCycles.assign(dutyCyclesArr2, dutyCyclesArr2 + 5);
  newValues.setDense(newValues2);
  sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
  ASSERT_TRUE(check_vector_eq(trueDutyCycles2, dutyCycles));

  Real dutyCyclesArr4[] = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f};
  Real newValues4[] = {0, 0, 0, 0, 0};
  period = 2;
  Real trueDutyCycles4[] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
  dutyCycles.assign(dutyCyclesArr4, dutyCyclesArr4 + 5);
  newValues.setDense(newValues4);
  sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
  ASSERT_TRUE(check_vector_eq(trueDutyCycles4, dutyCycles));
}


TEST(SpatialPoolerTest, testUpdateBoostFactors) {
  SpatialPooler sp;
  setup(sp, 5, 6);

  Real32 initActiveDutyCycles1[] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
  Real32 initBoostFactors1[] = {0, 0, 0, 0, 0, 0};
  vector<Real32> trueBoostFactors1 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  vector<Real32> resultBoostFactors1(6, 0);
  sp.setGlobalInhibition(false);
  sp.setBoostStrength(10);
  sp.setBoostFactors(initBoostFactors1);
  sp.setActiveDutyCycles(initActiveDutyCycles1);
  sp.updateBoostFactors_();
  sp.getBoostFactors(resultBoostFactors1.data());
  ASSERT_TRUE(check_vector_eq(trueBoostFactors1, resultBoostFactors1));

  Real32 initActiveDutyCycles2[] = {0.1f, 0.3f, 0.02f, 0.04f, 0.7f, 0.12f};
  Real32 initBoostFactors2[] = {0, 0, 0, 0, 0, 0};
  vector<Real32> trueBoostFactors2 = {3.10599f, 0.42035f,    6.91251f,
                                      5.65949f, 0.00769898f, 2.54297f};
  vector<Real32> resultBoostFactors2(6, 0);
  sp.setGlobalInhibition(false);
  sp.setBoostStrength(10);
  sp.setBoostFactors(initBoostFactors2);
  sp.setActiveDutyCycles(initActiveDutyCycles2);
  sp.updateBoostFactors_();
  sp.getBoostFactors(resultBoostFactors2.data());

  ASSERT_TRUE(check_vector_eq(trueBoostFactors2, resultBoostFactors2));

  Real32 initActiveDutyCycles3[] = {0.1f, 0.3f, 0.02f, 0.04f, 0.7f, 0.12f};
  Real initBoostFactors3[] = {0, 0, 0, 0, 0, 0};
  vector<Real32> trueBoostFactors3 = {1.25441f, 0.840857f, 1.47207f,
                                      1.41435f, 0.377822f, 1.20523f};
  vector<Real32> resultBoostFactors3(6, 0);
  sp.setWrapAround(true);
  sp.setGlobalInhibition(false);
  sp.setBoostStrength(2.0);
  sp.setInhibitionRadius(5);
  sp.setBoostFactors(initBoostFactors3);
  sp.setActiveDutyCycles(initActiveDutyCycles3);
  sp.updateBoostFactors_();
  sp.getBoostFactors(resultBoostFactors3.data());

  ASSERT_TRUE(check_vector_eq(trueBoostFactors3, resultBoostFactors3));

  Real32 initActiveDutyCycles4[] = {0.1f, 0.3f, 0.02f, 0.04f, 0.7f, 0.12f};
  Real32 initBoostFactors4[] = {0, 0, 0, 0, 0, 0};
  vector<Real32> trueBoostFactors4 = {1.94773f, 0.263597f,   4.33476f,
                                      3.549f,   0.00482795f, 1.59467f};
  vector<Real32> resultBoostFactors4(6, 0);
  sp.setGlobalInhibition(true);
  sp.setBoostStrength(10);
  sp.setInhibitionRadius(3);
  sp.setBoostFactors(initBoostFactors4);
  sp.setActiveDutyCycles(initActiveDutyCycles4);
  sp.updateBoostFactors_();
  sp.getBoostFactors(resultBoostFactors4.data());

  ASSERT_TRUE(check_vector_eq(trueBoostFactors3, resultBoostFactors3));
}


TEST(SpatialPoolerTest, testUpdateBookeepingVars) {
  SpatialPooler sp;
  sp.setIterationNum(5);
  sp.setIterationLearnNum(3);
  sp.updateBookeepingVars_(true);
  ASSERT_TRUE(6 == sp.getIterationNum());
  ASSERT_TRUE(4 == sp.getIterationLearnNum());

  sp.updateBookeepingVars_(false);
  ASSERT_TRUE(7 == sp.getIterationNum());
  ASSERT_TRUE(4 == sp.getIterationLearnNum());
}


TEST(SpatialPoolerTest, testCalculateOverlap) {
  SpatialPooler sp;
  UInt numInputs = 10;
  UInt numColumns = 5;
  UInt numTrials = 5;
  setup(sp, numInputs, numColumns);
  sp.setStimulusThreshold(0);

  Real permArr[5][10] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};

  UInt inputs[5][10] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                        {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

  UInt trueOverlaps[5][5] = {{0, 0, 0, 0, 0},
                             {10, 8, 6, 4, 2},
                             {5, 4, 3, 2, 1},
                             {5, 3, 1, 0, 0},
                             {1, 1, 1, 1, 1}};

  for (UInt i = 0; i < numColumns; i++) {
    vector<UInt> potential;
    for(Size j=0; j < numInputs; j++)
      potential.push_back((UInt)permArr[i][j]);
    sp.setPotential(i, potential.data());
    sp.setPermanence(i, permArr[i]);
  }

  for (UInt i = 0; i < numTrials; i++) {
    vector<SynapseIdx> overlaps;
    SDR input({numInputs});
    input.setDense(SDR_dense_t(inputs[i], inputs[i] + numInputs));
    sp.calculateOverlap_(input, overlaps);
    ASSERT_TRUE(check_vector_eq(trueOverlaps[i], overlaps));
  }
}


TEST(SpatialPoolerTest, testInhibitColumns) {
  SpatialPooler sp;
  setup(sp, 10, 10);

  vector<Real> overlapsReal;
  vector<Real> overlaps;
  vector<UInt> activeColumns;
  vector<UInt> activeColumnsGlobal;
  vector<UInt> activeColumnsLocal;
  Real density;
  UInt inhibitionRadius;
  UInt numColumns;

  density = 0.3f;
  numColumns = 10;
  Real overlapsArray[10] = {10, 21, 34, 4, 18, 3, 12, 5, 7, 1};

  overlapsReal.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumnsGlobal_(overlapsReal, density, activeColumnsGlobal);
  overlapsReal.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumnsLocal_(overlapsReal, density, activeColumnsLocal);

  sp.setInhibitionRadius(5);
  sp.setGlobalInhibition(true);
  sp.setLocalAreaDensity(density);

  overlaps.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumns_(overlaps, activeColumns);

  ASSERT_TRUE(check_vector_eq(activeColumns, activeColumnsGlobal));
  ASSERT_TRUE(!check_vector_eq(activeColumns, activeColumnsLocal));

  sp.setGlobalInhibition(false);
  sp.setInhibitionRadius(numColumns + 1);

  overlaps.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumns_(overlaps, activeColumns);

  ASSERT_TRUE(check_vector_eq(activeColumns, activeColumnsGlobal));
  ASSERT_TRUE(!check_vector_eq(activeColumns, activeColumnsLocal));

  inhibitionRadius = 2;
  density = 2.0f / 5;

  sp.setInhibitionRadius(inhibitionRadius);

  overlapsReal.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumnsGlobal_(overlapsReal, density, activeColumnsGlobal);
  overlapsReal.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumnsLocal_(overlapsReal, density, activeColumnsLocal);

  overlaps.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumns_(overlaps, activeColumns);

  ASSERT_TRUE(!check_vector_eq(activeColumns, activeColumnsGlobal));
  ASSERT_TRUE(check_vector_eq(activeColumns, activeColumnsLocal));
}


TEST(SpatialPoolerTest, testInhibitColumnsGlobal) {
  SpatialPooler sp;
  UInt numInputs = 10;
  UInt numColumns = 10;
  setup(sp, numInputs, numColumns);
  vector<Real> overlaps;
  vector<UInt> activeColumns;
  vector<UInt> trueActive;
  vector<UInt> active;
  Real density;

  density = 0.3f;
  Real overlapsArray[10] = {1, 2, 1, 4, 8, 3, 12, 5, 4, 1};
  overlaps.assign(&overlapsArray[0], &overlapsArray[numColumns]);
  sp.inhibitColumnsGlobal_(overlaps, density, activeColumns);
  UInt trueActiveArray1[3] = {4, 6, 7};

  trueActive.assign(numColumns, 0);
  active.assign(numColumns, 0);

  for (auto &elem : trueActiveArray1) {
    trueActive[elem] = 1;
  }

  for (auto &activeColumn : activeColumns) {
    active[activeColumn] = 1;
  }

  ASSERT_TRUE(check_vector_eq(trueActive, active));

  density = 0.5f;
  Real overlapsArray2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  overlaps.assign(&overlapsArray2[0], &overlapsArray2[numColumns]);
  sp.inhibitColumnsGlobal_(overlaps, density, activeColumns);
  UInt trueActiveArray2[5] = {5, 6, 7, 8, 9};

  for (auto &elem : trueActiveArray2) {
    trueActive[elem] = 1;
  }

  for (auto &activeColumn : activeColumns) {
    active[activeColumn] = 1;
  }

  ASSERT_TRUE(check_vector_eq(trueActive, active));
}


TEST(SpatialPoolerTest, testValidateGlobalInhibitionParameters) {
  // With 10 columns the minimum sparsity for global inhibition is 10%
  // Setting sparsity to 2% should throw an exception
  SpatialPooler sp;
  setup(sp, 10, 10);
  sp.setGlobalInhibition(true);
  SDR input( {sp.getNumInputs()} );
  SDR out1( {sp.getNumColumns()} );
  //throws
  EXPECT_ANY_THROW(sp.setLocalAreaDensity(0.02f));
//  EXPECT_THROW(sp.compute(input, false, out1), htm::LoggingException);
  //good parameter
  EXPECT_NO_THROW(sp.setLocalAreaDensity(0.1f));
  EXPECT_NO_THROW(sp.compute(input, false, out1));
}


TEST(SpatialPoolerTest, testFewColumnsGlobalInhibitionCrash) {
  /** this test exposes bug where too small (few columns) SP crashes with global inhibition  */
  SpatialPooler sp{std::vector<UInt>{1000} /* input*/, std::vector<UInt>{200}/* SP output cols XXX sensitive*/ };
  sp.setBoostStrength(0.0);
  sp.setPotentialRadius(20);
  sp.setPotentialPct(0.5);
  sp.setGlobalInhibition(true);
  sp.setLocalAreaDensity(0.02f);

  SDR input( {sp.getNumInputs()} );
  SDR out1(  {sp.getNumColumns()} );

  EXPECT_NO_THROW(sp.compute(input, false, out1));
}


TEST(SpatialPoolerTest, testInhibitColumnsLocal) {
  // wrapAround = false
  {
    SpatialPooler sp(
        /*inputDimensions*/ {10},
        /*columnDimensions*/ {10},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5f,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ 0.1f,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008f,
        /*synPermActiveInc*/ 0.05f,
        /*synPermConnected*/ 0.1f,
        /*minPctOverlapDutyCycles*/ 0.001f,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 10.0f,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ false);

    Real density;
    UInt inhibitionRadius;

    vector<Real> overlaps;
    vector<UInt> active;

    Real overlapsArray1[10] = {1, 2, 7, 0, 3, 4, 16, 1, 1.5f, 1.7f};
    //                         L  W  W  L  L  W  W   L   L     W

    inhibitionRadius = 2;
    density = 0.5;
    overlaps.assign(&overlapsArray1[0], &overlapsArray1[10]);
    UInt trueActive[5] = {1, 2, 5, 6, 9};
    sp.setInhibitionRadius(inhibitionRadius);
    sp.inhibitColumnsLocal_(overlaps, density, active);
    ASSERT_EQ(5ul, active.size());
    ASSERT_TRUE(check_vector_eq(trueActive, active));

    Real overlapsArray2[10] = {1, 2, 7, 0, 3, 4, 16, 1, 1.5f, 1.7f};
    //                         L  W  W  L  L  W   W  L   L     W
    overlaps.assign(&overlapsArray2[0], &overlapsArray2[10]);
    UInt trueActive2[6] = {1, 2, 4, 5, 6, 9};
    inhibitionRadius = 3;
    density = 0.5;
    sp.setInhibitionRadius(inhibitionRadius);
    sp.inhibitColumnsLocal_(overlaps, density, active);
    ASSERT_TRUE(active.size() == 6);
    ASSERT_TRUE(check_vector_eq(trueActive2, active));

    // Test arbitration

    Real overlapsArray3[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    //                         W  L  W  L  W  L  W  L  L  L
    overlaps.assign(&overlapsArray3[0], &overlapsArray3[10]);
    UInt trueActive3[4] = {0, 2, 4, 6};
    inhibitionRadius = 3;
    density = 0.25;
    sp.setInhibitionRadius(inhibitionRadius);
    sp.inhibitColumnsLocal_(overlaps, density, active);

    ASSERT_TRUE(active.size() == 4);
    ASSERT_TRUE(check_vector_eq(trueActive3, active));
  }

  // wrapAround = true
  {
    SpatialPooler sp(
        /*inputDimensions*/ {10},
        /*columnDimensions*/ {10},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5f,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ 0.1f,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008f,
        /*synPermActiveInc*/ 0.05f,
        /*synPermConnected*/ 0.1f,
        /*minPctOverlapDutyCycles*/ 0.001f,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 10.0f,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ true);

    Real density;
    UInt inhibitionRadius;

    vector<Real> overlaps;
    vector<UInt> active;

    Real overlapsArray1[10] = {1, 2, 7, 0, 3, 4, 16, 1, 1.5f, 1.7f};
    //                         L  W  W  L  L  W  W   L   W     W

    inhibitionRadius = 2;
    density = 0.5f;
    overlaps.assign(&overlapsArray1[0], &overlapsArray1[10]);
    UInt trueActive[6] = {1, 2, 5, 6, 8, 9};
    sp.setInhibitionRadius(inhibitionRadius);
    sp.inhibitColumnsLocal_(overlaps, density, active);
    ASSERT_EQ(6ul, active.size());
    ASSERT_TRUE(check_vector_eq(trueActive, active));

    Real overlapsArray2[10] = {1, 2, 7, 0, 3, 4, 16, 1, 1.5f, 1.7f};
    //                         L  W  W  L  W  W   W  L   L     W
    overlaps.assign(&overlapsArray2[0], &overlapsArray2[10]);
    UInt trueActive2[6] = {1, 2, 4, 5, 6, 9};
    inhibitionRadius = 3;
    density = 0.5;
    sp.setInhibitionRadius(inhibitionRadius);
    sp.inhibitColumnsLocal_(overlaps, density, active);
    ASSERT_TRUE(active.size() == 6);
    ASSERT_TRUE(check_vector_eq(trueActive2, active));

    // Test arbitration

    Real overlapsArray3[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    //                         W  W  L  L  W  W  L  L  L  W
    overlaps.assign(&overlapsArray3[0], &overlapsArray3[10]);
    UInt trueActive3[4] = {0, 1, 4, 5};
    inhibitionRadius = 3;
    density = 0.25;
    sp.setInhibitionRadius(inhibitionRadius);
    sp.inhibitColumnsLocal_(overlaps, density, active);

    ASSERT_TRUE(active.size() == 4ul);
    ASSERT_TRUE(check_vector_eq(trueActive3, active));
  }
}

TEST(SpatialPoolerTest, testIsUpdateRound) {
  SpatialPooler sp;
  sp.setUpdatePeriod(50);
  sp.setIterationNum(1);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(39);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(50);
  ASSERT_TRUE(sp.isUpdateRound_());
  sp.setIterationNum(1009);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(1250);
  ASSERT_TRUE(sp.isUpdateRound_());

  sp.setUpdatePeriod(125);
  sp.setIterationNum(0);
  ASSERT_TRUE(sp.isUpdateRound_());
  sp.setIterationNum(200);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(249);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(1330);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(1249);
  ASSERT_TRUE(!sp.isUpdateRound_());
  sp.setIterationNum(1375);
  ASSERT_TRUE(sp.isUpdateRound_());
}


TEST(SpatialPoolerTest, testSetPermanence) {
  vector<UInt> inputDim;
  vector<UInt> columnDim;
  UInt numInputs = 5;
  UInt numColumns = 5;
  SpatialPooler sp;
  setup(sp, numInputs, numColumns);
  UInt potential[5] = {1, 1, 1, 1, 1}; // Fully connected, all possible synapses are potential synapses.
  Real permArr[5][5] = {{-0.10f, 0.500f, 0.400f, 0.010f, 0.020f},
                        {0.300f, 0.010f, 0.020f, 0.120f, 0.090f},
                        {0.070f, 0.050f, 1.030f, 0.190f, 0.060f},
                        {0.180f, 0.090f, 0.110f, 0.010f, 0.030f},
                        {0.200f, 0.101f, 0.050f, -0.09f, 1.100f}};

  Real truePerm[5][5] = {{0.000f, 0.500f, 0.400f, 0.010f, 0.020f},
                         // Clip     -       -       -       -
                         {0.300f, 0.010f, 0.020f, 0.120f, 0.090f},
                         //  -       -       -       -       -
                         {0.070f, 0.050f, 1.000f, 0.190f, 0.060f},
                         //  -       -      Clip     -       -
                         {0.180f, 0.090f, 0.110f, 0.010f, 0.030f},
                         //  -       -       -       -       -
                         {0.200f, 0.101f, 0.050f, 0.000f, 1.000f}};
                         //  -       -       -      Clip    Clip
  UInt trueConnectedSynapses[5][5] = {{0, 1, 1, 0, 0},
                                      {1, 0, 0, 1, 0},
                                      {0, 0, 1, 1, 0},
                                      {1, 0, 1, 0, 0},
                                      {1, 1, 0, 0, 1}};
  UInt trueConnectedCount[5] = {2, 2, 2, 2, 3};
  for (UInt i = 0; i < 5; i++) {
    sp.setPotential(i, potential);
    sp.setPermanence(i, permArr[i]);
    auto permArr = new Real[numInputs];
    auto connectedArr = new UInt[numInputs];
    auto connectedCountsArr = new UInt[numColumns];
    sp.getPermanence(i, permArr);
    sp.getConnectedSynapses(i, connectedArr);
    sp.getConnectedCounts(connectedCountsArr);
    ASSERT_TRUE(check_vector_eq(truePerm[i], permArr, numInputs));
    ASSERT_TRUE(
        check_vector_eq(trueConnectedSynapses[i], connectedArr, numInputs));
    ASSERT_TRUE(trueConnectedCount[i] == connectedCountsArr[i]);
    delete[] permArr;
    delete[] connectedArr;
    delete[] connectedCountsArr;
  }
}


TEST(SpatialPoolerTest, testInitPermanence) {
  vector<UInt> inputDim;
  vector<UInt> columnDim;
  inputDim.push_back(8);
  columnDim.push_back(20);

  SpatialPooler sp;
  Real synPermConnected = 0.2f;
  Real synPermActiveInc = 0.05f;
  sp.initialize(inputDim, columnDim, 16u, 0.5f, true, 0.5f, 0u, 0.01f, 0.1f,
                synPermConnected);
  sp.setSynPermActiveInc(synPermActiveInc);

  UInt arr[8] = {0, 1, 1, 0, 0, 1, 0, 1};
  vector<UInt> potential(&arr[0], &arr[8]);
  vector<Real> perm = sp.initPermanence_(potential, 1.0);
  for (UInt i = 0; i < 8; i++)
    if (potential[i])
      ASSERT_GE(perm[i], synPermConnected);
    else
      ASSERT_LT(perm[i], 1e-5);

  perm = sp.initPermanence_(potential, 0);
  for (UInt i = 0; i < 8; i++)
    if (potential[i])
      ASSERT_LE(perm[i], synPermConnected);
    else
      ASSERT_LT(perm[i], 1e-5);

  inputDim[0] = 100;
  sp.initialize(inputDim, columnDim, 16u, 0.5f, true, 0.5f, 0u, 0.01f, 0.1f,
                synPermConnected);
  sp.setSynPermActiveInc(synPermActiveInc);
  potential.clear();

  for (UInt i = 0; i < 100; i++)
    potential.push_back(1);

  perm = sp.initPermanence_(potential, 0.5);
  int count = 0;
  for (UInt i = 0; i < 100; i++) {
    if (perm[i] >= synPermConnected)
      count++;
  }
  ASSERT_TRUE(count > 5 && count < 95);
}


TEST(SpatialPoolerTest, testInitPermConnected) {
  Real synPermConnected = 0.2f;
  Real synPermMax       = 1.0f;
  SpatialPooler sp({10}, {10}, 16u, 0.5f, true, 0.5f, 0u, 0.01f, 0.1f,
                   synPermConnected);

  for (UInt i = 0; i < 100; i++) {
    Real permVal = sp.initPermConnected_();
    ASSERT_GE(permVal, synPermConnected);
    ASSERT_LE(permVal, synPermMax);
  }
}


TEST(SpatialPoolerTest, testInitPermNonConnected) {
  Real32 synPermConnected = 0.2f;
  SpatialPooler sp({10}, {10}, 16u, 0.5f, true, 0.5f, 0u, 0.01f, 0.1f,
                   synPermConnected);
  EXPECT_EQ(sp.getSynPermMax(), 1.0f);

  for (UInt i = 0; i < 100; i++) {
    Real permVal = sp.initPermNonConnected_();
    ASSERT_GE(permVal, 0.0f);
    ASSERT_LE(permVal, synPermConnected);
  }
}


TEST(SpatialPoolerTest, testinitMapColumn) {
  {
    // Test 1D.
    SpatialPooler sp;
    setup(sp, /*inputDimensions*/ 12, /*columnDimensions*/ 4);

    EXPECT_EQ(1ul, sp.initMapColumn_(0));
    EXPECT_EQ(4ul, sp.initMapColumn_(1));
    EXPECT_EQ(7ul, sp.initMapColumn_(2));
    EXPECT_EQ(10ul,sp.initMapColumn_(3));
  }

  {
    // Test 1D with same dimensions of columns and inputs.
    SpatialPooler sp;
    setup(sp, /*inputDimensions*/ 4, /*columnDimensions*/ 4);

    EXPECT_EQ(0ul, sp.initMapColumn_(0));
    EXPECT_EQ(1ul, sp.initMapColumn_(1));
    EXPECT_EQ(2ul, sp.initMapColumn_(2));
    EXPECT_EQ(3ul, sp.initMapColumn_(3));
  }

  {
    // Test 2D.
    SpatialPooler sp;
    setup(sp, 
        /*inputDimensions*/ {36, 12},
        /*columnDimensions*/ {12, 4});

    EXPECT_EQ(13ul, sp.initMapColumn_(0));
    EXPECT_EQ(49ul, sp.initMapColumn_(4));
    EXPECT_EQ(52ul, sp.initMapColumn_(5));
    EXPECT_EQ(58ul, sp.initMapColumn_(7));
    EXPECT_EQ(418ul,sp.initMapColumn_(47));
  }

  {
    // Test 2D, some input dimensions smaller than column dimensions.
    SpatialPooler sp;
    setup(sp, 
        /*inputDimensions*/ {3, 5},
        /*columnDimensions*/ {4, 4});

    EXPECT_EQ(0ul, sp.initMapColumn_(0));
    EXPECT_EQ(4ul, sp.initMapColumn_(3));
    EXPECT_EQ(14ul,sp.initMapColumn_(15));
  }
}


TEST(SpatialPoolerTest, testinitMapPotential1D) {
  vector<UInt> inputDim, columnDim;
  inputDim.push_back(12);
  columnDim.push_back(4);
  UInt potentialRadius = 2;

  SpatialPooler sp;
  setup(sp, inputDim, columnDim);
  sp.setPotentialRadius(potentialRadius);

  vector<UInt> mask;

  // Test without wrapAround and potentialPct = 1
  sp.setPotentialPct(1.0);

  UInt expectedMask1[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  mask = sp.initMapPotential_(0, false);
  ASSERT_TRUE(check_vector_eq(expectedMask1, mask));

  UInt expectedMask2[12] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0};
  mask = sp.initMapPotential_(2, false);
  ASSERT_TRUE(check_vector_eq(expectedMask2, mask));

  // Test with wrapAround and potentialPct = 1
  sp.setPotentialPct(1.0);

  UInt expectedMask3[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
  mask = sp.initMapPotential_(0, true);
  ASSERT_TRUE(check_vector_eq(expectedMask3, mask));

  UInt expectedMask4[12] = {1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
  mask = sp.initMapPotential_(3, true);
  ASSERT_TRUE(check_vector_eq(expectedMask4, mask));

  // Test with potentialPct < 1
  sp.setPotentialPct(0.5);
  UInt supersetMask1[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
  mask = sp.initMapPotential_(0, true);
  ASSERT_TRUE(accumulate(mask.begin(), mask.end(), 0.0f) == 3u);

  UInt unionMask1[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (UInt i = 0; i < 12; i++) {
    unionMask1[i] = supersetMask1[i] | mask.at(i);
  }

  ASSERT_TRUE(check_vector_eq(unionMask1, supersetMask1, 12));
}


TEST(SpatialPoolerTest, testinitMapPotential2D) {
  vector<UInt> inputDim, columnDim;
  inputDim.push_back(6);
  inputDim.push_back(12);
  columnDim.push_back(2);
  columnDim.push_back(4);
  UInt potentialRadius = 1;
  Real potentialPct = 1.0;

  SpatialPooler sp;
  setup(sp, inputDim, columnDim);
  sp.setPotentialRadius(potentialRadius);
  sp.setPotentialPct(potentialPct);

  vector<UInt> mask;

  // Test without wrapAround
  UInt expectedMask1[72] = {
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  mask = sp.initMapPotential_(0, false);
  ASSERT_TRUE(check_vector_eq(expectedMask1, mask));

  UInt expectedMask2[72] = {
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  mask = sp.initMapPotential_(2, false);
  ASSERT_TRUE(check_vector_eq(expectedMask2, mask));

  // Test with wrapAround
  potentialRadius = 2;
  sp.setPotentialRadius(potentialRadius);
  UInt expectedMask3[72] = {
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
  mask = sp.initMapPotential_(0, true);
  ASSERT_TRUE(check_vector_eq(expectedMask3, mask));

  UInt expectedMask4[72] = {
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
  mask = sp.initMapPotential_(3, true);
  ASSERT_TRUE(check_vector_eq(expectedMask4, mask));
}


TEST(SpatialPoolerTest, getOverlaps) {
  SpatialPooler sp;
  const vector<UInt> inputDim = {5};
  const vector<UInt> columnDim = {3};
  setup(sp, inputDim, columnDim);

  UInt potential[5] = {1, 1, 1, 1, 1};
  sp.setPotential(0, potential);
  sp.setPotential(1, potential);
  sp.setPotential(2, potential);

  Real permanence0[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  sp.setPermanence(0, permanence0);
  Real permanence1[5] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
  sp.setPermanence(1, permanence1);
  Real permanence2[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  sp.setPermanence(2, permanence2);

  vector<Real> boostFactors = {1.0f, 2.0f, 3.0f};
  sp.setBoostFactors(boostFactors.data());
  sp.setBoostStrength(0.0f); //default, effectively disables boosting

  SDR input( {5}); 
  input.setDense(vector<UInt>{1, 1, 1, 1, 1});
  SDR activeColumns( {3} );
  activeColumns.setDense(vector<UInt>{0, 0, 0});
  sp.compute(input, true, activeColumns);

  //overlaps (not boosted)
  const auto &overlaps = sp.getOverlaps();
  const vector<SynapseIdx> expectedOverlaps = {0, 3, 5};
  EXPECT_EQ(expectedOverlaps, overlaps);

  //boosted overlaps, but boost strength=0.0
  const auto& boostedOverlaps = sp.getBoostedOverlaps();
  const vector<Real> expectedBoostedOverlaps = {0.0f, 3.0f, 5.0f}; //same as orig above (but float)
  EXPECT_EQ(expectedBoostedOverlaps, boostedOverlaps) << "SP with boost strength " << sp.getBoostStrength() << " must not change boosting ";

  //boosted overlaps, but boost strength=2.0
  //recompute
  sp.setBoostFactors(boostFactors.data());
  sp.setBoostStrength(2.0f);
  
  activeColumns.setDense(vector<UInt>{0, 0, 0});
  sp.compute(input, true, activeColumns);

  const auto& boostedOverlaps2 = sp.getBoostedOverlaps();
  const vector<Real> expectedBoostedOverlaps2 = {0.0f, 6.0f, 15.0f};
  EXPECT_EQ(expectedBoostedOverlaps2, boostedOverlaps2) << "SP with boost strength " << sp.getBoostStrength() << " must change boosting ";
}


TEST(SpatialPoolerTest, ZeroOverlap_NoStimulusThreshold_GlobalInhibition) {
  const UInt inputSize = 10;
  const UInt nColumns = 20;

  SpatialPooler sp({inputSize}, {nColumns},
                   /*potentialRadius*/ 10,
                   /*potentialPct*/ 0.5f,
                   /*globalInhibition*/ true,
                   /*localAreaDensity*/ 0.1f,
                   /*stimulusThreshold*/ 0,
                   /*synPermInactiveDec*/ 0.008f,
                   /*synPermActiveInc*/ 0.05f,
                   /*synPermConnected*/ 0.1f,
                   /*minPctOverlapDutyCycles*/ 0.001f,
                   /*dutyCyclePeriod*/ 1000,
                   /*boostStrength*/ 10.0f,
                   /*seed*/ 1,
                   /*spVerbosity*/ 0,
                   /*wrapAround*/ true);

  SDR input( {inputSize} );
  SDR activeColumns( {nColumns} );
  sp.compute(input, true, activeColumns);

  EXPECT_GE(activeColumns.getSum(), 1u) << "zero overlap, but with no stim threshold -> some should be active";
}


TEST(SpatialPoolerTest, ZeroOverlap_StimulusThreshold_GlobalInhibition) {
  const UInt inputSize = 10;
  const UInt nColumns = 20;

  SpatialPooler sp({inputSize}, {nColumns},
                   /*potentialRadius*/ 5,
                   /*potentialPct*/ 0.5f,
                   /*globalInhibition*/ true,
                   /*localAreaDensity*/ 0.1f,
                   /*stimulusThreshold*/ 1,
                   /*synPermInactiveDec*/ 0.008f,
                   /*synPermActiveInc*/ 0.05f,
                   /*synPermConnected*/ 0.1f,
                   /*minPctOverlapDutyCycles*/ 0.001f,
                   /*dutyCyclePeriod*/ 1000,
                   /*boostStrength*/ 10.0f,
                   /*seed*/ 1,
                   /*spVerbosity*/ 0,
                   /*wrapAround*/ true);

  SDR input( {inputSize} );
  SDR activeColumns( {nColumns} );
  sp.compute(input, true, activeColumns);

  EXPECT_EQ(0ul, activeColumns.getSum()) << "Zero overlap and stimulus threshold > 0 -> none active";
}


TEST(SpatialPoolerTest, ZeroOverlap_NoStimulusThreshold_LocalInhibition) {
  const UInt inputSize = 10;
  const UInt nColumns = 20;

  SpatialPooler sp({inputSize}, {nColumns},
                   /*potentialRadius*/ 5,
                   /*potentialPct*/ 0.5f,
                   /*globalInhibition*/ false,
                   /*localAreaDensity*/ 0.1f,
                   /*stimulusThreshold*/ 0,
                   /*synPermInactiveDec*/ 0.008f,
                   /*synPermActiveInc*/ 0.05f,
                   /*synPermConnected*/ 0.1f,
                   /*minPctOverlapDutyCycles*/ 0.001f,
                   /*dutyCyclePeriod*/ 1000,
                   /*boostStrength*/ 10.0f,
                   /*seed*/ 1,
                   /*spVerbosity*/ 0,
                   /*wrapAround*/ true);

  SDR input( {inputSize} );
  SDR activeColumns( {nColumns} );
  sp.compute(input, true, activeColumns);

  EXPECT_GE(activeColumns.getSum(), 1u) << "No overlap, but also no threshold -> some can be active";
}


TEST(SpatialPoolerTest, ZeroOverlap_StimulusThreshold_LocalInhibition) {
  const UInt inputSize = 10;
  const UInt nColumns = 20;

  SpatialPooler sp({inputSize}, {nColumns},
                   /*potentialRadius*/ 10,
                   /*potentialPct*/ 0.5f,
                   /*globalInhibition*/ false,
                   /*localAreaDensity*/ 0.1f,
                   /*stimulusThreshold*/ 1,
                   /*synPermInactiveDec*/ 0.008f,
                   /*synPermActiveInc*/ 0.05f,
                   /*synPermConnected*/ 0.1f,
                   /*minPctOverlapDutyCycles*/ 0.001f,
                   /*dutyCyclePeriod*/ 1000,
                   /*boostStrength*/ 10.0f,
                   /*seed*/ 1,
                   /*spVerbosity*/ 0,
                   /*wrapAround*/ true);

  SDR input({inputSize});
  SDR activeColumns({nColumns});
  sp.compute(input, true, activeColumns);

  EXPECT_EQ(0ul, activeColumns.getSum()) << "No overlap and threshold > 0 -> none will be active";
}


TEST(SpatialPoolerTest, testSaveLoad) {
  const char *filename = "SpatialPoolerSerialization.tmp";
  SpatialPooler sp1, sp2;
  UInt numInputs = 6;
  UInt numColumns = 12;
  setup(sp1, numInputs, numColumns);

  ofstream outfile;
  outfile.open(filename, ifstream::binary);
  sp1.save(outfile);
  outfile.close();

  ifstream infile(filename, ifstream::binary);
  sp2.load(infile);
  infile.close();

  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;

  check_spatial_eq(sp1, sp2);
}


TEST(SpatialPoolerTest, testSerialization2) {
  Random random(10);

  const UInt inputSize = 200;
  const UInt numColumns = 200;

  SpatialPooler sp1;
  sp1.initialize({inputSize}, {numColumns});

  SDR input({inputSize});
  SDR output({numColumns});

  for (UInt i = 0; i < 100; ++i) {
    input.randomize(0.15f, random); //15% bits change
    EXPECT_GT(input.getSum(), (size_t)0) << "No input!";
    sp1.compute(input, true, output);
  }


  // Save initial trained model
  ofstream osC("outC.stream", ofstream::binary);
	osC.precision(std::numeric_limits<double>::digits10 + 1);
	osC.precision(std::numeric_limits<float>::digits10 + 1);
  sp1.save(osC);
  osC.close();

  htm::Timer testTimer;

  for (UInt i = 0; i < 10; ++i) {
    // Create new input
    input.randomize(0.24f, random);

    // Get expected output
    SDR  outputBaseline({numColumns});
    sp1.compute(input, true, outputBaseline);

    {
      SpatialPooler spTemp;

      testTimer.start();

      // Deserialize
      ifstream is("outC.stream", ifstream::binary);
      spTemp.load(is);
      is.close();

      // Feed new record through
      SDR outputC({numColumns});
      spTemp.compute(input, true, outputC);

      // Serialize
      ofstream os("outC.stream", ofstream::binary);
	    os.precision(std::numeric_limits<double>::digits10 + 1);
	    os.precision(std::numeric_limits<float>::digits10 + 1);
      spTemp.save(os);
      os.close();

      testTimer.stop();

      EXPECT_EQ(outputBaseline, outputC);
    }
  }

//  cout << "[          ] Timing for SP serialization: " << testTimer.getElapsed() << "sec" << endl;

  remove("outC.stream");
}


TEST(SpatialPoolerTest, testSaveLoad_ar) {
  const char *filename = "SpatialPoolerSerializationAR.tmp";
  SpatialPooler sp1, sp2;
  UInt numInputs = 6;
  UInt numColumns = 12;
  setup(sp1, numInputs, numColumns);

  sp1.saveToFile(filename);
  sp2.loadFromFile(filename);

  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;

  check_spatial_eq(sp1, sp2);
}



TEST(SpatialPoolerTest, testSerialization_ar) {
  Random random(10);

  const UInt inputSize = 200;
  const UInt numColumns = 200;

  vector<UInt> inputDims{inputSize};
  vector<UInt> colDims{numColumns};

  SpatialPooler sp1;
  sp1.initialize(inputDims, colDims);

  SDR input(inputDims);
  SDR output(colDims);

  for (UInt i = 0; i < 100; ++i) {
    input.randomize(0.05f, random); //5% random ON
    sp1.compute(input, true, output);
  }

  // Now we reuse the last input to test after serialization

  auto activeColumnsBefore = output.getSparse();

  // Save initial trained model
  stringstream ss;
	ss.precision(std::numeric_limits<double>::digits10 + 1);
	ss.precision(std::numeric_limits<float>::digits10 + 1);
  sp1.save(ss);

  SpatialPooler sp2;

  htm::Timer testTimer;

  for (UInt i = 0; i < 6; ++i) {
    // Create new input
    input.randomize(0.05f, random);

    // Get expected output
    SDR outputBaseline(output);
    sp1.compute(input, true, outputBaseline);

    // C - Next do old version
    {
      SpatialPooler spTemp;
      testTimer.start();

      // Deserialize
      ss.seekg(0);
      spTemp.load(ss);

      // Feed new record through
      SDR outputC({numColumns});
      spTemp.compute(input, true, outputC);

      // Serialize
      ss.clear();
      spTemp.save(ss);

      testTimer.stop();

      EXPECT_EQ(outputBaseline, outputC);
    }
  }
  ss.clear();

  cout << "[          ] Timing for SP serialization: " << testTimer.getElapsed() << "sec" << endl;
}


TEST(SpatialPoolerTest, testConstructorVsInitialize) {
  // Initialize SP using the constructor
  SpatialPooler sp1(
      /*inputDimensions*/ {100},
      /*columnDimensions*/ {100},
      /*potentialRadius*/ 16,
      /*potentialPct*/ 0.5f,
      /*globalInhibition*/ true,
      /*localAreaDensity*/ 0.1f,
      /*stimulusThreshold*/ 0,
      /*synPermInactiveDec*/ 0.008f,
      /*synPermActiveInc*/ 0.05f,
      /*synPermConnected*/ 0.1f,
      /*minPctOverlapDutyCycles*/ 0.001f,
      /*dutyCyclePeriod*/ 1000,
      /*boostStrength*/ 0.0f,
      /*seed*/ 1,
      /*spVerbosity*/ 0,
      /*wrapAround*/ true);

  // Initialize SP using the "initialize" method
  SpatialPooler sp2;
  sp2.initialize(
      /*inputDimensions*/ {100},
      /*columnDimensions*/ {100},
      /*potentialRadius*/ 16,
      /*potentialPct*/ 0.5f,
      /*globalInhibition*/ true,
      /*localAreaDensity*/ 0.1f,
      /*stimulusThreshold*/ 0,
      /*synPermInactiveDec*/ 0.008f,
      /*synPermActiveInc*/ 0.05f,
      /*synPermConnected*/ 0.1f,
      /*minPctOverlapDutyCycles*/ 0.001f,
      /*dutyCyclePeriod*/ 1000,
      /*boostStrength*/ 0.0,
      /*seed*/ 1,
      /*spVerbosity*/ 0,
      /*wrapAround*/ true);

  // The two SP should be the same
  check_spatial_eq(sp1, sp2);
  EXPECT_EQ(sp1, sp2);
  EXPECT_TRUE(sp1 == sp2) << "Spatial Poolers not equal";
}


TEST(SpatialPoolerTest, ExactOutput) { 
  // Silver is an SDR that is loaded by direct initalization from a vector.
  SDR silver_sdr({ 200 });
  SDR_sparse_t data = {
    4, 64, 74, 78, 85, 113, 125, 126, 127, 153
  };
  silver_sdr.setSparse(data);


  // Gold tests initalizing an SDR from a manually created string in JSON format.
	// hint: you can generate this string using
	//       silver_sdr.save(std::cout, JSON);
  string gold = "{\"dimensions\": [200],\"sparse\": [4, 64, 74, 78, 85, 113, 125, 126, 127, 153]}";
  std::stringstream gold_stream( gold );
  SDR gold_sdr;
  gold_sdr.load( gold_stream, JSON );
	
	// both SCR's should be the same
  EXPECT_EQ(silver_sdr, gold_sdr);

  SDR inputs({ 1000 });
  SDR columns({ 200 });
  SpatialPooler sp({inputs.dimensions}, {columns.dimensions},
                   /*potentialRadius*/ 99999,
                   /*potentialPct*/ 0.5f,
                   /*globalInhibition*/ true,
                   /*localAreaDensity*/ 0.05f,
                   /*stimulusThreshold*/ 3u,
                   /*synPermInactiveDec*/ 0.008f,
                   /*synPermActiveInc*/ 0.05f,
                   /*synPermConnected*/ 0.1f,
                   /*minPctOverlapDutyCycles*/ 0.001f,
                   /*dutyCyclePeriod*/ 200,
                   /*boostStrength*/ 10.0f,
                   /*seed*/ 42,
                   /*spVerbosity*/ 0,
                   /*wrapAround*/ true);

  for(UInt i = 0; i < 1000; i++) {
    Random rng(i + 1); // Random seed 0 is magic, don't use it.
    inputs.randomize( 0.15f, rng );
    sp.compute(inputs, true, columns);
  }
  ASSERT_EQ( columns, gold_sdr );
}


} // end anonymous namespace
