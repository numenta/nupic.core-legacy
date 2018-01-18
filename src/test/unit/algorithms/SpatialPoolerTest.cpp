/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of unit tests for SpatialPooler
 */

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include "gtest/gtest.h"

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::spatial_pooler;

namespace {
  UInt countNonzero(const vector<UInt>& vec)
  {
    UInt count = 0;

    for (UInt x : vec)
    {
      if (x > 0)
      {
        count++;
      }
    }

    return count;
  }

  bool almost_eq(Real a, Real b)
  {
    Real diff = a - b;
    return (diff > -1e-5 && diff < 1e-5);
  }

  bool check_vector_eq(UInt arr[], vector<UInt> vec)
  {
    for (UInt i = 0; i < vec.size(); i++) {
      if (arr[i] != vec[i]) {
        return false;
      }
    }
    return true;
  }

  bool check_vector_eq(Real arr[], vector<Real> vec)
  {
    for (UInt i = 0; i < vec.size(); i++) {
      if (!almost_eq(arr[i],vec[i])) {
        return false;
      }
    }
    return true;
  }

  bool check_vector_eq(UInt arr1[], UInt arr2[], UInt n)
  {
    for (UInt i = 0; i < n; i++) {
      if (arr1[i] != arr2[i]) {
        return false;
      }
    }
    return true;
  }

  bool check_vector_eq(Real arr1[], Real arr2[], UInt n)
  {
    for (UInt i = 0; i < n; i++) {
      if (!almost_eq(arr1[i], arr2[i])) {
        return false;
      }
    }
    return true;
  }

  bool check_vector_eq(vector<UInt> vec1, vector<UInt> vec2)
  {
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

  bool check_vector_eq(vector<Real> vec1, vector<Real> vec2)
  {
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

  void check_spatial_eq(SpatialPooler sp1, SpatialPooler sp2)
  {
    UInt numColumns = sp1.getNumColumns();
    UInt numInputs = sp2.getNumInputs();

    ASSERT_TRUE(sp1.getNumColumns() == sp2.getNumColumns());
    ASSERT_TRUE(sp1.getNumInputs() == sp2.getNumInputs());
    ASSERT_TRUE(sp1.getPotentialRadius() ==
              sp2.getPotentialRadius());
    ASSERT_TRUE(sp1.getPotentialPct() == sp2.getPotentialPct());
    ASSERT_TRUE(sp1.getGlobalInhibition() ==
              sp2.getGlobalInhibition());
    ASSERT_TRUE(sp1.getNumActiveColumnsPerInhArea() ==
              sp2.getNumActiveColumnsPerInhArea());
    ASSERT_TRUE(almost_eq(sp1.getLocalAreaDensity(),
              sp2.getLocalAreaDensity()));
    ASSERT_TRUE(sp1.getStimulusThreshold() ==
              sp2.getStimulusThreshold());
    ASSERT_TRUE(sp1.getDutyCyclePeriod() == sp2.getDutyCyclePeriod());
    ASSERT_TRUE(almost_eq(sp1.getBoostStrength(), sp2.getBoostStrength()));
    ASSERT_TRUE(sp1.getIterationNum() == sp2.getIterationNum());
    ASSERT_TRUE(sp1.getIterationLearnNum() ==
              sp2.getIterationLearnNum());
    ASSERT_TRUE(sp1.getSpVerbosity() == sp2.getSpVerbosity());
    ASSERT_TRUE(sp1.getWrapAround() == sp2.getWrapAround());
    ASSERT_TRUE(sp1.getUpdatePeriod() == sp2.getUpdatePeriod());
    ASSERT_TRUE(almost_eq(sp1.getSynPermTrimThreshold(),
              sp2.getSynPermTrimThreshold()));
    cout << "check: " << sp1.getSynPermActiveInc() << " " <<
      sp2.getSynPermActiveInc() << endl;
    ASSERT_TRUE(almost_eq(sp1.getSynPermActiveInc(),
              sp2.getSynPermActiveInc()));
    ASSERT_TRUE(almost_eq(sp1.getSynPermInactiveDec(),
              sp2.getSynPermInactiveDec()));
    ASSERT_TRUE(almost_eq(sp1.getSynPermBelowStimulusInc(),
              sp2.getSynPermBelowStimulusInc()));
    ASSERT_TRUE(almost_eq(sp1.getSynPermConnected(),
              sp2.getSynPermConnected()));
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
    ASSERT_TRUE(check_vector_eq(overlapDutyCycles1, overlapDutyCycles2, numColumns));
    delete[] overlapDutyCycles1;
    delete[] overlapDutyCycles2;

    auto activeDutyCycles1 = new Real[numColumns];
    auto activeDutyCycles2 = new Real[numColumns];
    sp1.getActiveDutyCycles(activeDutyCycles1);
    sp2.getActiveDutyCycles(activeDutyCycles2);
    ASSERT_TRUE(check_vector_eq(activeDutyCycles1, activeDutyCycles2, numColumns));
    delete[] activeDutyCycles1;
    delete[] activeDutyCycles2;

    auto minOverlapDutyCycles1 = new Real[numColumns];
    auto minOverlapDutyCycles2 = new Real[numColumns];
    sp1.getMinOverlapDutyCycles(minOverlapDutyCycles1);
    sp2.getMinOverlapDutyCycles(minOverlapDutyCycles2);
    ASSERT_TRUE(check_vector_eq(minOverlapDutyCycles1, minOverlapDutyCycles2, numColumns));
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

  void setup(SpatialPooler& sp, UInt numInputs,
                                UInt numColumns)
  {
    vector<UInt> inputDim;
    vector<UInt> columnDim;
    inputDim.push_back(numInputs);
    columnDim.push_back(numColumns);
    sp.initialize(inputDim,columnDim);
  }

  TEST(SpatialPoolerTest, testUpdateInhibitionRadius)
  {
    SpatialPooler sp;
    vector<UInt> colDim, inputDim;
    colDim.push_back(57);
    colDim.push_back(31);
    colDim.push_back(2);
    inputDim.push_back(1);
    inputDim.push_back(1);
    inputDim.push_back(1);

    sp.initialize(inputDim, colDim);
    sp.setGlobalInhibition(true);
    ASSERT_TRUE(sp.getInhibitionRadius() == 57);

    colDim.clear();
    inputDim.clear();
    // avgColumnsPerInput = 4
    // avgConnectedSpanForColumn = 3
    UInt numInputs = 3;
    inputDim.push_back(numInputs);
    UInt numCols = 12;
    colDim.push_back(numCols);
    sp.initialize(inputDim, colDim);
    sp.setGlobalInhibition(false);

    for (UInt i = 0; i < numCols; i++) {
      Real permArr[] = {1, 1, 1};
      sp.setPermanence(i, permArr);
    }
    UInt trueInhibitionRadius = 6;
    // ((3 * 4) - 1)/2 => round up
    sp.updateInhibitionRadius_();
    ASSERT_TRUE(trueInhibitionRadius == sp.getInhibitionRadius());

    colDim.clear();
    inputDim.clear();
    // avgColumnsPerInput = 1.2
    // avgConnectedSpanForColumn = 0.5
    numInputs = 5;
    inputDim.push_back(numInputs);
    numCols = 6;
    colDim.push_back(numCols);
    sp.initialize(inputDim, colDim);
    sp.setGlobalInhibition(false);

    for (UInt i = 0; i < numCols; i++) {
      Real permArr[] = {1, 0, 0, 0, 0};
      if (i % 2 == 0) {
        permArr[0] = 0;
      }
      sp.setPermanence(i, permArr);
    }
    trueInhibitionRadius = 1;
    sp.updateInhibitionRadius_();
    ASSERT_TRUE(trueInhibitionRadius == sp.getInhibitionRadius());

    colDim.clear();
    inputDim.clear();
    // avgColumnsPerInput = 2.4
    // avgConnectedSpanForColumn = 2
    numInputs = 5;
    inputDim.push_back(numInputs);
    numCols = 12;
    colDim.push_back(numCols);
    sp.initialize(inputDim, colDim);
    sp.setGlobalInhibition(false);

    for (UInt i = 0; i < numCols; i++) {
      Real permArr[] = {1, 1, 0, 0, 0};
      sp.setPermanence(i,permArr);
    }
    trueInhibitionRadius = 2;
    // ((2.4 * 2) - 1)/2 => round up
    sp.updateInhibitionRadius_();
    ASSERT_TRUE(trueInhibitionRadius == sp.getInhibitionRadius());
  }

  TEST(SpatialPoolerTest, testUpdateMinDutyCycles)
  {
    SpatialPooler sp;
    UInt numColumns = 10;
    UInt numInputs = 5;
    setup(sp, numInputs, numColumns);
    sp.setMinPctOverlapDutyCycles(0.01);

    Real initOverlapDuty[10] = {0.01, 0.001, 0.02, 0.3, 0.012, 0.0512,
                                0.054, 0.221, 0.0873, 0.309};

    Real initActiveDuty[10] = {0.01, 0.045, 0.812, 0.091, 0.001, 0.0003,
                               0.433, 0.136, 0.211, 0.129};

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

    ASSERT_TRUE(check_vector_eq(resultMinOverlap, resultMinOverlapGlobal,
                              numColumns));

    sp.setGlobalInhibition(false);
    sp.updateMinDutyCycles_();
    sp.getMinOverlapDutyCycles(resultMinOverlap);

    ASSERT_TRUE(!check_vector_eq(resultMinOverlap, resultMinOverlapGlobal,
                               numColumns));

  }

  TEST(SpatialPoolerTest, testUpdateMinDutyCyclesGlobal) {
    SpatialPooler sp;
    UInt numColumns = 5;
    UInt numInputs = 5;
    setup(sp, numInputs, numColumns);
    Real minPctOverlap;

    minPctOverlap = 0.01;

    sp.setMinPctOverlapDutyCycles(minPctOverlap);

    Real overlapArr1[] =
      {0.06, 1, 3, 6, 0.5};
    Real activeArr1[] =
      {0.6, 0.07, 0.5, 0.4, 0.3};

    sp.setOverlapDutyCycles(overlapArr1);
    sp.setActiveDutyCycles(activeArr1);

    Real trueMinOverlap1 = 0.01 * 6;

    sp.updateMinDutyCyclesGlobal_();
    Real resultOverlap1[5];
    sp.getMinOverlapDutyCycles(resultOverlap1);
    for (UInt i = 0; i < numColumns; i++) {
      ASSERT_TRUE(resultOverlap1[i] == trueMinOverlap1);
    }


    minPctOverlap = 0.015;

    sp.setMinPctOverlapDutyCycles(minPctOverlap);

    Real overlapArr2[] = {0.86, 2.4, 0.03, 1.6, 1.5};
    Real activeArr2[] = {0.16, 0.007, 0.15, 0.54, 0.13};

    sp.setOverlapDutyCycles(overlapArr2);
    sp.setActiveDutyCycles(activeArr2);

    Real trueMinOverlap2 = 0.015 * 2.4;

    sp.updateMinDutyCyclesGlobal_();
    Real resultOverlap2[5];
    sp.getMinOverlapDutyCycles(resultOverlap2);
    for (UInt i = 0; i < numColumns; i++) {
      ASSERT_TRUE(almost_eq(resultOverlap2[i],trueMinOverlap2));
    }


    minPctOverlap = 0.015;

    sp.setMinPctOverlapDutyCycles(minPctOverlap);

    Real overlapArr3[] = {0, 0, 0, 0, 0};
    Real activeArr3[] = {0, 0, 0, 0, 0};

    sp.setOverlapDutyCycles(overlapArr3);
    sp.setActiveDutyCycles(activeArr3);

    Real trueMinOverlap3 = 0;

    sp.updateMinDutyCyclesGlobal_();
    Real resultOverlap3[5];
    sp.getMinOverlapDutyCycles(resultOverlap3);
    for (UInt i = 0; i < numColumns; i++) {
      ASSERT_TRUE(almost_eq(resultOverlap3[i],trueMinOverlap3));
    }
  }

  TEST(SpatialPoolerTest, testUpdateMinDutyCyclesLocal)
  {
    // wrapAround=false
    {
      UInt numColumns = 8;
      SpatialPooler sp(
        /*inputDimensions*/{5},
        /*columnDimensions*/ {numColumns},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ -1.0,
        /*numActiveColumnsPerInhArea*/ 3,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008,
        /*synPermActiveInc*/ 0.05,
        /*synPermConnected*/ 0.1,
        /*minPctOverlapDutyCycles*/ 0.001,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 0.0,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ false);

      sp.setInhibitionRadius(1);

      Real activeDutyArr[] = {0.9, 0.3, 0.5, 0.7, 0.1, 0.01, 0.08, 0.12};
      sp.setActiveDutyCycles(activeDutyArr);

      Real overlapDutyArr[] = {0.7, 0.1, 0.5, 0.01, 0.78, 0.55, 0.1, 0.001};
      sp.setOverlapDutyCycles(overlapDutyArr);

      sp.setMinPctOverlapDutyCycles(0.2);

      sp.updateMinDutyCyclesLocal_();

      Real trueOverlapArr[] = {0.2*0.7,
                               0.2*0.7,
                               0.2*0.5,
                               0.2*0.78,
                               0.2*0.78,
                               0.2*0.78,
                               0.2*0.55,
                               0.2*0.1};
      Real resultMinOverlapArr[8];
      sp.getMinOverlapDutyCycles(resultMinOverlapArr);
      ASSERT_TRUE(check_vector_eq(resultMinOverlapArr, trueOverlapArr,
                                  numColumns));
    }

    // wrapAround=true
    {
      UInt numColumns = 8;
      SpatialPooler sp(
        /*inputDimensions*/{5},
        /*columnDimensions*/ {numColumns},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ -1.0,
        /*numActiveColumnsPerInhArea*/ 3,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008,
        /*synPermActiveInc*/ 0.05,
        /*synPermConnected*/ 0.1,
        /*minPctOverlapDutyCycles*/ 0.001,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 10.0,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ true);

      sp.setInhibitionRadius(1);

      Real activeDutyArr[] = {0.9, 0.3, 0.5, 0.7, 0.1, 0.01, 0.08, 0.12};
      sp.setActiveDutyCycles(activeDutyArr);

      Real overlapDutyArr[] = {0.7, 0.1, 0.5, 0.01, 0.78, 0.55, 0.1, 0.001};
      sp.setOverlapDutyCycles(overlapDutyArr);

      sp.setMinPctOverlapDutyCycles(0.2);

      sp.updateMinDutyCyclesLocal_();

      Real trueOverlapArr[] = {0.2*0.7,
                               0.2*0.7,
                               0.2*0.5,
                               0.2*0.78,
                               0.2*0.78,
                               0.2*0.78,
                               0.2*0.55,
                               0.2*0.7};
      Real resultMinOverlapArr[8];
      sp.getMinOverlapDutyCycles(resultMinOverlapArr);
      ASSERT_TRUE(check_vector_eq(resultMinOverlapArr, trueOverlapArr,
                                  numColumns));
    }
  }

  TEST(SpatialPoolerTest, testUpdateDutyCycles)
  {
    SpatialPooler sp;
    UInt numInputs = 5;
    UInt numColumns = 5;
    setup(sp, numInputs, numColumns);
    vector<UInt> overlaps;

    Real initOverlapArr1[] = {1, 1, 1, 1, 1};
    sp.setOverlapDutyCycles(initOverlapArr1);
    Real overlapNewVal1[] = {1, 5, 7, 0, 0};
    overlaps.assign(overlapNewVal1, overlapNewVal1+numColumns);
    UInt active[] = {0, 0, 0, 0, 0};

    sp.setIterationNum(2);
    sp.updateDutyCycles_(overlaps, active);

    Real resultOverlapArr1[5];
    sp.getOverlapDutyCycles(resultOverlapArr1);

    Real trueOverlapArr1[] = {1, 1, 1, 0.5, 0.5};
    ASSERT_TRUE(check_vector_eq(resultOverlapArr1, trueOverlapArr1, numColumns));

    sp.setOverlapDutyCycles(initOverlapArr1);
    sp.setIterationNum(2000);
    sp.setUpdatePeriod(1000);
    sp.updateDutyCycles_(overlaps, active);

    Real resultOverlapArr2[5];
    sp.getOverlapDutyCycles(resultOverlapArr2);
    Real trueOverlapArr2[] = {1, 1, 1, 0.999, 0.999};

    ASSERT_TRUE(check_vector_eq(resultOverlapArr2, trueOverlapArr2, numColumns));
  }

  TEST(SpatialPoolerTest, testAvgColumnsPerInput)
  {
    SpatialPooler sp;
    vector<UInt> inputDim, colDim;
    inputDim.clear();
    colDim.clear();

    UInt colDim1[4] =   {2, 2, 2, 2};
    UInt inputDim1[4] = {4, 4, 4, 4};
    Real trueAvgColumnPerInput1 = 0.5;

    inputDim.assign(inputDim1, inputDim1+4);
    colDim.assign(colDim1, colDim1+4);
    sp.initialize(inputDim, colDim);
    Real result = sp.avgColumnsPerInput_();
    ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput1);

    UInt colDim2[4] =   {2, 2, 2, 2};
    UInt inputDim2[4] = {7, 5, 1, 3};
    Real trueAvgColumnPerInput2 = (2.0/7 + 2.0/5 + 2.0/1 + 2/3.0) / 4;

    inputDim.assign(inputDim2, inputDim2+4);
    colDim.assign(colDim2, colDim2+4);
    sp.initialize(inputDim, colDim);
    result = sp.avgColumnsPerInput_();
    ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput2);

    UInt colDim3[2] =   {3, 3};
    UInt inputDim3[2] = {3, 3};
    Real trueAvgColumnPerInput3 = 1;

    inputDim.assign(inputDim3, inputDim3+2);
    colDim.assign(colDim3, colDim3+2);
    sp.initialize(inputDim, colDim);
    result = sp.avgColumnsPerInput_();
    ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput3);


    UInt colDim4[1] =   {25};
    UInt inputDim4[1] = {5};
    Real trueAvgColumnPerInput4 = 5;

    inputDim.assign(inputDim4, inputDim4+1);
    colDim.assign(colDim4, colDim4+1);
    sp.initialize(inputDim, colDim);
    result = sp.avgColumnsPerInput_();
    ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput4);

    UInt colDim5[7] =   {3, 5, 6};
    UInt inputDim5[7] = {3, 5, 6};
    Real trueAvgColumnPerInput5 = 1;

    inputDim.assign(inputDim5, inputDim5+3);
    colDim.assign(colDim5, colDim5+3);
    sp.initialize(inputDim, colDim);
    result = sp.avgColumnsPerInput_();
    ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput5);

    UInt colDim6[4] =   {2, 4, 6, 8};
    UInt inputDim6[4] = {2, 2, 2, 2};
                    //  1  2  3  4
    Real trueAvgColumnPerInput6 = 2.5;

    inputDim.assign(inputDim6, inputDim6+4);
    colDim.assign(colDim6, colDim6+4);
    sp.initialize(inputDim, colDim);
    result = sp.avgColumnsPerInput_();
    ASSERT_FLOAT_EQ(result, trueAvgColumnPerInput6);
  }

  TEST(SpatialPoolerTest, testAvgConnectedSpanForColumn1D)
  {

    SpatialPooler sp;
    UInt numColumns = 9;
    UInt numInputs = 8;
    setup(sp, numInputs, numColumns);

    Real permArr[9][8] =
      {{0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 0, 0, 1},
       {0, 0, 0, 0, 0, 0, 1, 0},
       {0, 0, 1, 0, 0, 0, 1, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 1, 1, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 0, 0, 0},
       {0, 0, 1, 0, 1, 0, 0, 0},
       {1, 1, 1, 1, 1, 1, 1, 1}};

    UInt trueAvgConnectedSpan[9] =
      {7, 5, 1, 5, 0, 2, 3, 3, 8};

    for (UInt i = 0; i < numColumns; i++) {
      sp.setPermanence(i, permArr[i]);
      UInt result = sp.avgConnectedSpanForColumn1D_(i);
      ASSERT_TRUE(result == trueAvgConnectedSpan[i]);
    }
  }

  TEST(SpatialPoolerTest, testAvgConnectedSpanForColumn2D)
  {
    SpatialPooler sp;

    UInt numColumns = 7;
    UInt numInputs = 20;

    vector<UInt> colDim, inputDim;
    Real permArr1[7][20] =
    {{0, 1, 1, 1,
      0, 1, 1, 1,
      0, 1, 1, 1,
      0, 0, 0, 0,
      0, 0, 0, 0},
  // rowspan = 3, colspan = 3, avg = 3

     {1, 1, 1, 1,
      0, 0, 1, 1,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0},
  // rowspan = 2 colspan = 4, avg = 3

     {1, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 1},
  // row span = 5, colspan = 4, avg = 4.5

     {0, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 1, 0, 0,
      0, 1, 0, 0},
  // rowspan = 5, colspan = 1, avg = 3

     {0, 0, 0, 0,
      1, 0, 0, 1,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0},
  // rowspan = 1, colspan = 4, avg = 2.5

     {0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1},
  // rowspan = 2, colspan = 2, avg = 2

     {0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0}
  // rowspan = 0, colspan = 0, avg = 0
    };

    inputDim.push_back(5);
    inputDim.push_back(4);
    colDim.push_back(10);
    colDim.push_back(1);
    sp.initialize(inputDim, colDim);

    UInt trueAvgConnectedSpan1[7] = {3, 3, 4, 3, 2, 2, 0};

    for (UInt i = 0; i < numColumns; i++) {
      sp.setPermanence(i, permArr1[i]);
      UInt result = sp.avgConnectedSpanForColumn2D_(i);
      ASSERT_TRUE(result == (trueAvgConnectedSpan1[i]));
    }

    //1D tests repeated
    numColumns = 9;
    numInputs = 8;

    colDim.clear();
    inputDim.clear();
    inputDim.push_back(numInputs);
    inputDim.push_back(1);
    colDim.push_back(numColumns);
    colDim.push_back(1);

    sp.initialize(inputDim, colDim);

    Real permArr2[9][8] =
      {{0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 0, 0, 1},
       {0, 0, 0, 0, 0, 0, 1, 0},
       {0, 0, 1, 0, 0, 0, 1, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 1, 1, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 0, 0, 0},
       {0, 0, 1, 0, 1, 0, 0, 0},
       {1, 1, 1, 1, 1, 1, 1, 1}};

    UInt trueAvgConnectedSpan2[9] =
      {8, 5, 1, 5, 0, 2, 3, 3, 8};

    for (UInt i = 0; i < numColumns; i++) {
      sp.setPermanence(i, permArr2[i]);
      UInt result = sp.avgConnectedSpanForColumn2D_(i);
      ASSERT_TRUE(result == (trueAvgConnectedSpan2[i] + 1)/2);
    }
  }

  TEST(SpatialPoolerTest, testAvgConnectedSpanForColumnND)
  {
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

    sp.initialize(inputDim, colDim);

    UInt numInputs = 160;
    UInt numColumns = 5;

    Real permArr0[4][4][2][5];
    Real permArr1[4][4][2][5];
    Real permArr2[4][4][2][5];
    Real permArr3[4][4][2][5];
    Real permArr4[4][4][2][5];

    for (UInt i = 0; i < numInputs; i++) {
      ((Real*)permArr0)[i] = 0;
      ((Real*)permArr1)[i] = 0;
      ((Real*)permArr2)[i] = 0;
      ((Real*)permArr3)[i] = 0;
      ((Real*)permArr4)[i] = 0;
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

    sp.setPermanence(0, (Real *) permArr0);
    sp.setPermanence(1, (Real *) permArr1);
    sp.setPermanence(2, (Real *) permArr2);
    sp.setPermanence(3, (Real *) permArr3);
    sp.setPermanence(4, (Real *) permArr4);

    Real trueAvgConnectedSpan[5] =
      {11.0/4, 6.0/4, 14.0/4, 15.0/4, 0};

    for (UInt i = 0; i < numColumns; i++) {
      Real result = sp.avgConnectedSpanForColumnND_(i);
      ASSERT_TRUE(result == trueAvgConnectedSpan[i]);
    }
  }

  TEST(SpatialPoolerTest, testAdaptSynapses)
  {
    SpatialPooler sp;
    UInt numColumns = 4;
    UInt numInputs = 8;
    setup(sp, numInputs, numColumns);

    vector<UInt> activeColumns;
    vector<UInt> inputVector;

    UInt potentialArr1[4][8] =
      {{1, 1, 1, 1, 0, 0, 0, 0},
       {1, 0, 0, 0, 1, 1, 0, 1},
       {0, 0, 1, 0, 0, 0, 1, 0},
       {1, 0, 0, 0, 0, 0, 1, 0}};

    Real permanencesArr1[5][8] =
      {{0.200, 0.120, 0.090, 0.060, 0.000, 0.000, 0.000, 0.000},
       {0.150, 0.000, 0.000, 0.000, 0.180, 0.120, 0.000, 0.450},
       {0.000, 0.000, 0.014, 0.000, 0.000, 0.000, 0.110, 0.000},
       {0.070, 0.000, 0.000, 0.000, 0.000, 0.000, 0.178, 0.000}};

    Real truePermanences1[5][8] =
      {{ 0.300, 0.110, 0.080, 0.160, 0.000, 0.000, 0.000, 0.000},
      //   Inc     Dec   Dec    Inc      -      -      -     -
        {0.250, 0.000, 0.000, 0.000, 0.280, 0.110, 0.000, 0.440},
      //   Inc      -      -     -      Inc    Dec    -     Dec
        {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.210, 0.000},
      //   -      -     Trim     -     -     -       Inc   -
        {0.070, 0.000, 0.000, 0.000, 0.000, 0.000, 0.178, 0.000}};
      //    -      -      -      -      -      -      -       -

    UInt inputArr1[8] = {1, 0, 0, 1, 1, 0, 1, 0};
    UInt activeColumnsArr1[3] = {0, 1, 2};

    for (UInt column = 0; column < numColumns; column++) {
      sp.setPotential(column, potentialArr1[column]);
      sp.setPermanence(column, permanencesArr1[column]);
    }

    activeColumns.assign(&activeColumnsArr1[0], &activeColumnsArr1[3]);

    sp.adaptSynapses_(inputArr1, activeColumns);
    cout << endl;
    for (UInt column = 0; column < numColumns; column++) {
      auto permArr = new Real[numInputs];
      sp.getPermanence(column, permArr);
      ASSERT_TRUE(check_vector_eq(truePermanences1[column],
                                permArr,
                                numInputs));
      delete[] permArr;
    }


    UInt potentialArr2[4][8] =
      {{1, 1, 1, 0, 0, 0, 0, 0},
       {0, 1, 1, 1, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 0, 0, 0},
       {1, 0, 0, 0, 0, 0, 1, 0}};

    Real permanencesArr2[4][8] =
      {{0.200, 0.120, 0.090, 0.000, 0.000, 0.000, 0.000, 0.000},
       {0.000, 0.017, 0.232, 0.400, 0.000, 0.000, 0.000, 0.000},
       {0.000, 0.000, 0.014, 0.051, 0.730, 0.000, 0.000, 0.000},
       {0.170, 0.000, 0.000, 0.000, 0.000, 0.000, 0.380, 0.000}};

    Real truePermanences2[4][8] =
      {{0.30, 0.110, 0.080, 0.000, 0.000, 0.000, 0.000, 0.000},
    //  #  Inc    Dec     Dec     -       -    -    -    -
       {0.000, 0.000, 0.222, 0.500, 0.000, 0.000, 0.000, 0.000},
    //  #  -     Trim    Dec    Inc    -       -      -      -
       {0.000, 0.000, 0.000, 0.151, 0.830, 0.000, 0.000, 0.000},
    //  #   -      -    Trim   Inc    Inc     -     -     -
       {0.170, 0.000, 0.000, 0.000, 0.000, 0.000, 0.380, 0.000}};
    //  #  -    -      -      -      -       -       -     -

    UInt inputArr2[8] = { 1, 0, 0, 1, 1, 0, 1, 0 };
    UInt activeColumnsArr2[3] = {0, 1, 2};

    for (UInt column = 0; column < numColumns; column++) {
      sp.setPotential(column, potentialArr2[column]);
      sp.setPermanence(column, permanencesArr2[column]);
    }

    activeColumns.assign(&activeColumnsArr2[0], &activeColumnsArr2[3]);

    sp.adaptSynapses_(inputArr2, activeColumns);
    cout << endl;
    for (UInt column = 0; column < numColumns; column++) {
      auto permArr = new Real[numInputs];
      sp.getPermanence(column, permArr);
      ASSERT_TRUE(check_vector_eq(truePermanences2[column], permArr, numInputs));
      delete[] permArr;
    }

  }

  TEST(SpatialPoolerTest, testBumpUpWeakColumns)
  {
    SpatialPooler sp;
    UInt numInputs = 8;
    UInt numColumns = 5;
    setup(sp,numInputs,numColumns);
    sp.setSynPermBelowStimulusInc(0.01);
    sp.setSynPermTrimThreshold(0.05);
    Real overlapDutyCyclesArr[] = {0, 0.009, 0.1, 0.001, 0.002};
    sp.setOverlapDutyCycles(overlapDutyCyclesArr);
    Real minOverlapDutyCyclesArr[] = {0.01, 0.01, 0.01, 0.01, 0.01};
    sp.setMinOverlapDutyCycles(minOverlapDutyCyclesArr);

    UInt potentialArr[5][8] =
      {{1, 1, 1, 1, 0, 0, 0, 0},
       {1, 0, 0, 0, 1, 1, 0, 1},
       {0, 0, 1, 0, 1, 1, 1, 0},
       {1, 1, 1, 0, 0, 0, 1, 0},
       {1, 1, 1, 1, 1, 1, 1, 1}};

    Real permArr[5][8] =
      {{0.200, 0.120, 0.090, 0.040, 0.000, 0.000, 0.000, 0.000},
       {0.150, 0.000, 0.000, 0.000, 0.180, 0.120, 0.000, 0.450},
       {0.000, 0.000, 0.074, 0.000, 0.062, 0.054, 0.110, 0.000},
       {0.051, 0.000, 0.000, 0.000, 0.000, 0.000, 0.178, 0.000},
       {0.100, 0.738, 0.085, 0.002, 0.052, 0.008, 0.208, 0.034}};

    Real truePermArr[5][8] =
      {{0.210, 0.130, 0.100, 0.000, 0.000, 0.000, 0.000, 0.000},
    //  Inc    Inc    Inc    Trim    -     -     -    -
       {0.160, 0.000, 0.000, 0.000, 0.190, 0.130, 0.000, 0.460},
    //  Inc   -     -    -     Inc   Inc    -     Inc
       {0.000, 0.000, 0.074, 0.000, 0.062, 0.054, 0.110, 0.000},  // unchanged
    //  -    -     -    -     -    -     -    -
       {0.061, 0.000, 0.000, 0.000, 0.000, 0.000, 0.188, 0.000},
    //   Inc   Trim    Trim    -     -      -     Inc     -
       {0.110, 0.748, 0.095, 0.000, 0.062, 0.000, 0.218, 0.000}};

    for (UInt i = 0; i < numColumns; i++) {
      sp.setPotential(i, potentialArr[i]);
      sp.setPermanence(i, permArr[i]);
      Real perm[8];
      sp.getPermanence(i, perm);
    }

    sp.bumpUpWeakColumns_();

    for (UInt i = 0; i < numColumns; i++) {
      Real perm[8];
      sp.getPermanence(i, perm);
      ASSERT_TRUE(check_vector_eq(truePermArr[i], perm, numInputs));
    }

  }

  TEST(SpatialPoolerTest, testUpdateDutyCyclesHelper)
  {
    SpatialPooler sp;
    vector<Real> dutyCycles;
    vector<UInt> newValues;
    UInt period;

    dutyCycles.clear();
    newValues.clear();
    Real dutyCyclesArr1[] = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
    Real newValues1[] = {0, 0, 0, 0, 0};
    period = 1000;
    Real trueDutyCycles1[] = {999.0, 999.0, 999.0, 999.0, 999.0};
    dutyCycles.assign(dutyCyclesArr1, dutyCyclesArr1+5);
    newValues.assign(newValues1, newValues1+5);
    sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
    ASSERT_TRUE(check_vector_eq(trueDutyCycles1, dutyCycles));

    dutyCycles.clear();
    newValues.clear();
    Real dutyCyclesArr2[] = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
    Real newValues2[] = {1000, 1000, 1000, 1000, 1000};
    period = 1000;
    Real trueDutyCycles2[] = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
    dutyCycles.assign(dutyCyclesArr2, dutyCyclesArr2+5);
    newValues.assign(newValues2, newValues2+5);
    sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
    ASSERT_TRUE(check_vector_eq(trueDutyCycles2, dutyCycles));

    dutyCycles.clear();
    newValues.clear();
    Real dutyCyclesArr3[] = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
    Real newValues3[] = {2000, 4000, 5000, 6000, 7000};
    period = 1000;
    Real trueDutyCycles3[] = {1001.0, 1003.0, 1004.0, 1005.0, 1006.0};
    dutyCycles.assign(dutyCyclesArr3, dutyCyclesArr3+5);
    newValues.assign(newValues3, newValues3+5);
    sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
    ASSERT_TRUE(check_vector_eq(trueDutyCycles3, dutyCycles));

    dutyCycles.clear();
    newValues.clear();
    Real dutyCyclesArr4[] = {1000.0, 800.0, 600.0, 400.0, 2000.0};
    Real newValues4[] = {0, 0, 0, 0, 0};
    period = 2;
    Real trueDutyCycles4[] = {500.0, 400.0, 300.0, 200.0, 1000.0};
    dutyCycles.assign(dutyCyclesArr4, dutyCyclesArr4+5);
    newValues.assign(newValues4, newValues4+5);
    sp.updateDutyCyclesHelper_(dutyCycles, newValues, period);
    ASSERT_TRUE(check_vector_eq(trueDutyCycles4, dutyCycles));

  }

  TEST(SpatialPoolerTest, testUpdateBoostFactors)
  {
    SpatialPooler sp;
    setup(sp, 5, 6);

    Real initActiveDutyCycles1[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    Real initBoostFactors1[] = {0, 0, 0, 0, 0, 0};
    vector<Real> trueBoostFactors1 =
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    vector<Real> resultBoostFactors1(6, 0);
    sp.setGlobalInhibition(false);
    sp.setBoostStrength(10);
    sp.setBoostFactors(initBoostFactors1);
    sp.setActiveDutyCycles(initActiveDutyCycles1);
    sp.updateBoostFactors_();
    sp.getBoostFactors(resultBoostFactors1.data());
    ASSERT_TRUE(check_vector_eq(trueBoostFactors1, resultBoostFactors1));

    Real initActiveDutyCycles2[] =
      {0.1, 0.3, 0.02, 0.04, 0.7, 0.12};
    Real initBoostFactors2[] =
      {0, 0, 0, 0, 0, 0};
    vector<Real> trueBoostFactors2 =
      {3.10599, 0.42035, 6.91251, 5.65949, 0.00769898, 2.54297};
    vector<Real> resultBoostFactors2(6, 0);
    sp.setGlobalInhibition(false);
    sp.setBoostStrength(10);
    sp.setBoostFactors(initBoostFactors2);
    sp.setActiveDutyCycles(initActiveDutyCycles2);
    sp.updateBoostFactors_();
    sp.getBoostFactors(resultBoostFactors2.data());

    ASSERT_TRUE(check_vector_eq(trueBoostFactors2, resultBoostFactors2));

    Real initActiveDutyCycles3[] =
      {0.1, 0.3, 0.02, 0.04, 0.7, 0.12};
    Real initBoostFactors3[] =
      {0, 0, 0, 0, 0, 0};
    vector<Real> trueBoostFactors3 =
      { 1.25441, 0.840857, 1.47207, 1.41435, 0.377822, 1.20523 };
    vector<Real> resultBoostFactors3(6, 0);
    sp.setWrapAround(true);
    sp.setGlobalInhibition(false);
    sp.setBoostStrength(2.0);
    sp.setInhibitionRadius(5);
    sp.setNumActiveColumnsPerInhArea(1);
    sp.setBoostFactors(initBoostFactors3);
    sp.setActiveDutyCycles(initActiveDutyCycles3);
    sp.updateBoostFactors_();
    sp.getBoostFactors(resultBoostFactors3.data());

    ASSERT_TRUE(check_vector_eq(trueBoostFactors3, resultBoostFactors3));

    Real initActiveDutyCycles4[] =
      {0.1, 0.3, 0.02, 0.04, 0.7, 0.12};
    Real initBoostFactors4[] =
      {0, 0, 0, 0, 0, 0};
    vector<Real> trueBoostFactors4 =
      { 1.94773, 0.263597, 4.33476, 3.549, 0.00482795, 1.59467 };
    vector<Real> resultBoostFactors4(6, 0);
    sp.setGlobalInhibition(true);
    sp.setBoostStrength(10);
    sp.setNumActiveColumnsPerInhArea(1);
    sp.setInhibitionRadius(3);
    sp.setBoostFactors(initBoostFactors4);
    sp.setActiveDutyCycles(initActiveDutyCycles4);
    sp.updateBoostFactors_();
    sp.getBoostFactors(resultBoostFactors4.data());

    ASSERT_TRUE(check_vector_eq(trueBoostFactors3, resultBoostFactors3));
  }

  TEST(SpatialPoolerTest, testUpdateBookeepingVars)
  {
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

  TEST(SpatialPoolerTest, testCalculateOverlap)
  {
    SpatialPooler sp;
    UInt numInputs = 10;
    UInt numColumns = 5;
    UInt numTrials = 5;
    setup(sp,numInputs,numColumns);
    sp.setStimulusThreshold(0);

    Real permArr[5][10] =
      {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
       {0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
       {0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
       {0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};


    UInt inputs[5][10] =
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
       {0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
       {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

    UInt trueOverlaps[5][5] =
      {{ 0,  0,  0,  0,  0},
       {10,  8,  6,  4,  2},
       { 5,  4,  3,  2,  1},
       { 5,  3,  1,  0,  0},
       { 1,  1,  1,  1,  1}};

    for (UInt i = 0; i < numColumns; i++)
    {
      sp.setPermanence(i, permArr[i]);
    }

    for (UInt i = 0; i < numTrials; i++)
    {
      vector<UInt> overlaps;
      sp.calculateOverlap_(inputs[i], overlaps);
      ASSERT_TRUE(check_vector_eq(trueOverlaps[i],overlaps));
    }
  }

  TEST(SpatialPoolerTest, testCalculateOverlapPct)
  {
    SpatialPooler sp;
    UInt numInputs = 10;
    UInt numColumns = 5;
    UInt numTrials = 5;
    setup(sp,numInputs,numColumns);
    sp.setStimulusThreshold(0);

    Real permArr[5][10] =
      {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
       {0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
       {0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
       {0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};

    UInt overlapsArr[5][10] =
     {{ 0,  0,  0,  0,  0},
       {10,  8,  6,  4,  2},
       { 5,  4,  3,  2,  1},
       { 5,  3,  1,  0,  0},
       { 1,  1,  1,  1,  1}};

    Real trueOverlapsPct[5][5] =
      {{0.0, 0.0, 0.0, 0.0, 0.0},
       {1.0, 1.0, 1.0, 1.0, 1.0},
       {0.5, 0.5, 0.5, 0.5, 0.5},
       {0.5, 3.0/8, 1.0/6,  0,  0},
       { 1.0/10,  1.0/8,  1.0/6,  1.0/4,  1.0/2}};

    for (UInt i = 0; i < numColumns; i++)
    {
      sp.setPermanence(i,permArr[i]);
    }

    for (UInt i = 0; i < numTrials; i++)
    {
      vector<Real> overlapsPct;
      vector<UInt> overlaps;
      overlaps.assign(&overlapsArr[i][0],&overlapsArr[i][numColumns]);
      sp.calculateOverlapPct_(overlaps,overlapsPct);
      ASSERT_TRUE(check_vector_eq(trueOverlapsPct[i],overlapsPct));
    }


  }

  TEST(SpatialPoolerTest, testIsWinner)
  {
    UInt numInputs = 10;
    UInt numColumns = 5;
    SpatialPooler sp({numInputs}, {numColumns});

    vector<pair<UInt, Real> > winners;

    UInt numWinners = 3;
    Real score = -5;
    ASSERT_FALSE(sp.isWinner_(score,winners,numWinners));
    score = 0;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));

    pair<UInt, Real> sc1; sc1.first = 1;  sc1.second = 32;
    pair<UInt, Real> sc2; sc2.first = 2;  sc2.second = 27;
    pair<UInt, Real> sc3; sc3.first = 17; sc3.second = 19.5;
    winners.push_back(sc1);
    winners.push_back(sc2);
    winners.push_back(sc3);

    numWinners = 3;
    score = -5;
    ASSERT_TRUE(!sp.isWinner_(score,winners,numWinners));
    score = 18;
    ASSERT_TRUE(!sp.isWinner_(score,winners,numWinners));
    score = 18;
    numWinners = 4;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
    numWinners = 3;
    score = 20;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
    score = 30;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
    score = 40;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
    score = 40;
    numWinners = 6;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));

    pair<UInt, Real> sc4; sc4.first = 34; sc4.second = 17.1;
    pair<UInt, Real> sc5; sc5.first = 51; sc5.second = 1.2;
    pair<UInt, Real> sc6; sc6.first = 19; sc6.second = 0.3;
    winners.push_back(sc4);
    winners.push_back(sc5);
    winners.push_back(sc6);

    score = 40;
    numWinners = 6;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
    score = 12;
    numWinners = 6;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
    score = 0.1;
    numWinners = 6;
    ASSERT_TRUE(!sp.isWinner_(score,winners,numWinners));
    score = 0.1;
    numWinners = 7;
    ASSERT_TRUE(sp.isWinner_(score,winners,numWinners));
  }

  TEST(SpatialPoolerTest, testAddToWinners)
  {
    SpatialPooler sp;
    vector<pair<UInt, Real> > winners;

    UInt index;
    Real score;

    index = 17; score = 19.5;
    sp.addToWinners_(index,score,winners);
    index = 1; score = 32;
    sp.addToWinners_(index,score,winners);
    index = 2; score = 27;
    sp.addToWinners_(index,score,winners);

    ASSERT_TRUE(winners[0].first == 1);
    ASSERT_TRUE(almost_eq(winners[0].second,32));
    ASSERT_TRUE(winners[1].first == 2);
    ASSERT_TRUE(almost_eq(winners[1].second,27));
    ASSERT_TRUE(winners[2].first == 17);
    ASSERT_TRUE(almost_eq(winners[2].second,19.5));

    index = 15; score = 20.5;
    sp.addToWinners_(index,score,winners);
    ASSERT_TRUE(winners[0].first == 1);
    ASSERT_TRUE(almost_eq(winners[0].second,32));
    ASSERT_TRUE(winners[1].first == 2);
    ASSERT_TRUE(almost_eq(winners[1].second,27));
    ASSERT_TRUE(winners[2].first == 15);
    ASSERT_TRUE(almost_eq(winners[2].second,20.5));
    ASSERT_TRUE(winners[3].first == 17);
    ASSERT_TRUE(almost_eq(winners[3].second,19.5));

    index = 7; score = 100;
    sp.addToWinners_(index,score,winners);
    ASSERT_TRUE(winners[0].first == 7);
    ASSERT_TRUE(almost_eq(winners[0].second,100));
    ASSERT_TRUE(winners[1].first == 1);
    ASSERT_TRUE(almost_eq(winners[1].second,32));
    ASSERT_TRUE(winners[2].first == 2);
    ASSERT_TRUE(almost_eq(winners[2].second,27));
    ASSERT_TRUE(winners[3].first == 15);
    ASSERT_TRUE(almost_eq(winners[3].second,20.5));
    ASSERT_TRUE(winners[4].first == 17);
    ASSERT_TRUE(almost_eq(winners[4].second,19.5));

    index = 22; score = 1;
    sp.addToWinners_(index,score,winners);
    ASSERT_TRUE(winners[0].first == 7);
    ASSERT_TRUE(almost_eq(winners[0].second,100));
    ASSERT_TRUE(winners[1].first == 1);
    ASSERT_TRUE(almost_eq(winners[1].second,32));
    ASSERT_TRUE(winners[2].first == 2);
    ASSERT_TRUE(almost_eq(winners[2].second,27));
    ASSERT_TRUE(winners[3].first == 15);
    ASSERT_TRUE(almost_eq(winners[3].second,20.5));
    ASSERT_TRUE(winners[4].first == 17);
    ASSERT_TRUE(almost_eq(winners[4].second,19.5));
    ASSERT_TRUE(winners[5].first == 22);
    ASSERT_TRUE(almost_eq(winners[5].second,1));

  }

  TEST(SpatialPoolerTest, testInhibitColumns)
  {
    SpatialPooler sp;
    setup(sp, 10,10);

    vector<Real> overlapsReal;
    vector<Real> overlaps;
    vector<UInt> activeColumns;
    vector<UInt> activeColumnsGlobal;
    vector<UInt> activeColumnsLocal;
    Real density;
    UInt inhibitionRadius;
    UInt numColumns;

    density = 0.3;
    numColumns = 10;
    Real overlapsArray[10] = {10,21,34,4,18,3,12,5,7,1};

    overlapsReal.assign(&overlapsArray[0],&overlapsArray[numColumns]);
    sp.inhibitColumnsGlobal_(overlapsReal, density,activeColumnsGlobal);
    overlapsReal.assign(&overlapsArray[0],&overlapsArray[numColumns]);
    sp.inhibitColumnsLocal_(overlapsReal, density, activeColumnsLocal);

    sp.setInhibitionRadius(5);
    sp.setGlobalInhibition(true);
    sp.setLocalAreaDensity(density);

    overlaps.assign(&overlapsArray[0],&overlapsArray[numColumns]);
    sp.inhibitColumns_(overlaps, activeColumns);

    ASSERT_TRUE(check_vector_eq(activeColumns, activeColumnsGlobal));
    ASSERT_TRUE(!check_vector_eq(activeColumns, activeColumnsLocal));

    sp.setGlobalInhibition(false);
    sp.setInhibitionRadius(numColumns + 1);

    overlaps.assign(&overlapsArray[0],&overlapsArray[numColumns]);
    sp.inhibitColumns_(overlaps, activeColumns);

    ASSERT_TRUE(check_vector_eq(activeColumns, activeColumnsGlobal));
    ASSERT_TRUE(!check_vector_eq(activeColumns, activeColumnsLocal));

    inhibitionRadius = 2;
    density = 2.0 / 5;

    sp.setInhibitionRadius(inhibitionRadius);
    sp.setNumActiveColumnsPerInhArea(2);

    overlapsReal.assign(&overlapsArray[0], &overlapsArray[numColumns]);
    sp.inhibitColumnsGlobal_(overlapsReal, density,activeColumnsGlobal);
    overlapsReal.assign(&overlapsArray[0], &overlapsArray[numColumns]);
    sp.inhibitColumnsLocal_(overlapsReal, density, activeColumnsLocal);

    overlaps.assign(&overlapsArray[0],&overlapsArray[numColumns]);
    sp.inhibitColumns_(overlaps, activeColumns);

    ASSERT_TRUE(!check_vector_eq(activeColumns, activeColumnsGlobal));
    ASSERT_TRUE(check_vector_eq(activeColumns, activeColumnsLocal));
  }

  TEST(SpatialPoolerTest, testInhibitColumnsGlobal)
  {
    SpatialPooler sp;
    UInt numInputs = 10;
    UInt numColumns = 10;
    setup(sp,numInputs,numColumns);
    vector<Real> overlaps;
    vector<UInt> activeColumns;
    vector<UInt> trueActive;
    vector<UInt> active;
    Real density;

    density = 0.3;
    Real overlapsArray[10] = {1,2,1,4,8,3,12,5,4,1};
    overlaps.assign(&overlapsArray[0],&overlapsArray[numColumns]);
    sp.inhibitColumnsGlobal_(overlaps,density,activeColumns);
    UInt trueActiveArray1[3] = {4,6,7};

    trueActive.assign(numColumns, 0);
    active.assign(numColumns, 0);

    for (auto & elem : trueActiveArray1) {
      trueActive[elem] = 1;
    }

    for (auto & activeColumn : activeColumns) {
      active[activeColumn] = 1;
    }

    ASSERT_TRUE(check_vector_eq(trueActive,active));


    density = 0.5;
    UInt overlapsArray2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    overlaps.assign(&overlapsArray2[0],&overlapsArray2[numColumns]);
    sp.inhibitColumnsGlobal_(overlaps, density, activeColumns);
    UInt trueActiveArray2[5] = {5,6,7,8,9};

    for (auto & elem : trueActiveArray2) {
      trueActive[elem] = 1;
    }

    for (auto & activeColumn : activeColumns) {
      active[activeColumn] = 1;
    }

    ASSERT_TRUE(check_vector_eq(trueActive,active));
  }

  TEST(SpatialPoolerTest, testInhibitColumnsLocal)
  {
    // wrapAround = false
    {
      SpatialPooler sp(
        /*inputDimensions*/{10},
        /*columnDimensions*/ {10},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ -1.0,
        /*numActiveColumnsPerInhArea*/ 3,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008,
        /*synPermActiveInc*/ 0.05,
        /*synPermConnected*/ 0.1,
        /*minPctOverlapDutyCycles*/ 0.001,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 10.0,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ false);

      Real density;
      UInt inhibitionRadius;

      vector<Real> overlaps;
      vector<UInt> active;

      Real overlapsArray1[10] = { 1, 2, 7, 0, 3, 4, 16, 1, 1.5, 1.7};
                              //  L  W  W  L  L  W  W   L   L    W

      inhibitionRadius = 2;
      density = 0.5;
      overlaps.assign(&overlapsArray1[0], &overlapsArray1[10]);
      UInt trueActive[5] = {1, 2, 5, 6, 9};
      sp.setInhibitionRadius(inhibitionRadius);
      sp.inhibitColumnsLocal_(overlaps, density, active);
      ASSERT_EQ(5, active.size());
      ASSERT_TRUE(check_vector_eq(trueActive, active));

      Real overlapsArray2[10] = {1, 2, 7, 0, 3, 4, 16, 1, 1.5, 1.7};
                            //   L  W  W  L  L  W   W  L   L    W
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
                              // W  L  W  L  W  L  W  L  L  L
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
        /*inputDimensions*/{10},
        /*columnDimensions*/ {10},
        /*potentialRadius*/ 16,
        /*potentialPct*/ 0.5,
        /*globalInhibition*/ false,
        /*localAreaDensity*/ -1.0,
        /*numActiveColumnsPerInhArea*/ 3,
        /*stimulusThreshold*/ 1,
        /*synPermInactiveDec*/ 0.008,
        /*synPermActiveInc*/ 0.05,
        /*synPermConnected*/ 0.1,
        /*minPctOverlapDutyCycles*/ 0.001,
        /*dutyCyclePeriod*/ 1000,
        /*boostStrength*/ 10.0,
        /*seed*/ 1,
        /*spVerbosity*/ 0,
        /*wrapAround*/ true);

      Real density;
      UInt inhibitionRadius;

      vector<Real> overlaps;
      vector<UInt> active;

      Real overlapsArray1[10] = { 1, 2, 7, 0, 3, 4, 16, 1, 1.5, 1.7};
                              //  L  W  W  L  L  W  W   L   W    W

      inhibitionRadius = 2;
      density = 0.5;
      overlaps.assign(&overlapsArray1[0], &overlapsArray1[10]);
      UInt trueActive[6] = {1, 2, 5, 6, 8, 9};
      sp.setInhibitionRadius(inhibitionRadius);
      sp.inhibitColumnsLocal_(overlaps, density, active);
      ASSERT_EQ(6, active.size());
      ASSERT_TRUE(check_vector_eq(trueActive, active));

      Real overlapsArray2[10] = {1, 2, 7, 0, 3, 4, 16, 1, 1.5, 1.7};
                            //   L  W  W  L  W  W   W  L   L    W
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
                              // W  W  L  L  W  W  L  L  L  W
      overlaps.assign(&overlapsArray3[0], &overlapsArray3[10]);
      UInt trueActive3[4] = {0, 1, 4, 5};
      inhibitionRadius = 3;
      density = 0.25;
      sp.setInhibitionRadius(inhibitionRadius);
      sp.inhibitColumnsLocal_(overlaps, density, active);

      ASSERT_TRUE(active.size() == 4);
      ASSERT_TRUE(check_vector_eq(trueActive3, active));
    }
  }

  TEST(SpatialPoolerTest, testIsUpdateRound)
  {
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

  TEST(SpatialPoolerTest, testRaisePermanencesToThreshold)
  {
    SpatialPooler sp;
    UInt stimulusThreshold = 3;
    Real synPermConnected = 0.1;
    Real synPermBelowStimulusInc = 0.01;
    UInt numInputs = 5;
    UInt numColumns = 7;
    setup(sp,numInputs,numColumns);
    sp.setStimulusThreshold(stimulusThreshold);
    sp.setSynPermConnected(synPermConnected);
    sp.setSynPermBelowStimulusInc(synPermBelowStimulusInc);

    UInt potentialArr[7][5] =
      {{ 1, 1, 1, 1, 1 },
       { 1, 1, 1, 1, 1 },
       { 1, 1, 1, 1, 1 },
       { 1, 1, 1, 1, 1 },
       { 1, 1, 1, 1, 1 },
       { 1, 1, 0, 0, 1 },
       { 0, 1, 1, 1, 0 }};


    Real permArr[7][5] =
      {{ 0.0,   0.11,   0.095, 0.092, 0.01  },
       { 0.12,  0.15,   0.02,  0.12,  0.09  },
       { 0.51,  0.081,  0.025, 0.089, 0.31  },
       { 0.18,  0.0601, 0.11,  0.011, 0.03  },
       { 0.011, 0.011,  0.011, 0.011, 0.011 },
       { 0.12,  0.056,  0,     0,     0.078 },
       { 0,     0.061,   0.07,   0.14,  0   }};

    Real truePerm[7][5] =
      {{  0.01, 0.12, 0.105, 0.102, 0.02      },  // incremented once
       {  0.12, 0.15, 0.02, 0.12, 0.09      },  // no change
       {  0.53, 0.101, 0.045, 0.109, 0.33     },  // increment twice
       {  0.22, 0.1001, 0.15, 0.051, 0.07   },  // increment four times
       {  0.101, 0.101, 0.101, 0.101, 0.101 },  // increment 9 times
       {  0.17,  0.106, 0,     0,     0.128 },  // increment 5 times
       {  0,     0.101, 0.11,    0.18,  0     }}; // increment 4 times


    UInt trueConnectedCount[7] =
      {3, 3, 4, 3, 5, 3, 3};

    for (UInt i = 0; i < numColumns; i++)
    {
      vector<Real> perm;
      vector<UInt> potential;
      perm.assign(&permArr[i][0],&permArr[i][numInputs]);
      for (UInt j = 0; j < numInputs; j++) {
        if (potentialArr[i][j] > 0) {
          potential.push_back(j);
        }
      }
      UInt connected =
        sp.raisePermanencesToThreshold_(perm, potential);
      ASSERT_TRUE(check_vector_eq(truePerm[i],perm));
      ASSERT_TRUE(connected == trueConnectedCount[i]);
    }

  }

  TEST(SpatialPoolerTest, testUpdatePermanencesForColumn)
  {
    vector<UInt> inputDim;
    vector<UInt> columnDim;

    UInt numInputs = 5;
    UInt numColumns = 5;
    SpatialPooler sp;
    setup(sp,numInputs,numColumns);
    Real synPermTrimThreshold = 0.05;
    sp.setSynPermTrimThreshold(synPermTrimThreshold);

    Real permArr[5][5] =
      {{ -0.10, 0.500, 0.400, 0.010, 0.020 },
       { 0.300, 0.010, 0.020, 0.120, 0.090 },
       { 0.070, 0.050, 1.030, 0.190, 0.060 },
       { 0.180, 0.090, 0.110, 0.010, 0.030 },
       { 0.200, 0.101, 0.050, -0.09, 1.100 }};

    Real truePerm[5][5] =
       {{ 0.000, 0.500, 0.400, 0.000, 0.000},
        // Clip     -     -      Trim   Trim
        {0.300, 0.000, 0.000, 0.120, 0.090},
         // -    Trim   Trim   -     -
        {0.070, 0.050, 1.000, 0.190, 0.060},
        // -     -   Clip   -     -
        {0.180, 0.090, 0.110, 0.000, 0.000},
        // -     -    -      Trim   Trim
        {0.200, 0.101, 0.050, 0.000, 1.000}};
        // -      -     -      Clip   Clip

    UInt trueConnectedSynapses[5][5] =
      {{0, 1, 1, 0, 0},
       {1, 0, 0, 1, 0},
       {0, 0, 1, 1, 0},
       {1, 0, 1, 0, 0},
       {1, 1, 0, 0, 1 }};

    UInt trueConnectedCount[5] = {2, 2, 2, 2, 3};

    for (UInt i = 0; i < 5; i ++)
    {
      vector<Real> perm(&permArr[i][0], &permArr[i][5]);
      sp.updatePermanencesForColumn_(perm, i, false);
      auto permArr = new Real[numInputs];
      auto connectedArr = new UInt[numInputs];
      auto connectedCountsArr = new UInt[numColumns];
      sp.getPermanence(i, permArr);
      sp.getConnectedSynapses(i, connectedArr);
      sp.getConnectedCounts(connectedCountsArr);
      ASSERT_TRUE(check_vector_eq(truePerm[i], permArr, numInputs));
      ASSERT_TRUE(check_vector_eq(trueConnectedSynapses[i],connectedArr,
                                numInputs));
      ASSERT_TRUE(trueConnectedCount[i] == connectedCountsArr[i]);
      delete[] permArr;
      delete[] connectedArr;
      delete[] connectedCountsArr;
    }

  }

  TEST(SpatialPoolerTest, testInitPermanence)
  {
    vector<UInt> inputDim;
    vector<UInt> columnDim;
    inputDim.push_back(8);
    columnDim.push_back(2);

    SpatialPooler sp;
    Real synPermConnected = 0.2;
    Real synPermTrimThreshold = 0.1;
    Real synPermActiveInc = 0.05;
    sp.initialize(inputDim,columnDim);
    sp.setSynPermConnected(synPermConnected);
    sp.setSynPermTrimThreshold(synPermTrimThreshold);
    sp.setSynPermActiveInc(synPermActiveInc);

    UInt arr[8] = { 0, 1, 1 , 0, 0, 1, 0, 1 };
    vector<UInt> potential(&arr[0], &arr[8]);
    vector<Real> perm = sp.initPermanence_(potential, 1.0);
    for (UInt i = 0; i < 8; i++)
      if (potential[i])
        ASSERT_TRUE(perm[i] >= synPermConnected);
      else
        ASSERT_TRUE(perm[i] < 1e-5);

    perm = sp.initPermanence_(potential, 0);
    for (UInt i = 0; i < 8; i++)
      if (potential[i])
        ASSERT_LE(perm[i], synPermConnected);
      else        
        ASSERT_LT(perm[i], 1e-5);

    inputDim[0] = 100;
    sp.initialize(inputDim,columnDim);
    sp.setSynPermConnected(synPermConnected);
    sp.setSynPermTrimThreshold(synPermTrimThreshold);
    sp.setSynPermActiveInc(synPermActiveInc);
    potential.clear();

    for(UInt i = 0; i < 100; i++)
      potential.push_back(1);

    perm = sp.initPermanence_(potential, 0.5);
    int count = 0;
    for (UInt i = 0; i < 100; i++)
    {
      ASSERT_TRUE(perm[i] < 1e-5 || perm[i] >= synPermTrimThreshold);
      if (perm[i] >= synPermConnected)
        count++;
    }
    ASSERT_TRUE(count > 5 && count < 95);
  }

  TEST(SpatialPoolerTest, testInitPermConnected)
  {
    SpatialPooler sp;
    Real synPermConnected = 0.2;
    Real synPermMax = 1.0;

    sp.setSynPermConnected(synPermConnected);
    sp.setSynPermMax(synPermMax);

    for (UInt i = 0; i < 100; i++) {
      Real permVal = sp.initPermConnected_();
      ASSERT_GE(permVal, synPermConnected);
      ASSERT_LE(permVal, synPermMax);
    }
  }

  TEST(SpatialPoolerTest, testInitPermNonConnected)
  {
    SpatialPooler sp;
    Real synPermConnected = 0.2;
    sp.setSynPermConnected(synPermConnected);
    for (UInt i = 0; i < 100; i++) {
      Real permVal = sp.initPermNonConnected_();
      ASSERT_GE(permVal, 0);
      ASSERT_LE(permVal, synPermConnected);
    }
  }

  TEST(SpatialPoolerTest, testMapColumn)
  {
    {
      // Test 1D.
      SpatialPooler sp(
        /*inputDimensions*/{12},
        /*columnDimensions*/{4});

      EXPECT_EQ(1, sp.mapColumn_(0));
      EXPECT_EQ(4, sp.mapColumn_(1));
      EXPECT_EQ(7, sp.mapColumn_(2));
      EXPECT_EQ(10, sp.mapColumn_(3));
    }

    {
      // Test 1D with same dimensions of columns and inputs.
      SpatialPooler sp(
        /*inputDimensions*/{4},
        /*columnDimensions*/{4});

      EXPECT_EQ(0, sp.mapColumn_(0));
      EXPECT_EQ(1, sp.mapColumn_(1));
      EXPECT_EQ(2, sp.mapColumn_(2));
      EXPECT_EQ(3, sp.mapColumn_(3));
    }

    {
      // Test 1D with dimensions of length 1.
      SpatialPooler sp(
        /*inputDimensions*/{1},
        /*columnDimensions*/{1});

      EXPECT_EQ(0, sp.mapColumn_(0));
    }

    {
      // Test 2D.
      SpatialPooler sp(
        /*inputDimensions*/{36, 12},
        /*columnDimensions*/{12, 4});

      EXPECT_EQ(13, sp.mapColumn_(0));
      EXPECT_EQ(49, sp.mapColumn_(4));
      EXPECT_EQ(52, sp.mapColumn_(5));
      EXPECT_EQ(58, sp.mapColumn_(7));
      EXPECT_EQ(418, sp.mapColumn_(47));
    }

    {
      // Test 2D, some input dimensions smaller than column dimensions.
      SpatialPooler sp(
        /*inputDimensions*/{3, 5},
        /*columnDimensions*/{4, 4});

      EXPECT_EQ(0, sp.mapColumn_(0));
      EXPECT_EQ(4, sp.mapColumn_(3));
      EXPECT_EQ(14, sp.mapColumn_(15));
    }
  }

  TEST(SpatialPoolerTest, testMapPotential1D)
  {
    vector<UInt> inputDim, columnDim;
    inputDim.push_back(12);
    columnDim.push_back(4);
    UInt potentialRadius = 2;

    SpatialPooler sp;
    sp.initialize(inputDim, columnDim);
    sp.setPotentialRadius(potentialRadius);

    vector<UInt> mask;

    // Test without wrapAround and potentialPct = 1
    sp.setPotentialPct(1.0);

    UInt expectedMask1[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    mask = sp.mapPotential_(0, false);
    ASSERT_TRUE(check_vector_eq(expectedMask1, mask));

    UInt expectedMask2[12] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0};
    mask = sp.mapPotential_(2, false);
    ASSERT_TRUE(check_vector_eq(expectedMask2, mask));

    // Test with wrapAround and potentialPct = 1
    sp.setPotentialPct(1.0);

    UInt expectedMask3[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
    mask = sp.mapPotential_(0, true);
    ASSERT_TRUE(check_vector_eq(expectedMask3, mask));

    UInt expectedMask4[12] = {1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    mask = sp.mapPotential_(3, true);
    ASSERT_TRUE(check_vector_eq(expectedMask4, mask));

    // Test with potentialPct < 1
    sp.setPotentialPct(0.5);
    UInt supersetMask1[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
    mask = sp.mapPotential_(0, true);
    ASSERT_TRUE(sum(mask) == 3);

    UInt unionMask1[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (UInt i = 0; i < 12; i++) {
      unionMask1[i] = supersetMask1[i] | mask.at(i);
    }

    ASSERT_TRUE(check_vector_eq(unionMask1, supersetMask1, 12));
  }

  TEST(SpatialPoolerTest, testMapPotential2D)
  {
    vector<UInt> inputDim, columnDim;
    inputDim.push_back(6);
    inputDim.push_back(12);
    columnDim.push_back(2);
    columnDim.push_back(4);
    UInt potentialRadius = 1;
    Real potentialPct = 1.0;

    SpatialPooler sp;
    sp.initialize(inputDim, columnDim);
    sp.setPotentialRadius(potentialRadius);
    sp.setPotentialPct(potentialPct);

    vector<UInt> mask;

    // Test without wrapAround
    UInt expectedMask1[72] = {
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    mask = sp.mapPotential_(0, false);
    ASSERT_TRUE(check_vector_eq(expectedMask1, mask));

    UInt expectedMask2[72] = {
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    mask = sp.mapPotential_(2, false);
    ASSERT_TRUE(check_vector_eq(expectedMask2, mask));

    // Test with wrapAround
    potentialRadius = 2;
    sp.setPotentialRadius(potentialRadius);
    UInt expectedMask3[72] = {
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1
    };
    mask = sp.mapPotential_(0, true);
    ASSERT_TRUE(check_vector_eq(expectedMask3, mask));

    UInt expectedMask4[72] = {
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1
    };
    mask = sp.mapPotential_(3, true);
    ASSERT_TRUE(check_vector_eq(expectedMask4, mask));
  }

  TEST(SpatialPoolerTest, testStripUnlearnedColumns)
  {
    SpatialPooler sp;
    vector<UInt> inputDim, columnDim;
    inputDim.push_back(5);
    columnDim.push_back(3);
    sp.initialize(inputDim, columnDim);

    // None learned, none active
    {
      Real activeDutyCycles[3] = {0, 0, 0};
      UInt activeArray[3] = {0, 0, 0};
      UInt expected[3] = {0, 0, 0};

      sp.setActiveDutyCycles(activeDutyCycles);
      sp.stripUnlearnedColumns(activeArray);

      ASSERT_TRUE(check_vector_eq(activeArray, expected, 3));
    }

    // None learned, some active
    {
      Real activeDutyCycles[3] = {0, 0, 0};
      UInt activeArray[3] = {1, 0, 1};
      UInt expected[3] = {0, 0, 0};

      sp.setActiveDutyCycles(activeDutyCycles);
      sp.stripUnlearnedColumns(activeArray);

      ASSERT_TRUE(check_vector_eq(activeArray, expected, 3));
    }

    // Some learned, none active
    {
      Real activeDutyCycles[3] = {1, 1, 0};
      UInt activeArray[3] = {0, 0, 0};
      UInt expected[3] = {0, 0, 0};

      sp.setActiveDutyCycles(activeDutyCycles);
      sp.stripUnlearnedColumns(activeArray);

      ASSERT_TRUE(check_vector_eq(activeArray, expected, 3));
    }

    // Some learned, some active
    {
      Real activeDutyCycles[3] = {1, 1, 0};
      UInt activeArray[3] = {1, 0, 1};
      UInt expected[3] = {1, 0, 0};

      sp.setActiveDutyCycles(activeDutyCycles);
      sp.stripUnlearnedColumns(activeArray);

      ASSERT_TRUE(check_vector_eq(activeArray, expected, 3));
    }
  }

  TEST(SpatialPoolerTest, getOverlaps)
  {
    SpatialPooler sp;
    const vector<UInt> inputDim = {5};
    const vector<UInt> columnDim = {3};
    sp.initialize(inputDim, columnDim);

    UInt potential[5] = {1, 1, 1, 1, 1};
    sp.setPotential(0, potential);
    sp.setPotential(1, potential);
    sp.setPotential(2, potential);

    Real permanence0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    sp.setPermanence(0, permanence0);
    Real permanence1[5] = {1.0, 1.0, 1.0, 0.0, 0.0};
    sp.setPermanence(1, permanence1);
    Real permanence2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    sp.setPermanence(2, permanence2);

    vector<Real> boostFactors = {1.0, 2.0, 3.0};
    sp.setBoostFactors(boostFactors.data());

    vector<UInt> input = {1, 1, 1, 1, 1};
    vector<UInt> activeColumns = {0, 0, 0};
    sp.compute(input.data(), true, activeColumns.data());

    const vector<UInt>& overlaps = sp.getOverlaps();
    const vector<UInt> expectedOverlaps = {0, 3, 5};
    EXPECT_EQ(expectedOverlaps, overlaps);

    const vector<Real>& boostedOverlaps = sp.getBoostedOverlaps();
    const vector<Real> expectedBoostedOverlaps = {0.0, 6.0, 15.0};
    EXPECT_EQ(expectedBoostedOverlaps, boostedOverlaps);
  }

  TEST(SpatialPoolerTest, ZeroOverlap_NoStimulusThreshold_GlobalInhibition)
  {
    const UInt inputSize = 10;
    const UInt nColumns = 20;

    SpatialPooler sp({inputSize},
                     {nColumns},
                     /*potentialRadius*/ 10,
                     /*potentialPct*/ 0.5,
                     /*globalInhibition*/ true,
                     /*localAreaDensity*/ -1.0,
                     /*numActiveColumnsPerInhArea*/ 3,
                     /*stimulusThreshold*/ 0,
                     /*synPermInactiveDec*/ 0.008,
                     /*synPermActiveInc*/ 0.05,
                     /*synPermConnected*/ 0.1,
                     /*minPctOverlapDutyCycles*/ 0.001,
                     /*dutyCyclePeriod*/ 1000,
                     /*boostStrength*/ 10.0,
                     /*seed*/ 1,
                     /*spVerbosity*/ 0,
                     /*wrapAround*/ true);

    vector<UInt> input(inputSize, 0);
    vector<UInt> activeColumns(nColumns, 0);
    sp.compute(input.data(), true, activeColumns.data());

    EXPECT_EQ(3, countNonzero(activeColumns));
  }

  TEST(SpatialPoolerTest, ZeroOverlap_StimulusThreshold_GlobalInhibition)
  {
    const UInt inputSize = 10;
    const UInt nColumns = 20;

    SpatialPooler sp({inputSize},
                     {nColumns},
                     /*potentialRadius*/ 5,
                     /*potentialPct*/ 0.5,
                     /*globalInhibition*/ true,
                     /*localAreaDensity*/ -1.0,
                     /*numActiveColumnsPerInhArea*/ 1,
                     /*stimulusThreshold*/ 1,
                     /*synPermInactiveDec*/ 0.008,
                     /*synPermActiveInc*/ 0.05,
                     /*synPermConnected*/ 0.1,
                     /*minPctOverlapDutyCycles*/ 0.001,
                     /*dutyCyclePeriod*/ 1000,
                     /*boostStrength*/ 10.0,
                     /*seed*/ 1,
                     /*spVerbosity*/ 0,
                     /*wrapAround*/ true);

    vector<UInt> input(inputSize, 0);
    vector<UInt> activeColumns(nColumns, 0);
    sp.compute(input.data(), true, activeColumns.data());

    EXPECT_EQ(0, countNonzero(activeColumns));
  }

  TEST(SpatialPoolerTest, ZeroOverlap_NoStimulusThreshold_LocalInhibition)
  {
    const UInt inputSize = 10;
    const UInt nColumns = 20;

    SpatialPooler sp({inputSize},
                     {nColumns},
                     /*potentialRadius*/ 5,
                     /*potentialPct*/ 0.5,
                     /*globalInhibition*/ false,
                     /*localAreaDensity*/ -1.0,
                     /*numActiveColumnsPerInhArea*/ 1,
                     /*stimulusThreshold*/ 0,
                     /*synPermInactiveDec*/ 0.008,
                     /*synPermActiveInc*/ 0.05,
                     /*synPermConnected*/ 0.1,
                     /*minPctOverlapDutyCycles*/ 0.001,
                     /*dutyCyclePeriod*/ 1000,
                     /*boostStrength*/ 10.0,
                     /*seed*/ 1,
                     /*spVerbosity*/ 0,
                     /*wrapAround*/ true);

    vector<UInt> input(inputSize, 0);
    vector<UInt> activeColumns(nColumns, 0);
    sp.compute(input.data(), true, activeColumns.data());

    // This exact number of active columns is determined by the inhibition
    // radius, which changes based on the random synapses (i.e. weird math).
    EXPECT_GT(countNonzero(activeColumns), 2);
    EXPECT_LT(countNonzero(activeColumns), 10);
  }

  TEST(SpatialPoolerTest, ZeroOverlap_StimulusThreshold_LocalInhibition)
  {
    const UInt inputSize = 10;
    const UInt nColumns = 20;

    SpatialPooler sp({inputSize},
                     {nColumns},
                     /*potentialRadius*/ 10,
                     /*potentialPct*/ 0.5,
                     /*globalInhibition*/ false,
                     /*localAreaDensity*/ -1.0,
                     /*numActiveColumnsPerInhArea*/ 3,
                     /*stimulusThreshold*/ 1,
                     /*synPermInactiveDec*/ 0.008,
                     /*synPermActiveInc*/ 0.05,
                     /*synPermConnected*/ 0.1,
                     /*minPctOverlapDutyCycles*/ 0.001,
                     /*dutyCyclePeriod*/ 1000,
                     /*boostStrength*/ 10.0,
                     /*seed*/ 1,
                     /*spVerbosity*/ 0,
                     /*wrapAround*/ true);

    vector<UInt> input(inputSize, 0);
    vector<UInt> activeColumns(nColumns, 0);
    sp.compute(input.data(), true, activeColumns.data());

    EXPECT_EQ(0, countNonzero(activeColumns));
  }

  TEST(SpatialPoolerTest, testSaveLoad)
  {
    const char* filename = "SpatialPoolerSerialization.tmp";
    SpatialPooler sp1, sp2;
    UInt numInputs = 6;
    UInt numColumns = 12;
    setup(sp1, numInputs, numColumns);

    ofstream outfile;
    outfile.open(filename);
    sp1.save(outfile);
    outfile.close();

    ifstream infile (filename);
    sp2.load(infile);
    infile.close();

    ASSERT_NO_FATAL_FAILURE(
      check_spatial_eq(sp1, sp2));

    int ret = ::remove(filename);
    ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
  }

  TEST(SpatialPoolerTest, testWriteRead)
  {
    const char* filename = "SpatialPoolerSerialization.tmp";
    SpatialPooler sp1, sp2;
    UInt numInputs = 6;
    UInt numColumns = 12;
    setup(sp1, numInputs, numColumns);

    ofstream os(filename, ios::binary);
    sp1.write(os);
    os.close();

    ifstream is(filename, ios::binary);
    sp2.read(is);
    is.close();

    ASSERT_NO_FATAL_FAILURE(
      check_spatial_eq(sp1, sp2));

    int ret = ::remove(filename);
    ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
  }

} // end anonymous namespace
