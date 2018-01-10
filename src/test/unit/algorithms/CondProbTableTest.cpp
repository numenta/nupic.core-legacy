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
* Notes
*/

#include <nupic/algorithms/CondProbTable.hpp>
#include <nupic/math/StlIo.hpp>
#include <fstream>
#include <boost/serialization/array_wrapper.hpp> //FIXME this include is fix for the include below in boost .64, remove later when fixed upstream
#include <boost/numeric/ublas/storage.hpp>
#include "gtest/gtest.h"


using namespace std;
using namespace boost;
using namespace nupic;


namespace {
  
  // Size of the table we construct
  Size numRows() {return 4;}
  Size numCols() {return 3;}

  static vector<Real> makeRow(Real a, Real b, Real c)
  { 
    vector<Real> result(3);
    result[0] = a;
    result[1] = b;
    result[2] = c;
  
    return result;
  }
  
  static vector<Real> makeCol(Real a, Real b, Real c, Real d)
  { 
    vector<Real> result(4);
    result[0] = a;
    result[1] = b;
    result[2] = c;
    result[3] = d;
  
    return result;
  }

  void testVectors(const string& testName, const vector<Real>& v1,
                   const vector<Real>& v2)
  {
    stringstream s1, s2;
    s1 << v1;
    s2 << v2;
    EXPECT_EQ(s1.str(), s2.str());
  }

  void testTable(const string& testName, CondProbTable& table, 
                 const vector<vector<Real> > & rows)
  {
  
    // Test the numRows(), numCols() calls
    ASSERT_EQ(numRows(), table.numRows());
    ASSERT_EQ(numCols(), table.numColumns());

    // See if they got added right
    vector<Real>  testRow(numCols());
    for (Size i=0; i<numRows(); i++) {
      stringstream ss;
      ss << "updateRow " << i;
    
      table.getRow((UInt)i, testRow);
      ASSERT_NO_FATAL_FAILURE(
        testVectors(testName+ss.str(), rows[i], testRow));
    }


    // --------------------------------------------------------------------
    // Try out normal inference
    vector<Real> expValue;
    vector<Real> output(numRows());
  
    // Row 0 matches row 3, so we get half and half hits on those rows
    table.inferRow (rows[0], output, CondProbTable::inferMarginal);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 0 infer", makeCol((Real).16, (Real)0, (Real)0, (Real).24), output));
  
    // Row 1 matches only row 1
    table.inferRow (rows[1], output, CondProbTable::inferMarginal);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 1 infer", makeCol((Real)0, 1, (Real)0, (Real)0), output));

    // Row 2 matches only row 2 and 3
    table.inferRow (rows[2], output, CondProbTable::inferMarginal);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 2 infer", makeCol((Real)0, (Real)0, (Real).36, (Real).24), output));

    // Row 3 matches row 0 & row 2 halfway, and row 3 exactly
    table.inferRow (rows[3], output, CondProbTable::inferMarginal);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 3 infer", makeCol((Real).24, (Real)0, (Real).24, (Real).52), output));
  
  
    // --------------------------------------------------------------------
    // Try out inferEvidence inference
  
    // Row 0 matches row 0 and half row 3, so we get half and half hits on those rows
    table.inferRow (rows[0], output, CondProbTable::inferRowEvidence);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 0 inferEvidence", makeCol((Real).4, (Real)0, (Real)0, (Real).24), output));
  
    // Row 1 matches only row 1
    table.inferRow (rows[1], output, CondProbTable::inferRowEvidence);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 1 inferEvidence", makeCol((Real)0, 1, (Real)0, (Real)0), output));

    // Row 2 matches only row 2 and half row 3
    table.inferRow (rows[2], output, CondProbTable::inferRowEvidence);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 2 inferEvidence", makeCol((Real)0, (Real)0, (Real).6, (Real).24), output));

    // Row 3 matches row 0 & row 2 halfway, and row 3 exactly
    table.inferRow (rows[3], output, CondProbTable::inferRowEvidence);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 3 inferEvidence", makeCol((Real).6, (Real)0, (Real).4, (Real).52), output));
  
  
    // --------------------------------------------------------------------
    // Try out inferMaxProd inference
  
    // Row 0 matches row 0 and half row 3, so we get half and half hits on those rows
    table.inferRow (rows[0], output, CondProbTable::inferMaxProd);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 0 inferMaxProd", makeCol((Real).16, (Real)0, (Real)0, (Real).24), output));
  
    // Row 1 matches only row 1
    table.inferRow (rows[1], output, CondProbTable::inferMaxProd);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 1 inferMaxProd", makeCol((Real)0, 1, (Real)0, (Real)0), output));

    // Row 2 matches only row 2 and half row 3
    table.inferRow (rows[2], output, CondProbTable::inferMaxProd);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 2 inferMaxProd", makeCol((Real)0, (Real)0, (Real).36, (Real).24), output));

    // Row 3 matches row 0 & row 2 halfway, and row 3 exactly
    table.inferRow (rows[3], output, CondProbTable::inferMaxProd);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 3 inferMaxProd", makeCol((Real).24, (Real)0, (Real).24, (Real).36), output));
  
  
    // --------------------------------------------------------------------
    // Try out inferViterbi inference
  
    // Row 0 matches row 0 and half row 3, so we get half and half hits on those rows
    table.inferRow (rows[0], output, CondProbTable::inferViterbi);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 0 inferViterbi", makeCol((Real)0, (Real)0, (Real)0, (Real).4), output));
  
    // Row 1 matches only row 1
    table.inferRow (rows[1], output, CondProbTable::inferViterbi);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 1 inferViterbi", makeCol((Real)0, 1, (Real)0, (Real)0), output));

    // Row 2 matches only row 2 and half row 3
    table.inferRow (rows[2], output, CondProbTable::inferViterbi);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 2 inferViterbi", makeCol((Real)0, (Real)0, (Real).6, (Real)0), output));

    // Row 3 matches row 0 & row 2 halfway, and row 3 exactly
    table.inferRow (rows[3], output, CondProbTable::inferViterbi);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 3 inferViterbi", makeCol((Real)0, (Real)0, (Real).4, (Real).6), output));
  
  
    // Add a row a second time, the row should double in value
    table.updateRow(0, rows[0]);
    expValue = rows[0];
    for (Size i=0; i<numCols(); i++)
      expValue[i] *= 2;
    table.getRow(0, testRow);
    ASSERT_NO_FATAL_FAILURE(
      testVectors(testName+"row 0 update#2", expValue, testRow));
  
  }

  //----------------------------------------------------------------------
  TEST(CondProbTableTest, Basic)
  {
    // Our 4 rows
    vector<vector<Real> > rows;
    rows.resize(numRows());
    rows[0] = makeRow((Real)0.0, (Real)0.4, (Real)0.0);
    rows[1] = makeRow((Real)1.0, (Real)0.0, (Real)0.0);
    rows[2] = makeRow((Real)0.0, (Real)0.0, (Real)0.6);
    rows[3] = makeRow((Real)0.0, (Real)0.6, (Real)0.4);

    // Test constructing without # of columns
    {
      CondProbTable table;
      
      // Add the 4 rows
      for (Size i=0; i<numRows(); i++)
        table.updateRow((UInt)i, rows[i]);
      
      // Test it
      ASSERT_NO_FATAL_FAILURE(
        testTable ("Dynamic columns:", table, rows));
    }


    // Test constructing and growing the columns dynamically
    {
      CondProbTable table;
      
      // Add the 2nd row first which has just 1 column
      vector<Real> row1(1);
      row1[0] = rows[1][0];
      table.updateRow(1, row1);
    
      // Add the first row first with just 2 columns
      vector<Real> row0(2);
      row0[0] = rows[0][0];
      row0[1] = rows[0][1];
      table.updateRow(0, row0);
    
      for (Size i=2; i<numRows(); i++)
        table.updateRow((UInt)i, rows[i]);
      
      // Test it
      ASSERT_NO_FATAL_FAILURE(
        testTable ("Growing columns:", table, rows));
    }


    // Make a table with 3 columns
    {
      CondProbTable table((UInt)numCols());
    
      // Add the 4 rows
      for (Size i=0; i<numRows(); i++)
        table.updateRow((UInt)i, rows[i]);
      
      // Test it
      ASSERT_NO_FATAL_FAILURE(
        testTable ("Fixed columns:", table, rows));
    }
  
  
    // Make a table, save to stream, then reload and test
    {
      CondProbTable table((UInt)numCols());
    
      // Add the 4 rows
      for (Size i=0; i<numRows(); i++)
        table.updateRow((UInt)i, rows[i]);
      
      // Save it
      stringstream state;
      table.saveState (state);
    
      CondProbTable newTable;
      newTable.readState (state);
      ASSERT_NO_FATAL_FAILURE(
        testTable ("Restored from state:", newTable, rows));
    }
  
        
    // Test saving an empty table
    {
      CondProbTable table;
      
      // Save it
      stringstream state;
      table.saveState (state);
    
      // Read it in
      CondProbTable newTable;
      newTable.readState (state);

      // Add the 4 rows
      for (Size i=0; i<numRows(); i++)
        newTable.updateRow((UInt)i, rows[i]);
      
      // Test it
      ASSERT_NO_FATAL_FAILURE(
        testTable ("Restored from empty state:", newTable, rows));
    }


  }
  
  //----------------------------------------------------------------------
} // end namespace


