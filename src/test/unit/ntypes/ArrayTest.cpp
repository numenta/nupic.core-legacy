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
 * Implementation of ArrayBase test
 */

#include <nupic/utils/Log.hpp>

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/BasicType.hpp>

#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>

#include <map>

#include <limits.h>

#include <gtest/gtest.h>

using namespace nupic;

struct ArrayTestParameters
{
  NTA_BasicType dataType;
  unsigned int dataTypeSize;
  int allocationSize; //We intentionally use an int instead of a size_t for
                      //these tests.  This is so that we can check test usage
                      //by a naive user who might use an int and accidentally
                      //pass negative values.
  std::string dataTypeText;
  bool testUsesInvalidParameters;
  
  ArrayTestParameters() :
    dataType((NTA_BasicType) -1),
    dataTypeSize(0),
    allocationSize(0),
    dataTypeText(""),
    testUsesInvalidParameters(true) {}
    
  ArrayTestParameters(NTA_BasicType dataTypeParam,
                      unsigned int dataTypeSizeParam,
                      int allocationSizeParam,
                      std::string dataTypeTextParam,
                      bool testUsesInvalidParametersParam) :
    dataType(dataTypeParam),
    dataTypeSize(dataTypeSizeParam),
    allocationSize(allocationSizeParam),
    dataTypeText(std::move(dataTypeTextParam)),
    testUsesInvalidParameters(testUsesInvalidParametersParam) { }
};


struct ArrayTest : public ::testing::Test
{
  std::map<std::string,ArrayTestParameters> testCases_;

  typedef std::map<std::string,ArrayTestParameters>::iterator
    TestCaseIterator;
  
  void setupArrayTests();
};




#ifdef NTA_INSTRUMENTED_MEMORY_GUARDED
//If we're running an appropriately instrumented build, then we're going
//to be running test code which intentionally commits access violations to
//verify proper functioning of the class; to do so, we're
//going to utilize the POSIX signal library and throw a C++ exception from
//our custom signal handler.
//
//This should be tested on Windows to verify POSIX compliance.  If it does
//not work, the Microsoft C++ extensions __try and __catch can be used to
//catch an access violation on Windows.
#include <signal.h>


class AccessViolationError
{
};

void AccessViolationHandler(int signal)
{
  throw AccessViolationError();
}

typedef void (*AccessViolationHandlerPointer)(int);

TEST_F(ArrayTest, testMemoryOperations)
{
  //Temporarily swap out the the segv and bus handlers.
  AccessViolationHandlerPointer existingSigsegvHandler;
  AccessViolationHandlerPointer existingSigbusHandler;

  existingSigsegvHandler = signal(SIGSEGV, AccessViolationHandler);
  existingSigbusHandler = signal(SIGBUS, AccessViolationHandler);
  
  //Since we're going to be testing the memory behavior of ArrayBase, we create a
  //pointer here (which will be set to the ArrayBase's buffer) while putting the
  //ArrayBase itself inside an artificial scope.  That way, when the ArrayBase goes out
  //of scope and is destructed we can test that ArrayBase doesn't leak the buffer
  //memory.  We prefer the artificial scope method to a pointer with new/delete
  //as it prevents our code from leaking memory under an unhandled error case.
  //
  //NOTE: For these tests to be consistent, the code must be built using
  //      instrumentation which intentionally guards memory and handles
  //      allocations/deallocations immediately (such as a debugging malloc
  //      library).  This test will NOT be run unless
  //      NTA_INSTRUMENTED_MEMORY_GUARDED is defined.
  
  void * ownedBufferLocation;

  {
    ArrayBase a(NTA_BasicType_Byte);
    
    a.allocateBuffer(10);

    ownedBufferLocation = a.getBuffer();
  
    //Verify that we can write into the buffer
    bool wasAbleToWriteToBuffer = true;
    try
    {
      for(unsigned int i = 0; i < 10; i++)
      {
        ((char *) ownedBufferLocation)[i] = 'A' + i;
      }
    }
    catch(AccessViolationError exception)
    {
      wasAbleToWriteToBuffer = false;
    }
    TEST2("Write to full length of allocated buffer should succeed",
          wasAbleToWriteToBuffer);

    //Verify that we can read from the buffer  
    char testRead = '\0';
    testRead = ((char *) ownedBufferLocation)[4];
    ASSERT_TRUE(!wasAbleToReadFromFreedBuffer) << "Read from freed buffer should fail";
  }

  bool wasAbleToReadFromFreedBuffer = true;
  try
  {
    char testRead = '\0';
    testRead = ((char *) ownedBufferLocation)[4];
  }
  catch(AccessViolationError exception)
  {
    wasAbleToReadFromFreedBuffer = false;
  }
  ASSERT_TRUE(!wasAbleToReadFromFreedBuffer) << "Read from freed buffer should fail";

  bool wasAbleToWriteToFreedBuffer = true;
  try
  {
    ((char *) ownedBufferLocation)[4] = 'A';
  }
  catch(AccessViolationError exception)
  {
    wasAbleToWriteToFreedBuffer = false;
  }
  ASSERT_TRUE(!wasAbleToWriteToFreedBuffer) << "Write to freed buffer should fail";

  signal(SIGSEGV, existingSigsegvHandler);
  signal(SIGBUS, existingSigbusHandler);
}

#endif

TEST_F(ArrayTest, testArrayCreation)
{
  setupArrayTests();
  
  boost::scoped_ptr<ArrayBase> arrayP;

  TestCaseIterator testCase;
  
  for(testCase = testCases_.begin(); testCase != testCases_.end(); testCase++)
  {
    char *buf = (char *) -1;
    
    if(testCase->second.testUsesInvalidParameters)
    {
      bool caughtException = false;

      try
      {
        arrayP.reset(new ArrayBase(testCase->second.dataType));
      }
      catch(nupic::Exception)
      {
        caughtException = true;
      }

      ASSERT_TRUE(caughtException)
        << "Test case: " +
        testCase->first +
        " - Should throw an exception on trying to create an invalid "
        "ArrayBase";
    }
    else
    {
      arrayP.reset(new ArrayBase(testCase->second.dataType));
      buf = (char *) arrayP->getBuffer();
      ASSERT_EQ(buf, nullptr)
                << "Test case: " +
                testCase->first +
                " - When not passed a size, a newly created ArrayBase should "
                "have a NULL buffer";
      ASSERT_EQ((size_t) 0, arrayP->getCount())
                  << "Test case: " +
                  testCase->first +
                  " - When not passed a size, a newly created ArrayBase should "
                  "have a count equal to zero";

      boost::scoped_array<char> buf2(new char[testCase->second.dataTypeSize *
                                              testCase->second.allocationSize]);
                                              
      arrayP.reset(new ArrayBase(testCase->second.dataType,
                             buf2.get(),
                             testCase->second.allocationSize));
      
      buf = (char *) arrayP->getBuffer();
      ASSERT_EQ(buf, buf2.get())
                << "Test case: " +
                testCase->first +
                " - Preallocating a buffer for a newly created ArrayBase should "
                "use the provided buffer";
      ASSERT_EQ((size_t) testCase->second.allocationSize, arrayP->getCount())
                << "Test case: " +
                testCase->first +
                " - Preallocating a buffer should have a count equal to our "
                "allocation size";
    }    
  }
}

TEST_F(ArrayTest, testBufferAllocation)
{
  testCases_.clear();
  testCases_["NTA_BasicType_Int32, size 0"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, 0, "Int32", false);
  testCases_["NTA_BasicType_Int32, size UINT_MAX"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, UINT_MAX, "Int32", true);
  testCases_["NTA_BasicType_Int32, size -10"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, -10, "Int32", true);
  testCases_["NTA_BasicType_Int32, size 10"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, 10, "Int32", false);
  
  bool caughtException;

  TestCaseIterator testCase;
  
  for(testCase = testCases_.begin(); testCase != testCases_.end(); testCase++)
  {
    caughtException = false;
    ArrayBase a(testCase->second.dataType);

    try
    {
      a.allocateBuffer((size_t) (testCase->second.allocationSize));
    }
    catch(std::exception& )
    {
      caughtException = true;
    }
      
    if(testCase->second.testUsesInvalidParameters)
    {
      ASSERT_TRUE(caughtException)
        << "Test case: " +
                  testCase->first +
                  " - allocation of an ArrayBase of invalid size should raise an "
                  "exception";
    }
    else
    {
      ASSERT_FALSE(caughtException)
        << "Test case: " + testCase->first +
            " - Allocation of an ArrayBase of valid size should return a "
            "valid pointer";
            
      caughtException = false;
      
      try
      {
        a.allocateBuffer(10);
      }
      catch(nupic::Exception)
      {
        caughtException = true;
      }
      
      ASSERT_TRUE(caughtException)
        << "Test case: " + testCase->first +
            " - allocating a buffer when one is already allocated should "
            "raise an exception";
      
      ASSERT_EQ((size_t) testCase->second.allocationSize, a.getCount())
        << "Test case: " + testCase->first +
           " - Size of allocated ArrayBase should match requested size";
    }
  }
}

TEST_F(ArrayTest, testBufferAssignment)
{
  testCases_.clear();
  testCases_["NTA_BasicType_Int32, buffer assignment"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, 10, "Int32", false);
  
  TestCaseIterator testCase;
  
  for(testCase = testCases_.begin(); testCase != testCases_.end(); testCase++)
  {
    boost::scoped_array<char> buf(new char[testCase->second.dataTypeSize *
                                           testCase->second.allocationSize]);
        
    ArrayBase a(testCase->second.dataType);
    a.setBuffer(buf.get(), testCase->second.allocationSize);
    
    ASSERT_EQ(buf.get(), a.getBuffer())
      << "Test case: " +
          testCase->first +
          " - setBuffer() should used the assigned buffer";

    boost::scoped_array<char> buf2(new char[testCase->second.dataTypeSize *
                                            testCase->second.allocationSize]);

    bool caughtException = false;
    
    try
    {
      a.setBuffer(buf2.get(), testCase->second.allocationSize);
    }
    catch(nupic::Exception)
    {
      caughtException = true;
    }
    
    ASSERT_TRUE(caughtException)
      << "Test case: " +
          testCase->first +
          " - setting a buffer when one is already set should raise an "
          "exception";
  }    
}

TEST_F(ArrayTest, testBufferRelease)
{
  testCases_.clear();
  testCases_["NTA_BasicType_Int32, buffer release"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, 10, "Int32", false);
  
  TestCaseIterator testCase;
  
  for(testCase = testCases_.begin(); testCase != testCases_.end(); testCase++)
  {
    boost::scoped_array<char> buf(new char[testCase->second.dataTypeSize *
                                           testCase->second.allocationSize]);
    
    ArrayBase a(testCase->second.dataType);
    a.setBuffer(buf.get(), testCase->second.allocationSize);
    a.releaseBuffer();
    
    ASSERT_EQ(nullptr, a.getBuffer())
      << "Test case: " +
          testCase->first +
          " - ArrayBase should no longer hold a reference to a locally allocated "
          "buffer after calling releaseBuffer";
  }    
}

TEST_F(ArrayTest, testArrayTyping)
{
  setupArrayTests();
  
  TestCaseIterator testCase;

  for(testCase = testCases_.begin(); testCase != testCases_.end(); testCase++)
  {
    //testArrayCreation() already validates that ArrayBase objects can't be created
    //using invalid NTA_BasicType parameters, so we skip those test cases here
    if(testCase->second.testUsesInvalidParameters)
    {
      continue;
    }
    
    ArrayBase a(testCase->second.dataType);

    ASSERT_EQ(testCase->second.dataType, a.getType())
      << "Test case: " +
          testCase->first +
          " - the type of a created ArrayBase should match the requested "
          "type";

    std::string name(BasicType::getName(a.getType()));
    ASSERT_EQ(testCase->second.dataTypeText, name)
      << "Test case: " +
          testCase->first +
          " - the string representation of a type contained in a "
          "created ArrayBase should match the expected string";
  } 
}

void ArrayTest::setupArrayTests()
{
  //we're going to test using all types that can be stored in the ArrayBase...
  //the NTA_BasicType enum overrides the default incrementing values for
  //some enumerated types, so we must reference them manually
  testCases_.clear();
  testCases_["NTA_BasicType_Byte"] =
    ArrayTestParameters(NTA_BasicType_Byte, 1, 10, "Byte", false);
  testCases_["NTA_BasicType_Int16"] =
    ArrayTestParameters(NTA_BasicType_Int16, 2, 10, "Int16", false);
  testCases_["NTA_BasicType_UInt16"] =
    ArrayTestParameters(NTA_BasicType_UInt16, 2, 10, "UInt16", false);
  testCases_["NTA_BasicType_Int32"] =
    ArrayTestParameters(NTA_BasicType_Int32, 4, 10, "Int32", false);
  testCases_["NTA_BasicType_UInt32"] =
    ArrayTestParameters(NTA_BasicType_UInt32, 4, 10, "UInt32", false); 
  testCases_["NTA_BasicType_Int64"] =
    ArrayTestParameters(NTA_BasicType_Int64, 8, 10, "Int64", false);
  testCases_["NTA_BasicType_UInt64"] =
    ArrayTestParameters(NTA_BasicType_UInt64, 8, 10, "UInt64", false);
  testCases_["NTA_BasicType_Real32"] =
    ArrayTestParameters(NTA_BasicType_Real32, 4, 10, "Real32", false);
  testCases_["NTA_BasicType_Real64"] =
    ArrayTestParameters(NTA_BasicType_Real64, 8, 10, "Real64", false);
  testCases_["NTA_BasicType_Bool"] =
    ArrayTestParameters(NTA_BasicType_Bool, sizeof(bool), 10, "Bool", false);
#ifdef NTA_DOUBLE_PRECISION 
  testCases_["NTA_BasicType_Real"] =
    ArrayTestParameters(NTA_BasicType_Real, 8, 10, "Real64", false);
#else 
  testCases_["NTA_BasicType_Real"] =
    ArrayTestParameters(NTA_BasicType_Real, 4, 10, "Real32", false);
#endif
  testCases_["Non-existent NTA_BasicType"] =
    ArrayTestParameters((NTA_BasicType) -1, 0, 10, "N/A", true);
}
