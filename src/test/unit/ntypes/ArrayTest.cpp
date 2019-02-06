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

#define UNUSED(x) (void)(x)

#include <nupic/utils/Log.hpp>

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/os/OS.hpp>


#include <map>
#include <memory>
#include <limits.h>

#include <gtest/gtest.h>

using namespace nupic;

// First, some structures to help in testing.
struct ArrayTestParameters {
  NTA_BasicType dataType;
  unsigned int dataTypeSize;
  int allocationSize; // We intentionally use an int instead of a size_t for
                      // these tests.  This is so that we can check test usage
                      // by a naive user who might use an int and accidentally
                      // pass negative values.
  std::string dataTypeText;
  bool testUsesInvalidParameters;

  ArrayTestParameters()
      : dataType((NTA_BasicType)-1), dataTypeSize(0), allocationSize(0),
        dataTypeText(""), testUsesInvalidParameters(true) {}

  ArrayTestParameters(NTA_BasicType dataTypeParam,
                      unsigned int dataTypeSizeParam, int allocationSizeParam,
                      std::string dataTypeTextParam,
                      bool testUsesInvalidParametersParam)
      : dataType(dataTypeParam), dataTypeSize(dataTypeSizeParam),
        allocationSize(allocationSizeParam),
        dataTypeText(std::move(dataTypeTextParam)),
        testUsesInvalidParameters(testUsesInvalidParametersParam) {}
};

struct ArrayTest : public ::testing::Test {
  std::map<std::string, ArrayTestParameters> testCases_;

  typedef std::map<std::string, ArrayTestParameters>::iterator TestCaseIterator;

  void setupArrayTests();
};

#ifdef NTA_INSTRUMENTED_MEMORY_GUARDED
// If we're running an appropriately instrumented build, then we're going
// to be running test code which intentionally commits access violations to
// verify proper functioning of the class; to do so, we're
// going to utilize the POSIX signal library and throw a C++ exception from
// our custom signal handler.
//
// This should be tested on Windows to verify POSIX compliance.  If it does
// not work, the Microsoft C++ extensions __try and __catch can be used to
// catch an access violation on Windows.
#include <signal.h>

class AccessViolationError {};

void AccessViolationHandler(int signal) { throw AccessViolationError(); }

typedef void (*AccessViolationHandlerPointer)(int);

TEST_F(ArrayTest, testMemoryOperations) {
  // Temporarily swap out the the segv and bus handlers.
  AccessViolationHandlerPointer existingSigsegvHandler;
  AccessViolationHandlerPointer existingSigbusHandler;

  existingSigsegvHandler = signal(SIGSEGV, AccessViolationHandler);
  existingSigbusHandler = signal(SIGBUS, AccessViolationHandler);

  // Since we're going to be testing the memory behavior of ArrayBase, we create
  // a pointer here (which will be set to the ArrayBase's buffer) while putting
  // the ArrayBase itself inside an artificial scope.  That way, when the
  // ArrayBase goes out of scope and is destructed we can test that ArrayBase
  // doesn't leak the buffer memory.  We prefer the artificial scope method to a
  // pointer with new/delete as it prevents our code from leaking memory under
  // an unhandled error case.
  //
  // NOTE: For these tests to be consistent, the code must be built using
  //      instrumentation which intentionally guards memory and handles
  //      allocations/deallocations immediately (such as a debugging malloc
  //      library).  This test will NOT be run unless
  //      NTA_INSTRUMENTED_MEMORY_GUARDED is defined.

  void *ownedBufferLocation;

  {
    ArrayBase a(NTA_BasicType_Byte);

    a.allocateBuffer(10);

    ownedBufferLocation = a.getBuffer();

    // Verify that we can write into the buffer
    bool wasAbleToWriteToBuffer = true;
    try {
      for (unsigned int i = 0; i < 10; i++) {
        ((char *)ownedBufferLocation)[i] = 'A' + i;
      }
    } catch (AccessViolationError& exception) {
	  UNUSED(exception);
      wasAbleToWriteToBuffer = false;
    }
    TEST2("Write to full length of allocated buffer should succeed",
          wasAbleToWriteToBuffer);

    // Verify that we can read from the buffer
    char testRead = '\0';
    testRead = ((char *)ownedBufferLocation)[4];
    ASSERT_TRUE(!wasAbleToReadFromFreedBuffer)
        << "Read from freed buffer should fail";
  }

  bool wasAbleToReadFromFreedBuffer = true;
  try {
    char testRead = '\0';
    testRead = ((char *)ownedBufferLocation)[4];
  } catch (AccessViolationError& exception) {
	  UNUSED(exception);
    wasAbleToReadFromFreedBuffer = false;
  }
  ASSERT_TRUE(!wasAbleToReadFromFreedBuffer)
      << "Read from freed buffer should fail";

  bool wasAbleToWriteToFreedBuffer = true;
  try {
    ((char *)ownedBufferLocation)[4] = 'A';
  } catch (AccessViolationError& exception) {
	  UNUSED(exception);
    wasAbleToWriteToFreedBuffer = false;
  }
  ASSERT_TRUE(!wasAbleToWriteToFreedBuffer)
      << "Write to freed buffer should fail";

  signal(SIGSEGV, existingSigsegvHandler);
  signal(SIGBUS, existingSigbusHandler);
}

#endif



TEST_F(ArrayTest, testMemory) {
  // We are going to try to perform the same test as above but without
  // messing with the signal handlers.
  char testValue;
  char *ownedBufferLocation;
  Array b(NTA_BasicType_Byte);
  const Array* c;  // a read only pointer version
  Array g; 
  Array d;
  {
    Array a(NTA_BasicType_Byte);

    a.allocateBuffer(100000);
    ownedBufferLocation = (char *)a.getBuffer();
    EXPECT_TRUE(a.getCount() == 100000);

    // Verify that we can write into the buffer
    bool wasAbleToWriteToBuffer = true;
    try {
      for (unsigned int i = 0; i < 9; i++) {
        ownedBufferLocation[i] = 'a' + i;
      }
      ownedBufferLocation[9] = '\0';
    } catch (std::exception &e) {
      UNUSED(e);
      wasAbleToWriteToBuffer = false;
    }
    EXPECT_TRUE(wasAbleToWriteToBuffer)
        << "Write to full length of allocated buffer should have succeeded.";

    // Verify that we can read from the buffer
    testValue = '\0';
    testValue = ((char *)ownedBufferLocation)[8];
    EXPECT_TRUE(testValue == 'i')
        << "Was not able to read the right thing from the buffer.";
    const Array f(a);
    EXPECT_TRUE(f.isInstance(a)); 
    EXPECT_TRUE(((char *)a.getBuffer())[4] == 'e');
    EXPECT_TRUE(((const char *)f.getBuffer())[4] == 'e');

    // full copy
    b = a.copy();
    EXPECT_TRUE(b.getType() == NTA_BasicType_Byte)
        << "The data type should have been copied to the Array.";
    EXPECT_TRUE(((char *)b.getBuffer())[4] == 'e')
        << "Should be able to read the full copy Array instance.";
    ((char *)b.getBuffer())[4] = 'z';
    EXPECT_TRUE(((char *)b.getBuffer())[4] == 'z')
        << "Should be able to modify the new Array instance.";
    EXPECT_TRUE(((char *)a.getBuffer())[4] == 'e')
        << "Should not be able to modify the original Array instance.";

    c = &b;
    EXPECT_FALSE(b.isInstance(a)); 
    EXPECT_TRUE(b.isInstance(*c)); 
  }
  // The Array object a is now out of scope so its buffer should be invalid.

  {
    Array e(NTA_BasicType_Byte);
    e.allocateBuffer(10);
    ownedBufferLocation = (char*)e.getBuffer();
    for (unsigned int i = 0; i < 9; i++) {
      ownedBufferLocation[i] = 'A' + i;
    }
    ownedBufferLocation[9] = '\0';
    char testRead = ownedBufferLocation[4];
    EXPECT_TRUE(testRead == 'E') << "Should be able to write and read the Array.";

    // make an asignment to another Array instance and to a const Array instance.
    d = e; // shallow copy.  Buffer not copied but remains valid after a is deleted.
    c = &d;
    EXPECT_TRUE(d.getType() == NTA_BasicType_Byte)  << "The data type should have been copied to the Array.";

    EXPECT_TRUE(((char *)d.getBuffer())[4] == 'E')  << "Should be able to read the new Array instance.";
    EXPECT_TRUE(((const char *)c->getBuffer())[4] == 'E')  << "Should be able to read the new const Array instance.";
    ((char *)d.getBuffer())[4] = 'Z';
    EXPECT_TRUE(((char *)d.getBuffer())[4] == 'Z') << "Should be able to modify the new Array instance.";
    EXPECT_TRUE(((char *)e.getBuffer())[4] == 'Z') << "The original buffer should also change.";
    
    EXPECT_TRUE(d.isInstance(e)); 
  }
  // the Array e is now out of scope but the buffer should still be valid.
  EXPECT_TRUE(((char *)d.getBuffer())[4] == 'Z')   << "Should still see the buffer in the new Array instance.";
  EXPECT_TRUE(((char *)d.getBuffer())[4] == 'Z')   << "The const Array instance should also still see the buffer.";
  EXPECT_TRUE(ownedBufferLocation[4] == 'Z')  << "The pointer should also still see the buffer.";

  g = b;
  EXPECT_TRUE(g.isInstance(b)); 

  b.releaseBuffer();
  // The b buffer is no longer valid but c still has a reference to it.
  EXPECT_TRUE(b.getBuffer() == nullptr)  << "expected a null pointer because the buffer was released.";
  EXPECT_TRUE(b.getCount() == 0)  << "expected a 0 length because the buffer was released.";
  EXPECT_TRUE(((char *)ownedBufferLocation)[4] == 'Z') << "The pointer should still see the e buffer.";

  // c->releaseBuffer();   //non-const functions on c should give compile errors.
}

// This test is ran with all data types contained by an Array
TEST_F(ArrayTest, testArrayCreation) {
  setupArrayTests();

  Array *arrayP;

  TestCaseIterator testCase;

  for (testCase = testCases_.begin(); testCase != testCases_.end();
       testCase++) {
    char *buf = (char *)-1;

    if (testCase->second.testUsesInvalidParameters) {
      bool caughtException = false;

      try {
        arrayP = new Array(testCase->second.dataType);
      } catch (nupic::Exception& e) {
        UNUSED(e);
        caughtException = true;
      }

      ASSERT_TRUE(caughtException)
          << "Test case: " + testCase->first +
                 " - Should throw an exception on trying to create an invalid "
                 "ArrayBase";
    } else {
      arrayP = new Array(testCase->second.dataType);
      buf = (char *)arrayP->getBuffer();
      ASSERT_EQ(buf, nullptr)
          << "Test case: " + testCase->first +
                 " - When not passed a size, a newly created Array should "
                 "have a NULL buffer";
      ASSERT_EQ((size_t)0, arrayP->getCount())
          << "Test case: " + testCase->first +
                 " - When not passed a size, a newly created Array should "
                 "have a count equal to zero";
      delete arrayP;

	    size_t capacity = testCase->second.dataTypeSize * testCase->second.allocationSize;
      std::shared_ptr<char> buf2(new char[capacity], std::default_delete<char[]>());
      std::memset(buf2.get(), 0, capacity); // fill with 0's

      // make copy of buffer into the Array
      arrayP = new Array(testCase->second.dataType, 
                         buf2.get(),
                         testCase->second.allocationSize);
      EXPECT_TRUE(arrayP->getType() == testCase->second.dataType)  << "The data type should have been copied to the Array.";

      buf = (char *)arrayP->getBuffer();
      ASSERT_EQ((size_t)testCase->second.allocationSize, arrayP->getCount())
          << "Test case: " + testCase->first +
                 " - Preallocating a buffer should have a count equal to our "
                 "allocation size";
      delete arrayP;
    }
  }
}

TEST_F(ArrayTest, testBufferAllocation) {
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

  for (testCase = testCases_.begin(); testCase != testCases_.end();
       testCase++) {
    caughtException = false;
    Array a(testCase->second.dataType);

    try {
      a.allocateBuffer((size_t)(testCase->second.allocationSize));
    } catch (std::exception &e) {
      UNUSED(e);
      caughtException = true;
    }

    if (testCase->second.testUsesInvalidParameters) {
      ASSERT_TRUE(caughtException) << "Test case: " + testCase->first +
                                          " - allocation of an ArrayBase of "
                                          "invalid size should raise an "
                                          "exception";
    } else {
      ASSERT_FALSE(caughtException)
          << "Test case: " + testCase->first +
                 " - Allocation of an ArrayBase of valid size should return a "
                 "valid pointer";

      // Note: reallocating a buffer is now allowed.  dek, 08/07/2017
      caughtException = false;

      try
      {
        a.allocateBuffer(10);
      }
      catch(nupic::Exception& e)
      {
        UNUSED(e);
        caughtException = true;
      }

      ASSERT_FALSE(caughtException)
        << "Test case: " + testCase->first +
            " - allocating a buffer when one is already allocated should "
            "not raise an exception. The allocation will release the previous buffer.";

      ASSERT_EQ((size_t)10, a.getCount())
          << "Test case: " + testCase->first +
                 " - Size of allocated ArrayBase should match requested size";
    }
  }
}

TEST_F(ArrayTest, testUnownedBuffer) {
  testCases_.clear();
  testCases_["NTA_BasicType_Int32, buffer assignment"] =
      ArrayTestParameters(NTA_BasicType_Int32, 4, 10, "Int32", false);

  TestCaseIterator testCase;

  for (testCase = testCases_.begin(); testCase != testCases_.end();  testCase++) {
    size_t capacity =
        testCase->second.dataTypeSize * testCase->second.allocationSize;
    std::shared_ptr<char> buf(new char[capacity], std::default_delete<char[]>());

    Array a(testCase->second.dataType);
    a.setBuffer(buf.get(), testCase->second.allocationSize);

    ASSERT_EQ(buf.get(), a.getBuffer())
        << "Test case: " + testCase->first +
               " - setBuffer() should used the assigned buffer";

    capacity = testCase->second.dataTypeSize * testCase->second.allocationSize;
    std::shared_ptr<char> buf2(new char[capacity], std::default_delete<char[]>());

    // setting a buffer when one is already set is NOW allowed. dek 08/07/2018
    // previous buffer is freed.
    bool caughtException = false;

    try
    {
      a.setBuffer(buf2.get(), testCase->second.allocationSize);
    }
    catch(nupic::Exception& e)
    {
      UNUSED(e);
      caughtException = true;
    }

    ASSERT_FALSE(caughtException)
      << "Test case: " +
          testCase->first +
          " - setting a buffer when one is already set should not raise an "
          "exception";
    ASSERT_EQ(a.getCount(), (size_t)testCase->second.allocationSize)
        << "Buffer size should be the requested amount.";
  }
}

TEST_F(ArrayTest, testBufferRelease) {
  //testCases_.clear();
  //testCases_["NTA_BasicType_Int32, buffer release"] =
  //    ArrayTestParameters<Int32>(NTA_BasicType_Int32, 4, 10, "Int32", false);

  TestCaseIterator testCase;

  for (testCase = testCases_.begin(); testCase != testCases_.end(); testCase++) {
    std::shared_ptr<char> buf(new char[testCase->second.dataTypeSize *
                                           testCase->second.allocationSize], std::default_delete<char[]>());

    Array a(testCase->second.dataType);
    a.setBuffer(buf.get(), testCase->second.allocationSize);
    a.releaseBuffer();

    ASSERT_EQ(nullptr, a.getBuffer())
        << "Test case: " + testCase->first +
               " - ArrayBase should no longer hold a reference to a locally "
               "allocated "
               "buffer after calling releaseBuffer";
  }
}

TEST_F(ArrayTest, testArrayTyping) {
  setupArrayTests();

  TestCaseIterator testCase;

  for (testCase = testCases_.begin(); testCase != testCases_.end();
       testCase++) {
    // testArrayCreation() already validates that Array objects can't be
    // created using invalid NTA_BasicType parameters, so we skip those test
    // cases here
    if (testCase->second.testUsesInvalidParameters) {
      continue;
    }

    Array a(testCase->second.dataType);

    ASSERT_EQ(testCase->second.dataType, a.getType())
        << "Test case: " + testCase->first +
               " - the type of a created ArrayBase should match the requested "
               "type";

    std::string name(BasicType::getName(a.getType()));
    ASSERT_EQ(testCase->second.dataTypeText, name)
        << "Test case: " + testCase->first +
               " - the string representation of a type contained in a "
               "created Array should match the expected string";
  }
}


TEST_F(ArrayTest, testArrayBasefunctions) {
  setupArrayTests();

  std::vector<Int32> testdata = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0};
  TestCaseIterator testCase;

  for (testCase = testCases_.begin(); testCase != testCases_.end(); testCase++) {
    // testArrayCreation() already validates that Array objects can't be
    // created using invalid NTA_BasicType parameters, so we skip those 
    // negetive test cases here
    if (testCase->second.testUsesInvalidParameters) {
      continue;
    }
    std::cerr << "  Iteration " << testCase->first << std::endl;

    size_t nCols = testCase->second.allocationSize;
    size_t bufsize = testCase->second.dataTypeSize * testCase->second.allocationSize;
    std::shared_ptr<char> buf(new char[bufsize], std::default_delete<char[]>());
    std::memset(buf.get(), 0, bufsize);

    // constructors;  Allocate an array using the test data.
    Array a(testCase->second.dataType);
    Array b(testCase->second.dataType, buf.get(), nCols);
    Array c;

    // getType
    ASSERT_EQ(testCase->second.dataType, a.getType())
        << "Test case: " + testCase->first + " type missmatch";
    ASSERT_EQ(b.getType(), a.getType())
        << "Test case: " + testCase->first + " type missmatch";



    // allocateBuffer
    // getCount
    a.allocateBuffer(nCols);
    EXPECT_EQ(a.getCount(), nCols);

    // zeroBuffer
    // operator==
    a.zeroBuffer();
    EXPECT_TRUE(a == b);

    a.populate(testdata);
    //std::cerr << "ArrayTest:testArrayBasefunctions a=" << a << std::endl;
    nCols = testdata.size();

    // getMaxElementsCount, getCount, setCount, copy
    if (testCase->second.dataType != NTA_BasicType_SDR) {
      // SDR cannot do a truncate of data keeping capacity fixed.
      // For everyone else, remove the last element.
      size_t buffer_cnt = a.getCount();
      c = a.copy();
      c.setCount(buffer_cnt - 1);
      EXPECT_EQ(c.getMaxElementsCount(), buffer_cnt);
      EXPECT_EQ(c.getCount(), buffer_cnt - 1);
    }
    
    c = a; // shallow copy
    EXPECT_EQ(c.getCount(), a.getCount());
    EXPECT_TRUE(c.getBuffer() == a.getBuffer());
    EXPECT_EQ(a.getCount(), nCols);
    EXPECT_EQ(a.getMaxElementsCount(), nCols);
    EXPECT_EQ(a.getBufferSize(), nCols * testCase->second.dataTypeSize );


    if (testCase->second.dataType == NTA_BasicType_SDR) {
      // Just to be sure an SDR can play here,
      // Only SDR has dimensions
      std::vector<UInt> dim({10, 10});
      SDR sdr(dim);
      Array s(sdr); // makes a copy of sdr
      EXPECT_TRUE(s.getSDR() != &sdr);
      EXPECT_EQ(s.getCount(), 100u);

      std::vector<UInt> dim2({10, 20});
      s.allocateBuffer(dim2);  // re-creates sdr
      EXPECT_EQ(s.getCount(), 200u);

      // wrapper an existing sdr
      Array m(NTA_BasicType_SDR);
      m.setBuffer(sdr);
      EXPECT_TRUE(m.getSDR() == &sdr);
      EXPECT_EQ(m.getCount(), 100u);

      std::vector<Byte> row = a.asVector<Byte>();
      SDR_flatSparse_t &v = a.getSDR()->getFlatSparse();

      EXPECT_EQ(v.size(), 10u);


    } else {
      // SDR cannot do subsets
      Array q = a.subset(5, 2);
      EXPECT_EQ(q.getCount(), 2u);
      std::vector<Int32> v = q.asVector<Int32>();
      EXPECT_EQ(v.size(), 2u);
      if (testCase->second.dataType == NTA_BasicType_Bool) {
        EXPECT_EQ(v[0], 1);
        EXPECT_EQ(v[1], 1);
      }
    }

  }
}


void ArrayTest::setupArrayTests() {
  // we're going to test using all types that can be stored in the ArrayBase...
  // the NTA_BasicType enum overrides the default incrementing values for
  // some enumerated types, so we must reference them manually
  //    Fields:
  //       template<type-of-element>
  //       dataType,  dataTypeSize,  allocationSize, dataTypeText, testUsesInvalidParameters
  testCases_.clear();
  testCases_["NTA_BasicType_Byte"] =
      ArrayTestParameters(NTA_BasicType_Byte, sizeof(Byte), 10, "Byte", false);
  testCases_["NTA_BasicType_Int16"] =
      ArrayTestParameters(NTA_BasicType_Int16, sizeof(Int16), 10, "Int16", false);
  testCases_["NTA_BasicType_UInt16"] =
      ArrayTestParameters(NTA_BasicType_UInt16, sizeof(UInt16), 10, "UInt16", false);
  testCases_["NTA_BasicType_Int32"] =
      ArrayTestParameters(NTA_BasicType_Int32, sizeof(Int32), 10, "Int32", false);
  testCases_["NTA_BasicType_UInt32"] =
      ArrayTestParameters(NTA_BasicType_UInt32, sizeof(UInt32), 10, "UInt32", false);
  testCases_["NTA_BasicType_Int64"] =
      ArrayTestParameters(NTA_BasicType_Int64, sizeof(Int64), 10, "Int64", false);
  testCases_["NTA_BasicType_UInt64"] =
      ArrayTestParameters(NTA_BasicType_UInt64, sizeof(UInt64), 10, "UInt64", false);
  testCases_["NTA_BasicType_Real32"] =
      ArrayTestParameters(NTA_BasicType_Real32, sizeof(Real32), 10, "Real32", false);
  testCases_["NTA_BasicType_Real64"] =
      ArrayTestParameters(NTA_BasicType_Real64, sizeof(Real64), 10, "Real64", false);
  testCases_["NTA_BasicType_Bool"] =
      ArrayTestParameters(NTA_BasicType_Bool, sizeof(bool), 10, "Bool", false);
  testCases_["NTA_BasicType_SDR"] =
      ArrayTestParameters(NTA_BasicType_SDR, sizeof(char), 10, "SDR", false);
#ifdef NTA_DOUBLE_PRECISION
  testCases_["NTA_BasicType_Real"] =
      ArrayTestParameters(NTA_BasicType_Real, 8, 10, "Real64", false);
#else
  testCases_["NTA_BasicType_Real"] =
      ArrayTestParameters(NTA_BasicType_Real, 4, 10, "Real32", false);
#endif
  testCases_["Non-existent NTA_BasicType"] =
      ArrayTestParameters((NTA_BasicType)-1, 0, 10, "N/A", true);
}
