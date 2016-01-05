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

#include <nupic/ntypes/MemStream.hpp>
#include <nupic/utils/Log.hpp>

#include <stdexcept>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

using namespace nupic;


static size_t memLimitsTest(size_t max)
{
  OMemStream ms;

  // Create a large string to dump the stream
  size_t chunkSize = 0x1000000;  // 16 MByte
  std::string test(chunkSize, 'M');  

  /*
    std::string test2 (0x10000000, '1');  
    std::string test3 (0x10000000, '2');  
    std::string test4 (0x10000000, '3');  
    std::string test5 (0x10000000, '4');  
    std::string test6 (0x10000000, '5');  
    std::string test7 (0x10000000, '6');  
    std::string test8 (0x10000000, '7');  
    std::string test9 (0x10000000, '8');  
  */


  size_t count = 1;
  while (count * chunkSize <= max) {
    //std::cout << hex << "0x" << count << ".";
    //std::cout.flush();
    try {
      ms << test;
    } catch (std::exception& /* unused */) {
      NTA_DEBUG << "Exceeded memory limit at " << std::hex << "0x" << count * chunkSize << std::dec 
                << " bytes.";
      break;
    }
    count++;
  }

  // Return largest size that worked
  return (count-1) * chunkSize;
}




// -------------------------------------------------------
// Test input stream
// -------------------------------------------------------
TEST(MemStreamTest, InputStream)
{
  std::string test("hi there");  

  IMemStream ms((char*)(test.data()), test.size());
  std::stringstream ss(test);

  for (int i=0; i<5; i++)
  {
    std::string s1, s2;
    ms >> s1;
    ss >> s2;
    ASSERT_EQ(s2, s1) << "in";
    ASSERT_EQ(ss.fail(), ms.fail()) << "in fail";
    ASSERT_EQ(ss.eof(), ms.eof()) << "in eof";
  }      


  // Test changing the buffer 
  std::string test2("bye now");  
  ms.str((char*)(test2.data()), test2.size());
  ms.seekg(0);
  ms.clear();
  std::stringstream ss2(test2);

  for (int i=0; i<5; i++)
  {
    std::string s1, s2;
    ms >> s1;
    ss2 >> s2;
    ASSERT_EQ(s2, s1) << "in2";
    ASSERT_EQ(ss2.fail(), ms.fail()) << "in2 fail";
    ASSERT_EQ(ss2.eof(), ms.eof()) << "in2 eof";
  }      
}


// -------------------------------------------------------
// Test setting the buffer on a default input stream
// -------------------------------------------------------
TEST(MemStreamTest, BufferDefaultInputStream)
{
  std::string test("third test");  

  IMemStream ms;
  ms.str((char*)(test.data()), test.size());
  std::stringstream ss(test);

  for (int i=0; i<5; i++)
  {
    std::string s1, s2;
    ms >> s1;
    ss >> s2;
    ASSERT_EQ(s2, s1) << "in2";
    ASSERT_EQ(ss.fail(), ms.fail()) << "in2 fail";
    ASSERT_EQ(ss.eof(), ms.eof()) << "in2 eof";
  }      
}


// -------------------------------------------------------
// Test output stream
// -------------------------------------------------------
TEST(MemStreamTest, OutputStream)
{
  OMemStream ms;
  std::stringstream ss;

  for (int i=0; i<500; i++)
  {
    ms << i << " ";
    ss << i << " ";
  }
    
  const char* dataP = ms.str();
  size_t size = ms.pcount();
  std::string msStr(dataP, size);
  std::string ssStr = ss.str();
  ASSERT_EQ(msStr, ssStr) << "out data";
  ASSERT_EQ(ms.eof(), ss.eof()) << "out eof";
  ASSERT_EQ(ms.fail(), ss.fail()) << "out fail";
}

// -------------------------------------------------------
// Test memory limits
// -------------------------------------------------------
// Set max at 0x10000000 for day to day testing so that test doesn't take too long.
// To determine the actual memory limits, change this max to something very large and
// see where we break. 
TEST(MemStreamTest, MemoryLimits)
{
  size_t max = 0x10000000L;
  size_t sizeLimit = memLimitsTest(max);
  ASSERT_EQ(sizeLimit >= max, true) << "maximum stream size";  
}
