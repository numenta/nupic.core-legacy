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
 * Implementation for Buffer unit tests
 */

#include <nupic/math/Math.hpp>
#include <nupic/utils/Log.hpp>
#include <cstring> // strlen

// This test accesses private methods. 
#define private public
#include <nupic/ntypes/Buffer.hpp>
#undef private

#include <algorithm>
#include <gtest/gtest.h>

using namespace nupic;

void testReadBytes_VariableSizeBufferHelper(Size buffSize)
{
  std::vector<Byte> in;
  std::vector<Byte> out;
  in.resize(buffSize+1);
  out.resize(buffSize+1);
    
  std::fill(in.begin(), in.begin()+in.capacity(), 'I');
  std::fill(out.begin(), out.begin()+out.capacity(), 'O');
  
  for (Size i = 0; i <= buffSize; ++i)
  {
    ASSERT_TRUE(in[i] == 'I');
    ASSERT_TRUE(out[i] == 'O');
  }
  
  // Populate the ReadBuffer with the input
  ReadBuffer rb(&in[0], buffSize);
    
  // Get the abstract interface
  IReadBuffer & r = rb;
  
  // Prepare for reading from the read buffer in chunks
  const Size CHUNK_SIZE = 10;
  Size size = CHUNK_SIZE;

  // Read chunks until the buffer is exhausted and write everything to out buffer
  Size index = 0;
  while (size == CHUNK_SIZE)
  {
    Int32 res = r.read(&out[index], size);
    ASSERT_TRUE(res == 0);
    index += size;
  }
  
  // Verify that last index and last read size are correct
  ASSERT_TRUE(index == buffSize);
  ASSERT_TRUE(size == buffSize % CHUNK_SIZE);
  
  // Check corner cases
  ASSERT_TRUE(out[0] == 'I');
  ASSERT_TRUE(out[buffSize-1] == 'I');
  ASSERT_TRUE(out[buffSize] == 'O');
  
  // Check that all other values have been read correctly
  Size i;
  for (i = 1; i < buffSize-1; ++i)
    ASSERT_TRUE(out[i] == 'I');
}

TEST(BufferTest, testReadBytes_VariableSizeBuffer)
{
  ASSERT_NO_FATAL_FAILURE(
    testReadBytes_VariableSizeBufferHelper(5));
  
//  testReadBytes_VariableSizeBufferHelpter(128);
//  testReadBytes_VariableSizeBufferHelpter(227);
//  testReadBytes_VariableSizeBufferHelpter(228);
//  testReadBytes_VariableSizeBufferHelpter(229);
//  testReadBytes_VariableSizeBufferHelpter(315);
//  testReadBytes_VariableSizeBufferHelpter(482);
//  testReadBytes_VariableSizeBufferHelpter(483);
//  testReadBytes_VariableSizeBufferHelpter(484);
//  testReadBytes_VariableSizeBufferHelpter(512);
//  testReadBytes_VariableSizeBufferHelpter(2000);
//  testReadBytes_VariableSizeBufferHelpter(20000);
}


TEST(BufferTest, testReadBytes_SmallBuffer)
{
  ReadBuffer b((const Byte *)"123", 3);

  IReadBuffer & reader = b;   

  Byte out[5];
  Size size = 0;
  Int32 res = 0;
  
  size = 2;
  res = reader.read(out, size);
  ASSERT_TRUE(res == 0) << "BufferTest::testReadBuffer(), reader.read(2) failed";
  ASSERT_TRUE(size == 2) << "BufferTest::testReadBuffer(), reader.read(2) failed";
  ASSERT_TRUE(out[0] == '1') << "BufferTest::testReadBuffer(), out[0] should be 1 after reading 1,2"; 
  ASSERT_TRUE(out[1] == '2') << "BufferTest::testReadBuffer(), out[1] should be 2 after reading 1,2";
  
  size = 2;
  res = reader.read(out+2, size);
  ASSERT_TRUE(res == 0) << "BufferTest::testReadBuffer(), reader.read(2) failed";
  ASSERT_TRUE(size == 1) << "BufferTest::testReadBuffer(), reader.read(2) failed";
  ASSERT_TRUE(out[0] == '1') << "BufferTest::testReadBuffer(), out[0] should be 1 after reading 3"; 
  ASSERT_TRUE(out[1] == '2') << "BufferTest::testReadBuffer(), out[1] should be 2 after reading 3";
  ASSERT_TRUE(out[2] == '3') << "BufferTest::testReadBuffer(), out[2] should be 3 after reading 3";  
}

TEST(BufferTest, testWriteBytes)
{
  WriteBuffer b;
  Byte out[5] = { 1, 2, 3, 4, 5 };
  IWriteBuffer & writer = b;   
  ASSERT_TRUE(writer.getSize() == 0) << "BufferTest::testWriteBuffer(), writer.getSize() should be 0 before putting anything in";
  Size size = 3;
  writer.write(out, size);
  ASSERT_TRUE(writer.getSize() == 3) << "BufferTest::testWriteBuffer(), writer.getSize() should be 3 after writing 1,2,3";
  size = 2;
  writer.write(out+3, size);
  ASSERT_TRUE(writer.getSize() == 5) << "BufferTest::testWriteBuffer(), writer.getSize() should be 5 after writing 4,5";
  const Byte * s = writer.getData();
  size = writer.getSize();
  //NTA_INFO << "s=" << string(s, size) << ", size=" << size;
  ASSERT_TRUE(std::string(s, size) == std::string("\1\2\3\4\5")) << "BufferTest::testWriteBuffer(), writer.str() == 12345";
}

TEST(BufferTest, testEvenMoreComplicatedSerialization)
{
  struct X
  {
    X() :  a((Real)3.4)
         , b(6)
         , c('c')
         , e((Real)-0.04)
    {
      for (int i = 0; i < 4; ++i)
        d[i] = 'A' + i;

      for (int i = 0; i < 3; ++i)
        f[i] = 100 + i;    
    }
    
    Real a;
    UInt32 b;
    Byte c;
    Byte d[4];
    Real e;
    Int32  f[3];
  };
  
  X xi[2];

  xi[0].a = (Real)8.8;
  xi[1].a = (Real)4.5;
  xi[1].c = 't';
  xi[1].d[0] = 'X';
  xi[1].e = (Real)3.14;
  xi[1].f[0] = -999;  
  // Write the two Xs to a buffer
  WriteBuffer wb;
  ASSERT_TRUE(wb.getSize() == 0) << "BufferTest::testComplicatedSerialization(), empty WriteBuffer should have 0 size";
  
  // Write the number of Xs
  UInt32 size = 2;
  wb.write((UInt32 &)size);
  // Write all Xs.
  for (UInt32 i = 0; i < size; ++i)
  {
    wb.write(xi[i].a);
    wb.write(xi[i].b);
    wb.write(xi[i].c);
    Size len = 4;
    wb.write((const Byte *)xi[i].d, len);
    wb.write(xi[i].e);
    len = 3;
    wb.write(xi[i].f, len);  
  }
  
  ReadBuffer rb(wb.getData(), wb.getSize());
  // Read number of Xs
  rb.read(size);
  // Allocate array of Xs
  auto xo = new X[size];
  for (Size i = 0; i < size; ++i)
  {
    rb.read(xo[i].a);
    rb.read(xo[i].b);
    rb.read(xo[i].c);
    Size len = 4;
    Int32 res = rb.read(xo[i].d, len);
    ASSERT_TRUE(res == 0) << "BufferTest::testComplicatedSerialization(), rb.read(xi[i].d, 4) failed";
    ASSERT_TRUE(len == 4) << "BufferTest::testComplicatedSerialization(), rb.read(xi[i].d, 4) == 4";
    rb.read(xo[i].e);
    len = 3;
    rb.read(xo[i].f, len);
    NTA_INFO << "xo[" << i << "]={" << xo[i].a << " "
             << xo[i].b << " " 
             << xo[i].c << " " 
             << "'" << std::string(xo[i].d, 4) << "'" 
             << " " << xo[i].e << " "
             << "'" << xo[i].f[0] << "," << xo[i].f[1] << "," << xo[i].f[2] << "'" 
             ;
  }
  
  ASSERT_TRUE(nearlyEqual(xo[0].a, nupic::Real(8.8))) << "BufferTest::testComplicatedSerialization(), xo[0].a == 8.8";
  ASSERT_TRUE(xo[0].b == 6) << "BufferTest::testComplicatedSerialization(), xo[0].b == 6";
  ASSERT_TRUE(xo[0].c == 'c') << "BufferTest::testComplicatedSerialization(), xo[0].c == 'c'";
  ASSERT_TRUE(std::string(xo[0].d, 4) == std::string("ABCD")) << "BufferTest::testComplicatedSerialization(), xo[0].d == ABCD";
  ASSERT_TRUE(nearlyEqual(xo[0].e, nupic::Real(-0.04))) << "BufferTest::testComplicatedSerialization(), xo[0].e == -0.04";
  ASSERT_TRUE(xo[0].f[0] == 100) << "BufferTest::testComplicatedSerialization(), xo[0].f[0] == 100";
  ASSERT_TRUE(xo[0].f[1] == 101) << "BufferTest::testComplicatedSerialization(), xo[0].f[1] == 101";
  ASSERT_TRUE(xo[0].f[2] == 102) << "BufferTest::testComplicatedSerialization(), xo[0].f[2] == 102";
  
  ASSERT_TRUE(xo[1].a == nupic::Real(4.5)) << "BufferTest::testComplicatedSerialization(), xo[1].a == 4.5";
  ASSERT_TRUE(xo[1].b == 6) << "BufferTest::testComplicatedSerialization(), xo[1].b == 6";
  ASSERT_TRUE(xo[1].c == 't') << "BufferTest::testComplicatedSerialization(), xo[1].c == 't'";
  ASSERT_TRUE(std::string(xo[1].d, 4) == std::string("XBCD")) << "BufferTest::testComplicatedSerialization(), xo[1].d == XBCD";
  ASSERT_TRUE(nearlyEqual(xo[1].e, nupic::Real(3.14))) << "BufferTest::testComplicatedSerialization(), xo[1].e == 3.14";
  ASSERT_TRUE(xo[1].f[0] == -999) << "BufferTest::testComplicatedSerialization(), xo[1].f[0] == -999";
  ASSERT_TRUE(xo[1].f[1] == 101) << "BufferTest::testComplicatedSerialization(), xo[1].f[1] == 101";
  ASSERT_TRUE(xo[1].f[2] == 102) << "BufferTest::testComplicatedSerialization(), xo[1].f[2] == 102";
}

TEST(BufferTest, testComplicatedSerialization)
{
  struct X
  {
    X() :  a((Real)3.4)
         , b(6)
         , c('c')
         , e((Real)-0.04)
    {
      for (int i = 0; i < 4; ++i)
        d[i] = 'A' + i;
    }
    
    Real a;
    UInt32 b;
    Byte c;
    Byte d[4];
    Real e;
  };
  
  X xi[2];

  xi[0].a = (Real)8.8;
  xi[1].a = (Real)4.5;
  xi[1].c = 't';
  xi[1].d[0] = 'X';
  xi[1].e = (Real)3.14;
  
  // Write the two Xs to a buffer
  WriteBuffer wb;
  ASSERT_TRUE(wb.getSize() == 0) << "BufferTest::testComplicatedSerialization(), empty WriteBuffer should have 0 size";
  
  // Write the number of Xs
  UInt32 size = 2;
  wb.write((UInt32 &)size);
  // Write all Xs.
  for (UInt32 i = 0; i < size; ++i)
  {
    wb.write(xi[i].a);
    wb.write(xi[i].b);
    wb.write(xi[i].c);
    Size len = 4;
    wb.write((const Byte *)xi[i].d, len);
    wb.write(xi[i].e);
  }
  
  ReadBuffer rb(wb.getData(), wb.getSize());
  // Read number of Xs
  rb.read(size);
  // Allocate array of Xs
  auto xo = new X[size];
  for (Size i = 0; i < size; ++i)
  {
    rb.read(xo[i].a);
    rb.read(xo[i].b);
    rb.read(xo[i].c);
    Size size = 4;
    Int32 res = rb.read(xo[i].d, size);
    ASSERT_TRUE(res == 0) << "BufferTest::testComplicatedSerialization(), rb.read(xi[i].d, 4) failed";
    ASSERT_TRUE(size == 4) << "BufferTest::testComplicatedSerialization(), rb.read(xi[i].d, 4) == 4";
    rb.read(xo[i].e);
    NTA_INFO << "xo[" << i << "]={" << xo[i].a << " "
             << xo[i].b << " " 
             << xo[i].c << " " 
             << "'" << std::string(xo[i].d, 4) << "'" 
             << " " << xo[i].e
             ;
  }
  
  ASSERT_TRUE(nearlyEqual(xo[0].a, nupic::Real(8.8))) << "BufferTest::testComplicatedSerialization(), xo[0].a == 8.8";
  ASSERT_TRUE(xo[0].b == 6) << "BufferTest::testComplicatedSerialization(), xo[0].b == 6";
  ASSERT_TRUE(xo[0].c == 'c') << "BufferTest::testComplicatedSerialization(), xo[0].c == 'c'";
  ASSERT_TRUE(std::string(xo[0].d, 4) == std::string("ABCD")) << "BufferTest::testComplicatedSerialization(), xo[0].d == ABCD";
  ASSERT_TRUE(nearlyEqual(xo[0].e, nupic::Real(-0.04))) << "BufferTest::testComplicatedSerialization(), xo[0].e == -0.04";

  ASSERT_TRUE(xo[1].a == nupic::Real(4.5)) << "BufferTest::testComplicatedSerialization(), xo[1].a == 4.5";
  ASSERT_TRUE(xo[1].b == 6) << "BufferTest::testComplicatedSerialization(), xo[1].b == 6";
  ASSERT_TRUE(xo[1].c == 't') << "BufferTest::testComplicatedSerialization(), xo[1].c == 't'";
  ASSERT_TRUE(std::string(xo[1].d, 4) == std::string("XBCD")) << "BufferTest::testComplicatedSerialization(), xo[1].d == XBCD";
  ASSERT_TRUE(nearlyEqual(xo[1].e, nupic::Real(3.14))) << "BufferTest::testComplicatedSerialization(), xo[1].e == 3.14";
}

TEST(BufferTest, testArrayMethods)
{
  // Test read UInt32 array
  {
    const Byte * s = "1 2 3 444";
    ReadBuffer b(s, (Size)::strlen(s));
    IReadBuffer & reader = b;   

    UInt32 result[4];
    std::fill(result, result+4, 0);
    for (auto & elem : result)
    {
      ASSERT_TRUE(elem== 0);
    }
  
    reader.read((UInt32 *)result, 3);
    for (UInt32 i = 0; i < 3; ++i)
    {
      ASSERT_TRUE(result[i] == i+1);
    }

    UInt32 val = 0;
    reader.read(val);
    ASSERT_TRUE(val == 444);
  }
  
  // Test read Int32 array
  {
    const Byte * s = "-1 -2 -3 444";
    ReadBuffer b(s, (Size)::strlen(s));
    IReadBuffer & reader = b;   

    Int32 result[4];
    std::fill(result, result+4, 0);
    for (auto & elem : result)
    {
      ASSERT_TRUE(elem== 0);
    }
  
    reader.read((Int32 *)result, 3);
    for (Int32 i = 0; i < 3; ++i)
    {
      ASSERT_TRUE(result[i] == -i-1);
    }

    Int32 val = 0;
    reader.read(val);
    ASSERT_TRUE(val == 444);
  }
  
  // Test read Real32 array
  {
    const Byte * s = "1.5 2.5 3.5 444.555";
    ReadBuffer b(s, (Size)::strlen(s));
    IReadBuffer & reader = b;   

    Real32 result[4];
    std::fill(result, result+4, (Real32)0);
    for (auto & elem : result)
    {
      ASSERT_TRUE(elem== 0);
    }
  
    reader.read((Real32 *)result, 3);
    for (UInt32 i = 0; i < 3; ++i)
    {
      ASSERT_TRUE(result[i] == i+1.5);
    }

    Real32 val = 0;
    reader.read(val);
    ASSERT_TRUE(nearlyEqual(val, Real32(444.555)));
  }
}

