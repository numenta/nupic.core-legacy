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
 * Implementation of BasicType test
 */

#include <limits>

#include <gtest/gtest.h>
#include <htm/ntypes/BasicType.hpp>
#include <htm/types/Sdr.hpp>

namespace testing {
    
using namespace htm;

TEST(BasicTypeTest, isValid)
{
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Byte));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Int16));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_UInt16));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Int32));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_UInt32));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Int64));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_UInt64));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Real32));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Real64));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Real));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Handle));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Bool));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_SDR));
  ASSERT_TRUE(BasicType::isValid(NTA_BasicType_Str));

  
  ASSERT_TRUE(!BasicType::isValid(NTA_BasicType_Last));
  ASSERT_TRUE(!(BasicType::isValid(NTA_BasicType(NTA_BasicType_Last + 777))));
  ASSERT_TRUE(!(BasicType::isValid(NTA_BasicType(-1))));
}

TEST(BasicTypeTest, getSize)
{
  // This is the size of an element of the array
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Byte) == 1);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Int16) == 2);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_UInt16) == 2);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Int32) == 4);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_UInt32) == 4);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Int64) == 8);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_UInt64) == 8);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Real32) == 4);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Real64) == 8);
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Bool) == sizeof(bool));
  #ifdef NTA_DOUBLE_PRECISION
    ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Real) == 8); // Real64
  #else
    ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Real) == 4); // Real32
  #endif
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Handle) == sizeof(void *));
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_SDR) == sizeof(char));
  ASSERT_TRUE(BasicType::getSize(NTA_BasicType_Str) == sizeof(std::string));
}

TEST(BasicTypeTest, getName)
{
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Byte) == std::string("Byte"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Int16) == std::string("Int16"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_UInt16) == std::string("UInt16"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Int32) == std::string("Int32"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_UInt32) == std::string("UInt32"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Int64) == std::string("Int64"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_UInt64) == std::string("UInt64"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real32) == std::string("Real32"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real64) == std::string("Real64"));
  #ifdef NTA_DOUBLE_PRECISION
    ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real) == std::string("Real64"));
  #else
    ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real) == std::string("Real32"));
  #endif      
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Handle) == std::string("Handle"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Bool) == std::string("Bool"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_SDR) == std::string("SDR"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Str) == std::string("String"));
}

TEST(BasicTypeTest, parse)
{
  ASSERT_TRUE(BasicType::parse("Byte") == NTA_BasicType_Byte);
  ASSERT_TRUE(BasicType::parse("Int16") == NTA_BasicType_Int16);
  ASSERT_TRUE(BasicType::parse("UInt16") == NTA_BasicType_UInt16);
  ASSERT_TRUE(BasicType::parse("Int32") == NTA_BasicType_Int32);
  ASSERT_TRUE(BasicType::parse("UInt32") == NTA_BasicType_UInt32);
  ASSERT_TRUE(BasicType::parse("Int64") == NTA_BasicType_Int64);
  ASSERT_TRUE(BasicType::parse("UInt64") == NTA_BasicType_UInt64);
  ASSERT_TRUE(BasicType::parse("Real32") == NTA_BasicType_Real32);
  ASSERT_TRUE(BasicType::parse("Real64") == NTA_BasicType_Real64);
  ASSERT_TRUE(BasicType::parse("Real") == NTA_BasicType_Real);
  ASSERT_TRUE(BasicType::parse("Handle") == NTA_BasicType_Handle);
  ASSERT_TRUE(BasicType::parse("Bool") == NTA_BasicType_Bool);
  ASSERT_TRUE(BasicType::parse("SDR") == NTA_BasicType_SDR);
  ASSERT_TRUE(BasicType::parse("std::string") == NTA_BasicType_Str);
}

class convertArrayTester {
public:
  Byte bufByte[8] = {0, 1, 2, 3, 4, 5, static_cast<Byte>(-128), 127}; //Byte is unsigned on ARM, signed on x86, therefore the cast is needed for ARM
  Int16 bufInt16[8] = {0, 1, 2, 3, 4, 5, -32768, 32767};
  UInt16 bufUInt16[8] = {0, 1, 2, 3, 4, 5, 0, 0xffff};
  Int32 bufInt32[8] = {0, 1, 2, 3, 4, 5, -2147483647L, 2147483647L};
  UInt32 bufUInt32[8] = {0, 1, 2, 3, 4, 5, 0, 0xffffffff};
  Int64 bufInt64[8] = { 0, 1, 2, 3, 4, 5, 
			std::numeric_limits<Int64>::min(), 
			std::numeric_limits<Int64>::max() };
  UInt64 bufUInt64[8] = {0, 1, 2, 3, 4, 5, 0, 0xffffffffffffffff};
  Real32 bufReal32[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
			-std::numeric_limits<Real32>::max(), 
			std::numeric_limits<Real32>::max()};
  Real64 bufReal64[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 
			-std::numeric_limits<Real64>::max(), 
			std::numeric_limits<Real32>::max()};
  bool bufBool[8] = {false, true, true, true, true, true, true, false};
  char dest[8 * sizeof(Real64)]; // make sure there is enough room for any type.

  template <typename T> 
  bool checkArray(char *buf) {
    T *a = (T *)buf;
    for (size_t i = 0; i < 6; i++) {
      if (a[i] != (T)bufByte[i]) return false;
    }
    return true;
  }
  template <typename T>
  bool checkArrayBool(char *buf) { 
    T *a = (T*)buf;
    for (size_t i = 0; i < 6; i++) {
      if (a[i] != (T)bufBool[i])
        return false;
    }
    return true;
  }
};


TEST(BasicTypeTest, convertArray) {

  convertArrayTester ca;
  //////////////////////// to Byte ////////////////////////////////////////////
  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufByte, NTA_BasicType_Byte, 8);
  EXPECT_TRUE(ca.checkArray<Byte>(ca.dest)) << "Byte to Byte conversion";
  // Note: Bytes are char, not unsigned char.

  // negative test
  EXPECT_THROW(BasicType::convertArray(ca.dest, NTA_BasicType_Byte, ca.bufInt16,
                                       NTA_BasicType_Int16, 7), std::exception);

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufInt16, NTA_BasicType_Int16, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "Int16 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufUInt16, NTA_BasicType_UInt16, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "UInt16 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufInt32, NTA_BasicType_Int32, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "Int32 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufUInt32, NTA_BasicType_UInt32, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "UInt32 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufInt64, NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "Int64 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, 
                          ca.bufUInt64, NTA_BasicType_UInt64, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "UInt64 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "Real32 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<Byte>(ca.dest)) << "Real64 to Byte conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Byte, ca.bufBool,
                          NTA_BasicType_Bool, 6);
  ASSERT_TRUE(ca.checkArrayBool<Byte>(ca.dest)) << "bool to Byte conversion";

  //////////////////////// To Int16 ////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufByte,
                          NTA_BasicType_Byte, 8);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "Byte to Int16 conversion";
  // Note: Bytes are char, not unsigned char on x86, but on ARM char is unsigned type.

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufInt16,
                          NTA_BasicType_Int16, 8);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "Int16 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufUInt16,
                          NTA_BasicType_UInt16, 6);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "UInt16 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufInt32,
                          NTA_BasicType_Int32, 6);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "Int32 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufUInt32,
                          NTA_BasicType_UInt32, 7);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "UInt32 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufInt64,
                          NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "Int64 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufUInt64,
                          NTA_BasicType_UInt64, 7);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "UInt64 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "Real32 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<Int16>(ca.dest)) << "Real64 to Int16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int16, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<Int16>(ca.dest)) << "bool to Int16 conversion";

  //////////////////////// To UInt16 ////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufByte,
                          NTA_BasicType_Byte, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "Byte to UInt16 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufInt16,
                          NTA_BasicType_Int16, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "Int16 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufUInt16,
                          NTA_BasicType_UInt16, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "UInt16 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufInt32,
                          NTA_BasicType_Int32, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "Int32 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufUInt32,
                          NTA_BasicType_UInt32, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "UInt32 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufInt64,
                          NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "Int64 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufUInt64,
                          NTA_BasicType_UInt64, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "UInt64 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "Real32 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<UInt16>(ca.dest)) << "Real64 to UInt16 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt16, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<UInt16>(ca.dest)) << "bool to UInt16 conversion";

  //////////////////////// To Int32 ///////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufByte,
                          NTA_BasicType_Byte, 8);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "Byte to Int32 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufInt16,
                          NTA_BasicType_Int16, 8);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "Int16 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufUInt16,
                          NTA_BasicType_UInt16, 6);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "UInt16 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufInt32,
                          NTA_BasicType_Int32, 8);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "Int32 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufUInt32,
                          NTA_BasicType_UInt32, 6);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "UInt32 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufInt64,
                          NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "Int64 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufUInt64,
                          NTA_BasicType_UInt64, 6);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "UInt64 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "Real32 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<Int32>(ca.dest)) << "Real64 to Int32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int32, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<Int32>(ca.dest)) << "bool to Int32 conversion";

  //////////////////////// To UInt32  //////////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufByte,
                          NTA_BasicType_Byte, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "Byte to UInt32 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufInt16,
                          NTA_BasicType_Int16, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "Int16 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufUInt16,
                          NTA_BasicType_UInt16, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "UInt16 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufInt32,
                          NTA_BasicType_Int32, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "Int32 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufUInt32,
                          NTA_BasicType_UInt32, 8);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "UInt32 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufInt64,
                          NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "Int64 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufUInt64,
                          NTA_BasicType_UInt64, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "UInt64 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "Real32 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<UInt32>(ca.dest)) << "Real64 to UInt32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt32, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<UInt32>(ca.dest)) << "bool to UInt32 conversion";

  //////////////////////// To Int64 /////////////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufByte,
                          NTA_BasicType_Byte, 8);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "Byte to Int64 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufInt16,
                          NTA_BasicType_Int16, 6);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "Int16 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufUInt16,
                          NTA_BasicType_UInt16, 8);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "UInt16 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufInt32,
                          NTA_BasicType_Int32, 8);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "Int32 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufUInt32,
                          NTA_BasicType_UInt32, 8);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "UInt32 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufInt64,
                          NTA_BasicType_Int64, 8);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "Int64 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufUInt64,
                          NTA_BasicType_UInt64, 6);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "UInt64 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "Real32 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<Int64>(ca.dest)) << "Real64 to Int64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Int64, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<Int64>(ca.dest)) << "bool to Int64 conversion";

  //////////////////////// To UInt64 ////////////////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufByte,
                          NTA_BasicType_Byte, 6);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "Byte to UInt64 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufInt16,
                          NTA_BasicType_Int16, 6);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "Int16 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufUInt16,
                          NTA_BasicType_UInt16, 8);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "UInt16 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufInt32,
                          NTA_BasicType_Int32, 6);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "Int32 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufUInt32,
                          NTA_BasicType_UInt32, 8);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "UInt32 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufInt64,
                          NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "Int64 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufUInt64,
                          NTA_BasicType_UInt64, 8);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest))  << "UInt64 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufReal32,
                          NTA_BasicType_Real32, 6);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "Real32 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<UInt64>(ca.dest)) << "Real64 to UInt64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_UInt64, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<UInt64>(ca.dest)) << "bool to UInt64 conversion";

  //////////////////////// To Real32 ///////////////////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufByte,
                          NTA_BasicType_Byte, 8);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "Byte to Real32 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufInt16,
                          NTA_BasicType_Int16, 8);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "Int16 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufUInt16,
                          NTA_BasicType_UInt16, 8);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "UInt16 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufInt32,
                          NTA_BasicType_Int32, 8);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "Int32 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufUInt32,
                          NTA_BasicType_UInt32, 8);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "UInt32 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufInt64,
                          NTA_BasicType_Int64, 6);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "Int64 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufUInt64,
                          NTA_BasicType_UInt64, 6);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "UInt64 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufReal32,
                          NTA_BasicType_Real32, 8);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "Real32 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufReal64,
                          NTA_BasicType_Real64, 6);
  ASSERT_TRUE(ca.checkArray<Real32>(ca.dest)) << "Real64 to Real32 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real32, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<Real32>(ca.dest)) << "bool to Real32 conversion";

  //////////////////////// To Real64 //////////////////////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufByte,
                          NTA_BasicType_Byte, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "Byte to Real64 conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufInt16,
                          NTA_BasicType_Int16, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "Int16 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufUInt16,
                          NTA_BasicType_UInt16, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "UInt16 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufInt32,
                          NTA_BasicType_Int32, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "Int32 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufUInt32,
                          NTA_BasicType_UInt32, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "UInt32 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufInt64,
                          NTA_BasicType_Int64, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "Int64 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufUInt64,
                          NTA_BasicType_UInt64, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "UInt64 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufReal32,
                          NTA_BasicType_Real32, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "Real32 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufReal64,
                          NTA_BasicType_Real64, 8);
  ASSERT_TRUE(ca.checkArray<Real64>(ca.dest)) << "Real64 to Real64 conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Real64, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<Real64>(ca.dest)) << "bool to Real64 conversion";

  //////////////////////// To Bool /////////////////////////////////////////////////////////////////

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufByte,
                          NTA_BasicType_Byte, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "Byte to bool conversion";
  // Note: Bytes are char, not unsigned char.

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufInt16,
                          NTA_BasicType_Int16, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "Int16 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufUInt16,
                          NTA_BasicType_UInt16, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "UInt16 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufInt32,
                          NTA_BasicType_Int32, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "Int32 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufUInt32,
                          NTA_BasicType_UInt32, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "UInt32 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufInt64,
                          NTA_BasicType_Int64, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "Int64 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufUInt64,
                          NTA_BasicType_UInt64, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "UInt64 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufReal32,
                          NTA_BasicType_Real32, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "Real32 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufReal64,
                          NTA_BasicType_Real64, 8);
  ASSERT_TRUE(ca.checkArray<bool>(ca.dest)) << "Real64 to bool conversion";

  BasicType::convertArray(ca.dest, NTA_BasicType_Bool, ca.bufBool,
                          NTA_BasicType_Bool, 8);
  ASSERT_TRUE(ca.checkArrayBool<bool>(ca.dest)) << "bool to bool conversion";
}
}
