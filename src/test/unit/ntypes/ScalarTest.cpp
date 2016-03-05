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
 * Implementation of Scalar test
 */

#include <nupic/ntypes/Scalar.hpp>
#include <gtest/gtest.h>

using namespace nupic;

TEST(ScalarTest, All)
{
  Scalar a(NTA_BasicType_UInt16);

  //Test UInt16
  a = Scalar(NTA_BasicType_UInt16);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ((UInt16)0, a.getValue<UInt16>());
  ASSERT_EQ(NTA_BasicType_UInt16, a.getType());
  a.value.uint16 = 10;
  ASSERT_EQ((UInt16)10, a.getValue<UInt16>());
  
  //Test UInt32
  a = Scalar(NTA_BasicType_UInt32);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ((UInt32)0, a.getValue<UInt32>());
  ASSERT_EQ(NTA_BasicType_UInt32, a.getType());
  a.value.uint32 = 10;
  ASSERT_EQ((UInt32)10, a.getValue<UInt32>());

  //Test UInt64
  a = Scalar(NTA_BasicType_UInt64);
  ASSERT_ANY_THROW(a.getValue<UInt32>());
  ASSERT_EQ((UInt64)0, a.getValue<UInt64>());
  ASSERT_EQ(NTA_BasicType_UInt64, a.getType());
  a.value.uint64 = 10;
  ASSERT_EQ((UInt64)10, a.getValue<UInt64>());
  
  //Test Int16
  a = Scalar(NTA_BasicType_Int16);
  ASSERT_ANY_THROW(a.getValue<Int32>());
  ASSERT_EQ((Int16)0, a.getValue<Int16>());
  ASSERT_EQ(NTA_BasicType_Int16, a.getType());
  a.value.int16 = 10;
  ASSERT_EQ((Int16)10, a.getValue<Int16>());

  //Test Int32
  a = Scalar(NTA_BasicType_Int32);
  ASSERT_ANY_THROW(a.getValue<Int64>());
  ASSERT_EQ((Int32)0, a.getValue<Int32>());
  ASSERT_EQ(NTA_BasicType_Int32, a.getType());
  a.value.int32 = 10;
  ASSERT_EQ((Int32)10, a.getValue<Int32>());

  //Test Int64
  a = Scalar(NTA_BasicType_Int64);
  ASSERT_ANY_THROW(a.getValue<UInt32>());
  ASSERT_EQ((Int64)0, a.getValue<Int64>());
  ASSERT_EQ(NTA_BasicType_Int64, a.getType());
  a.value.int64 = 10;
  ASSERT_EQ((Int64)10, a.getValue<Int64>());

  //Test Real32
  a = Scalar(NTA_BasicType_Real32);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ((Real32)0, a.getValue<Real32>());
  ASSERT_EQ(NTA_BasicType_Real32, a.getType());
  a.value.real32 = 10;
  ASSERT_EQ((Real32)10, a.getValue<Real32>());

  //Test Real64
  a = Scalar(NTA_BasicType_Real64);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ((Real64)0, a.getValue<Real64>());
  ASSERT_EQ(NTA_BasicType_Real64, a.getType());
  a.value.real64 = 10;
  ASSERT_EQ((Real64)10, a.getValue<Real64>());

  //Test Handle
  a = Scalar(NTA_BasicType_Handle);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ((Handle)nullptr, a.getValue<Handle>());
  ASSERT_EQ(NTA_BasicType_Handle, a.getType());
  int x = 10;
  a.value.handle = &x;
  int* p = (int*)(a.getValue<Handle>());
  ASSERT_EQ(&x, a.getValue<Handle>());
  ASSERT_EQ(x, *p);
  (*p)++;
  ASSERT_EQ(11, *p);
  
  //Test Byte
  a = Scalar(NTA_BasicType_Byte);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ((Byte)0, a.getValue<Byte>());
  ASSERT_EQ(NTA_BasicType_Byte, a.getType());
  a.value.byte = 'a';
  ASSERT_EQ('a', a.getValue<Byte>());
  a.value.byte++;
  ASSERT_EQ('b', a.getValue<Byte>());

  //Test Bool
  a = Scalar(NTA_BasicType_Bool);
  ASSERT_ANY_THROW(a.getValue<UInt64>());
  ASSERT_EQ(false, a.getValue<bool>());
  ASSERT_EQ(NTA_BasicType_Bool, a.getType());
  a.value.boolean = true;
  ASSERT_EQ(true, a.getValue<bool>());
  a.value.boolean = false;
  ASSERT_EQ(false, a.getValue<bool>());
}
