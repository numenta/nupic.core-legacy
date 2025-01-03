/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Implementation of BasicType test
 */

#include <gtest/gtest.h>
#include <nupic/types/BasicType.hpp>

using namespace nupic;

TEST(BasicTypeTest, isValid) {
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

  ASSERT_TRUE(!BasicType::isValid(NTA_BasicType_Last));
  ASSERT_TRUE(!(BasicType::isValid(NTA_BasicType(NTA_BasicType_Last + 777))));
  ASSERT_TRUE(!(BasicType::isValid(NTA_BasicType(-1))));
}

TEST(BasicTypeTest, getSize) {
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
}

TEST(BasicTypeTest, getName) {
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Byte) == std::string("Byte"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Int16) == std::string("Int16"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_UInt16) ==
              std::string("UInt16"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Int32) == std::string("Int32"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_UInt32) ==
              std::string("UInt32"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Int64) == std::string("Int64"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_UInt64) ==
              std::string("UInt64"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real32) ==
              std::string("Real32"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real64) ==
              std::string("Real64"));
#ifdef NTA_DOUBLE_PRECISION
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real) == std::string("Real64"));
#else
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Real) == std::string("Real32"));
#endif
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Handle) ==
              std::string("Handle"));
  ASSERT_TRUE(BasicType::getName(NTA_BasicType_Bool) == std::string("Bool"));
}

TEST(BasicTypeTest, parse) {
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
}
