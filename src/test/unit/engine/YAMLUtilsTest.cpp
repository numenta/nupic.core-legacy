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
 * Implementation of YAMLUtils test
 */

#include "gtest/gtest.h"
#include <nupic/engine/Network.hpp>
#include <nupic/engine/YAMLUtils.hpp>
#include <nupic/engine/Spec.hpp>

using namespace nupic;

TEST(YAMLUtilsTest, toValueTestInt)
{
  const char* s1 = "10";
  Value v = YAMLUtils::toValue(s1, NTA_BasicType_Int32);
  EXPECT_TRUE(v.isScalar()) << "assertion v.isScalar() failed at "
    << __FILE__ << ":" << __LINE__ ;
  ASSERT_EQ(v.getType(), NTA_BasicType_Int32);
  Int32 i = v.getScalarT<Int32>();
  ASSERT_EQ(10, i);
  boost::shared_ptr<Scalar> s = v.getScalar();
  i = s->value.int32;
  ASSERT_EQ(10, i);
}

TEST(YAMLUtilsTest, toValueTestReal32)
{
  const char* s1 = "10.1";
  Value v = YAMLUtils::toValue(s1, NTA_BasicType_Real32);
  EXPECT_TRUE(v.isScalar()) << "assertion v.isScalar() failed at "
    << __FILE__ << ":" << __LINE__ ;
  ASSERT_EQ(v.getType(), NTA_BasicType_Real32);
  Real32 x = v.getScalarT<Real32>();
  EXPECT_NEAR(10.1, x, 0.000001) << "assertion 10.1 == " << x
    << "\" failed at " << __FILE__ << ":" << __LINE__;

  boost::shared_ptr<Scalar> s = v.getScalar();
  x = s->value.real32;
  EXPECT_NEAR(10.1, x, 0.000001) << "assertion 10.1 == " << x
    << "\" failed at " << __FILE__ << ":" << __LINE__;
}

TEST(YAMLUtilsTest, toValueTestByte)
{
  const char* s1 = "this is a string";
  Value v = YAMLUtils::toValue(s1, NTA_BasicType_Byte);
  EXPECT_TRUE(!v.isScalar()) << "assertion !v.isScalar() failed at "
    << __FILE__ << ":" << __LINE__ ;
  EXPECT_TRUE(v.isString()) << "assertion v.isScalar() failed at "
    << __FILE__ << ":" << __LINE__ ;
  ASSERT_EQ(v.getType(), NTA_BasicType_Byte);
  std::string s = *v.getString();
  EXPECT_STREQ(s1, s.c_str());
}

TEST(YAMLUtilsTest, toValueTestBool)
{
  const char* s1 = "true";
  Value v = YAMLUtils::toValue(s1, NTA_BasicType_Bool);
  EXPECT_TRUE(v.isScalar()) << "assertion v.isScalar() failed at "
                            << __FILE__ << ":" << __LINE__ ;
  ASSERT_EQ(v.getType(), NTA_BasicType_Bool);
  bool b = v.getScalarT<bool>();
  ASSERT_EQ(true, b);
  boost::shared_ptr<Scalar> s = v.getScalar();
  b = s->value.boolean;
  ASSERT_EQ(true, b);
}

TEST(YAMLUtilsTest, ParameterSpec)
{
  Collection<ParameterSpec> ps;
  ps.add(
    "int32Param", 
    ParameterSpec(
      "Int32 scalar parameter",  // description
      NTA_BasicType_Int32,
      1,                         // elementCount
      "",                        // constraints
      "32",                      // defaultValue
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "uint32Param", 
    ParameterSpec(
      "UInt32 scalar parameter", // description
      NTA_BasicType_UInt32, 
      1,                         // elementCount
      "",                        // constraints
      "33",                      // defaultValue
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "int64Param", 
    ParameterSpec(
      "Int64 scalar parameter",  // description
      NTA_BasicType_Int64, 
      1,                         // elementCount
      "",                        // constraints
      "64",                       // defaultValue
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "uint64Param", 
    ParameterSpec(
      "UInt64 scalar parameter", // description
      NTA_BasicType_UInt64,
      1,                         // elementCount
      "",                        // constraints
      "65",                       // defaultValue
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "real32Param", 
    ParameterSpec(
      "Real32 scalar parameter",  // description
      NTA_BasicType_Real32,
      1,                         // elementCount
      "",                        // constraints
      "32.1",                    // defaultValue
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "real64Param", 
    ParameterSpec(
      "Real64 scalar parameter",  // description
      NTA_BasicType_Real64,
      1,                         // elementCount
      "",                        // constraints
      "64.1",                    // defaultValue
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "real32ArrayParam",
    ParameterSpec(
      "int32 array parameter", 
      NTA_BasicType_Real32,
      0, // array
      "", 
      "",
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "int64ArrayParam",
    ParameterSpec(
      "int64 array parameter", 
      NTA_BasicType_Int64,
      0, // array
      "", 
      "",
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "computeCallback",
    ParameterSpec(
      "address of a function that is called at every compute()",
      NTA_BasicType_Handle, 
      1,
      "", 
      "",  // handles must not have a default value
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "stringParam", 
    ParameterSpec(
      "string parameter", 
      NTA_BasicType_Byte, 
      0, // length=0 required for strings
      "", 
      "default value", 
      ParameterSpec::ReadWriteAccess));

  ps.add(
    "boolParam",
    ParameterSpec(
      "bool parameter",
      NTA_BasicType_Bool,
      1,
      "",
      "false",
      ParameterSpec::ReadWriteAccess));

  NTA_DEBUG << "ps count: " << ps.getCount();

  ValueMap vm = YAMLUtils::toValueMap("", ps);
  EXPECT_TRUE(vm.contains("int32Param")) 
    << "assertion vm.contains(\"int32Param\") failed at "
    << __FILE__ << ":" << __LINE__ ;
  ASSERT_EQ((Int32)32, vm.getScalarT<Int32>("int32Param"));

  EXPECT_TRUE(vm.contains("boolParam"))
    << "assertion vm.contains(\"boolParam\") failed at "
    << __FILE__ << ":" << __LINE__ ;
  ASSERT_EQ(false, vm.getScalarT<bool>("boolParam"));

  // disabled until we fix default string params
  // TEST(vm.contains("stringParam"));
  // EXPECT_STREQ("default value", vm.getString("stringParam")->c_str());

  // Test error message in case of invalid parameter with and without nodeType and regionName    
  try
  {
    YAMLUtils::toValueMap("{ blah: True }", ps, "nodeType", "regionName");
  }
  catch (nupic::Exception & e)
  {
    std::string s("Unknown parameter 'blah' for region 'regionName'");
    EXPECT_TRUE(std::string(e.getMessage()).find(s) == 0)
      << "assertion std::string(e.getMessage()).find(s) == 0 failed at "
      << __FILE__ << ":" << __LINE__ ;
  }

  try
  {
    YAMLUtils::toValueMap("{ blah: True }", ps);
  }
  catch (nupic::Exception & e)
  {
    std::string s("Unknown parameter 'blah'\nValid");
    EXPECT_TRUE(std::string(e.getMessage()).find(s) == 0)
      << "assertion std::string(e.getMessage()).find(s) == 0 failed at "
      << __FILE__ << ":" << __LINE__ ;
  }
}
