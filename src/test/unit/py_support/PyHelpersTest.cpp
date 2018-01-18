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
 * Implementation of PyHelpers test
 */

#include <nupic/py_support/PyHelpers.hpp>
#include <limits>
#include <gtest/gtest.h>

using namespace nupic;

class PyHelpersTest : public ::testing::Test
{ 
  public:
  PyHelpersTest()
  {
    NTA_DEBUG << "Py_Initialize()";
    Py_Initialize();
  }

  ~PyHelpersTest()
  {
    NTA_DEBUG << "Py_Finalize()";
    Py_Finalize();
  }
};


TEST_F(PyHelpersTest, pyPtrConstructionNULL)
{
  PyObject * p  = NULL;
  EXPECT_THROW(py::Ptr(p, /* allowNULL: */false), std::exception);

  py::Ptr pp1(p, /* allowNULL: */true);
  ASSERT_TRUE((PyObject *)pp1 == NULL);
  ASSERT_TRUE(pp1.isNULL());
}

TEST_F(PyHelpersTest, pyPtrConstructionNonNULL)
{
  PyObject * p = PyTuple_New(1);
  py::Ptr pp2(p);
  ASSERT_TRUE(!pp2.isNULL());
  ASSERT_TRUE((PyObject *)pp2 == p);
  pp2.release();
  ASSERT_TRUE(pp2.isNULL());
  Py_DECREF(p);
}
  
TEST_F(PyHelpersTest, pyPtrConstructionAssign)
{
  PyObject * p = PyTuple_New(1);
  ASSERT_TRUE(p->ob_refcnt == 1);
  py::Ptr pp(NULL, /* allowNULL */ true);
  ASSERT_TRUE(pp.isNULL());
  NTA_DEBUG << "*** Before assign";
  pp.assign(p);
  NTA_DEBUG << "*** After assign";
  ASSERT_TRUE(p->ob_refcnt == 2);
  ASSERT_TRUE(!(pp.isNULL()));
  Py_DECREF(p);
  ASSERT_TRUE(p->ob_refcnt == 1);
}

TEST_F(PyHelpersTest, pyString)
{
  py::String ps1(std::string("123"));
  ASSERT_TRUE(PyString_Check(ps1) != 0);

  py::String ps2("123", size_t(3));
  ASSERT_TRUE(PyString_Check(ps2) != 0);

  py::String ps3("123");
  ASSERT_TRUE(PyString_Check(ps3) != 0);

  std::string s1(PyString_AsString(ps1));
  std::string s2(PyString_AsString(ps2));
  std::string s3(PyString_AsString(ps3));
  std::string expected("123");
  ASSERT_TRUE(s1 == expected);
  ASSERT_TRUE(s2 == expected);
  ASSERT_TRUE(s3 == expected);

  ASSERT_TRUE(std::string(ps1) == expected);
  ASSERT_TRUE(std::string(ps2) == expected);
  ASSERT_TRUE(std::string(ps3) == expected);

  PyObject * p = PyString_FromString("777");
  py::String ps4(p);
  ASSERT_TRUE(std::string(ps4) == std::string("777"));
}

TEST_F(PyHelpersTest, pyInt)
{
  py::Int n1(-5);
  py::Int n2(-6666);
  py::Int n3(long(0));
  py::Int n4(555);
  py::Int n5(6666);
  
  ASSERT_TRUE(n1 == -5);
  int x = n2; 
  int expected = -6666;
  ASSERT_TRUE(x == expected);
  ASSERT_TRUE(n3 == 0);
  ASSERT_TRUE(n4 == 555);
  x = n5;
  expected = 6666;
  ASSERT_TRUE(x == expected);
}

TEST_F(PyHelpersTest, pyLong)
{
  py::Long n1(-5);
  py::Long n2(-66666666);
  py::Long n3(long(0));
  py::Long n4(555);
  py::Long n5(66666666);
  
  ASSERT_TRUE(n1 == -5);
  long x = n2; 
  long expected = -66666666;
  ASSERT_TRUE(x == expected);
  ASSERT_TRUE(n3 == 0);
  ASSERT_TRUE(n4 == 555);
  x = n5;
  expected = 66666666;
  ASSERT_TRUE(x == expected);
}

TEST_F(PyHelpersTest, pyUnsignedLong)
{
  py::UnsignedLong n1((unsigned long)(-5));
  py::UnsignedLong n2((unsigned long)(-66666666));
  py::UnsignedLong n3((unsigned long)(0));
  py::UnsignedLong n4(555);
  py::UnsignedLong n5(66666666);
  
  ASSERT_TRUE(n1 == (unsigned long)(-5));
  ASSERT_TRUE(n2 == (unsigned long)(-66666666));
  ASSERT_TRUE(n3 == 0);
  ASSERT_TRUE(n4 == 555);
  ASSERT_TRUE(n5 == 66666666);
}

TEST_F(PyHelpersTest, pyFloat)
{
  ASSERT_TRUE(py::Float::getMax() == std::numeric_limits<double>::max());
  ASSERT_TRUE(py::Float::getMin() == std::numeric_limits<double>::min());

  py::Float max(std::numeric_limits<double>::max());
  py::Float min(std::numeric_limits<double>::min());
  py::Float n1(-0.5);
  py::Float n2(double(0));
  py::Float n3(333.555);
  py::Float n4(0.02);
  py::Float n5("0.02");
  
  ASSERT_TRUE(max == py::Float::getMax());
  ASSERT_TRUE(min == py::Float::getMin());
  ASSERT_TRUE(n1 == -0.5);
  ASSERT_TRUE(n2 == 0);
  ASSERT_TRUE(n3 == 333.555);
  ASSERT_TRUE(n4 == 0.02);
  ASSERT_TRUE(n5 == 0.02);
}

TEST_F(PyHelpersTest, pyBool)
{
  const auto trueRefcount = Py_REFCNT(Py_True);
  const auto falseRefcount = Py_REFCNT(Py_False);

  // Construct from true.
  {
    py::Bool t(true);
    ASSERT_TRUE((bool)t);
  }

  // Verify refcounts were preserved.
  ASSERT_EQ(trueRefcount, Py_REFCNT(Py_True));
  ASSERT_EQ(falseRefcount, Py_REFCNT(Py_False));

  // Construct from false.
  {
    py::Bool f(false);
    ASSERT_FALSE((bool)f);
  }

  // Verify refcounts were preserved.
  ASSERT_EQ(trueRefcount, Py_REFCNT(Py_True));
  ASSERT_EQ(falseRefcount, Py_REFCNT(Py_False));

  // Construct from an existing PyObject.
  {
    Py_XINCREF(Py_True);
    py::Bool t(Py_True);
    ASSERT_TRUE((bool)t);
  }

  // Verify refcounts were preserved.
  ASSERT_EQ(trueRefcount, Py_REFCNT(Py_True));
  ASSERT_EQ(falseRefcount, Py_REFCNT(Py_False));
}

TEST_F(PyHelpersTest, pyTupleEmpty)
{
  py::String s1("item_1");
  py::String s2("item_2");
  
  py::Tuple empty;
  ASSERT_TRUE(PyTuple_Check(empty) != 0);
  ASSERT_TRUE(empty.getCount() == 0);
  
  EXPECT_THROW(empty.setItem(0, s1), std::exception);
  EXPECT_THROW(empty.getItem(0), std::exception);
}

TEST_F(PyHelpersTest, pyTupleOneItem)
{
  py::String s1("item_1");
  py::String s2("item_2");

  py::Tuple t1(1);
  ASSERT_TRUE(PyTuple_Check(t1) != 0);
  ASSERT_TRUE(t1.getCount() == 1);

  t1.setItem(0, s1);
  py::String item1(t1.getItem(0));
  ASSERT_TRUE(std::string(item1) == std::string(s1));
  
  py::String fastItem1(t1.fastGetItem(0));
  ASSERT_TRUE(std::string(fastItem1) == std::string(s1));
  fastItem1.release();
  
  EXPECT_THROW(t1.setItem(1, s2), std::exception);
  EXPECT_THROW(t1.getItem(1), std::exception);

  ASSERT_TRUE(t1.getCount() == 1);
}

TEST_F(PyHelpersTest, pyTupleTwoItems)
{
  py::String s1("item_1");
  py::String s2("item_2");

  py::Tuple t2(2);
  ASSERT_TRUE(PyTuple_Check(t2) != 0);
  ASSERT_TRUE(t2.getCount() == 2);

  t2.setItem(0, s1);
  py::String item1(t2.getItem(0));
  ASSERT_TRUE(std::string(item1) == std::string(s1));
  py::String fastItem1(t2.fastGetItem(0));
  ASSERT_TRUE(std::string(fastItem1) == std::string(s1));
  fastItem1.release();

  t2.setItem(1, s2);
  py::String item2(t2.getItem(1));
  ASSERT_TRUE(std::string(item2) == std::string(s2));
  py::String fastItem2(t2.fastGetItem(1));
  ASSERT_TRUE(std::string(fastItem2) == std::string(s2));
  fastItem2.release();


  EXPECT_THROW(t2.setItem(2, s2), std::exception);
  EXPECT_THROW(t2.getItem(2), std::exception);

  ASSERT_TRUE(t2.getCount() == 2);
}


TEST_F(PyHelpersTest, pyListEmpty)
{
  py::String s1("item_1");
  py::String s2("item_2");

  py::List empty;
  ASSERT_TRUE(PyList_Check(empty) != 0);
  ASSERT_TRUE(empty.getCount() == 0);
  
  EXPECT_THROW(empty.setItem(0, s1), std::exception);
  EXPECT_THROW(empty.getItem(0), std::exception);
}

TEST_F(PyHelpersTest, pyListOneItem)
{
  py::String s1("item_1");
  py::String s2("item_2");

  py::List t1;
  ASSERT_TRUE(PyList_Check(t1) != 0);
  ASSERT_TRUE(t1.getCount() == 0);

  t1.append(s1);
  py::String item1(t1.getItem(0));
  ASSERT_TRUE(std::string(item1) == std::string(s1));
  py::String fastItem1(t1.fastGetItem(0));
  ASSERT_TRUE(std::string(fastItem1) == std::string(s1));
  fastItem1.release();

  ASSERT_TRUE(t1.getCount() == 1);
  ASSERT_TRUE(std::string(item1) == std::string(s1));
  
  EXPECT_THROW(t1.getItem(1), std::exception);
}

TEST_F(PyHelpersTest, pyListTwoItems)
{
  py::String s1("item_1");
  py::String s2("item_2");

  py::List t2;
  ASSERT_TRUE(PyList_Check(t2) != 0);
  ASSERT_TRUE(t2.getCount() == 0);

  t2.append(s1);
  py::String item1(t2.getItem(0));
  ASSERT_TRUE(std::string(item1) == std::string(s1));
  py::String fastItem1(t2.fastGetItem(0));
  ASSERT_TRUE(std::string(fastItem1) == std::string(s1));
  fastItem1.release();

  t2.append(s2);
  ASSERT_TRUE(t2.getCount() == 2);
  
  py::String item2(t2.getItem(1));
  ASSERT_TRUE(std::string(item2) == std::string(s2));
  py::String fastItem2(t2.fastGetItem(1));
  ASSERT_TRUE(std::string(fastItem2) == std::string(s2));
  fastItem2.release();


  EXPECT_THROW(t2.getItem(2), std::exception);
}

TEST_F(PyHelpersTest, pyDictEmpty)
{
  py::Dict d;
  ASSERT_EQ(PyDict_Size(d), 0);

  ASSERT_TRUE(d.getItem("blah") == NULL);
}

TEST_F(PyHelpersTest, pyDictExternalPyObjectFailed)
{
  // NULL object
  EXPECT_THROW(py::Dict(NULL), std::exception);

  // Wrong type (must be a dictionary)
  py::String s("1234");
  try
  {
    py::Dict d(s.release());
    NTA_THROW << "py::Dict d(s) Should fail!!!";
  }
  catch(...)
  {
  }
  // SHOULDFAIL fails to fail :-)
  //SHOULDFAIL(py::Dict(s));
}

TEST_F(PyHelpersTest, pyDictExternalPyObjectSuccessful)
{
  PyObject * p = PyDict_New();
  PyDict_SetItem(p, py::String("1234"), py::String("5678"));
  
  py::Dict d(p);

  ASSERT_TRUE(PyDict_Contains(d, py::String("1234")) == 1);

  PyDict_SetItem(d, py::String("777"), py::String("999"));

  ASSERT_TRUE(PyDict_Contains(d, py::String("777")) == 1);
}
  
// getItem with default (exisiting and non-exisitng key)
TEST_F(PyHelpersTest, pyDictGetItem)
{
  py::Dict d;
  d.setItem("A", py::String("AAA"));

  PyObject * defaultItem = (PyObject *)123;
  
  py::String A(d.getItem("A"));             
  ASSERT_TRUE(std::string(A) == std::string("AAA"));

  // No "B" in the dict, so expect to get the default item
  PyObject * B = (d.getItem("B", defaultItem));
  ASSERT_TRUE(B == defaultItem);

  PyDict_SetItem(d, py::String("777"), py::String("999"));
  ASSERT_TRUE(PyDict_Contains(d, py::String("777")) == 1);
}
  

TEST_F(PyHelpersTest, pyModule)
{
  py::Module module("sys");
  ASSERT_TRUE(std::string(PyModule_GetName(module)) == std::string("sys"));
}

TEST_F(PyHelpersTest, pyClass)
{
  py::Class c("datetime", "date");
}

TEST_F(PyHelpersTest, pyInstance)
{
  
  py::Tuple args(3);
  args.setItem(0, py::Long(2000));
  args.setItem(1, py::Long(11));
  args.setItem(2, py::Long(5));
  py::Instance date("datetime", "date", args, py::Dict());

  // Test invoke()
  {
    py::String result(date.invoke("__str__", py::Tuple(), py::Dict()));
    std::string res((const char *)result);
    std::string expected("2000-11-05");
    ASSERT_TRUE(res == expected);
  }

  // Test hasAttr()
  {
    py::String result(date.invoke("__str__", py::Tuple(), py::Dict()));
    std::string res((const char *)result);
    std::string expected("2000-11-05");
    ASSERT_TRUE(!(date.hasAttr("No such attribute")));
    ASSERT_TRUE(date.hasAttr("year"));
  }

  // Test getAttr()
  {
    py::Int year(date.getAttr("year"));
    ASSERT_TRUE(2000 == long(year));
  }

  // Test toString()
  {
    std::string res((const char *)py::String(date.toString()));
    std::string expected("2000-11-05");
    ASSERT_TRUE(res == expected);
  }
}

TEST_F(PyHelpersTest, pyCustomException)
{
  py::Tuple args(1);
  args.setItem(0, py::String("error message!"));
  py::Instance e(PyExc_RuntimeError, args);
  e.setAttr("traceback", py::String("traceback!!!"));

  PyErr_SetObject(PyExc_RuntimeError, e);

  try
  {
    py::checkPyError(0);
  }
  catch (const nupic::Exception & e)
  {
    NTA_DEBUG << e.getMessage();
  }
}
