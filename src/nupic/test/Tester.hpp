/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file 
 * Declarations for unit tester interface.
 */

#ifndef NTA_TESTER_HPP
#define NTA_TESTER_HPP

//----------------------------------------------------------------------

#include "gtest/gtest.h"
#include <sstream>
#include <vector>
#include <cmath> // fabs
#include <nupic/utils/Log.hpp>

//----------------------------------------------------------------------

namespace nupic {

  /** 
   * 
   * Abstract base class for unit testers.
   *
   * This class specifies a simple interface for unit testing all Numenta classes.
   * A unit test is created by subclassing from Tester and implementing RunTests().
   *
   * @deprecated This class is preserved to ease the transition from old test framework
   * to Google Test framework. Should be removed eventually and replaced by 
   * Google Test macros eventually.
   * 
   */
  class Tester  {
  public:

    /** Default constructor	 */
    Tester();

    virtual ~Tester();
    
    /**
     * This is the main method that subclasses should implement.
     *
     * It is expected that this method will thoroughly test the target class by calling
     * each method and testing boundary conditions. 
     * 
     * @todo change capitalization. Must be changed in all tests. 
     * 
     */
    
    virtual void RunTests() {}
      
    static void init();

    /*
     * All tests have access to a standard input dir (read only) and output dir
     * Make these static so that they can be accessed even if you don't have a tester object.
     */
    static std::string testInputDir_;
    static std::string testOutputDir_;
    static std::string fromTestInputDir(const std::string& path);
    static std::string fromTestOutputDir(const std::string& path);

  private:
    
    // Default copy ctor and assignment operator forbidden by default
    Tester(const Tester&);
    Tester& operator=(const Tester&);
    
  }; // end class Tester
  

} // end namespace nupic

/**
 * A proxy macro to Google Test macro `EXPECT_EQ`.
 * 
 * See 
 * https://code.google.com/p/googletest/wiki/V1_7_Primer#Binary_Comparison
 * for documentation.
 * 
 * @param  expected
 *         The expected value
 * @param  actual  
 *         The actual value
 *
 * @note You should NOT use this macro to ensure the equality of C-style strings
 * (i.e. char *, wchart_t *), use `EXPECT_STREQ` in Google Test instead, see
 * https://code.google.com/p/googletest/wiki/V1_7_Primer#String_Comparison
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_EQ` instead.
 * 
 */
#define TESTEQUAL(expected, actual) \
  EXPECT_EQ(expected, actual)

/**
 * A proxy macro to Google Test macro `EXPECT_EQ`, also adds the test name, failed
 * file, line etc. 
 * 
 * See https://code.google.com/p/googletest/wiki/V1_7_Primer#Binary_Comparison
 * for documentation.
 * 
 * @param  expected
 *         The expected value
 * @param  actual
 *         The actual value
 *
 * @note You should NOT use this macro to ensure the equality of C-style strings
 * (i.e. char *, wchart_t *), use `EXPECT_STREQ` in Google Test instead, see
 * https://code.google.com/p/googletest/wiki/V1_7_Primer#String_Comparison
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_EQ` instead.
 * 
 */
#define TESTEQUAL2(name, expected, actual) \
  EXPECT_EQ(expected, actual) << "assertion \"" #name "\" failed at " << __FILE__ << ":" << __LINE__ 

/**
 * A proxy macro to Google Test macro `EXPECT_STREQ`, also adds the test name, failed
 * file, line etc. 
 * 
 * See https://code.google.com/p/googletest/wiki/V1_7_Primer#String_Comparison
 * for documentation.
 * 
 * @param  expected
 *         The expected value
 * @param  actual
 *         The actual value * 
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_STREQ` instead.
 * 
 */
#define TESTEQUAL2_STR(name, expected, actual) \
  EXPECT_STREQ(expected, actual) << "assertion \"" #name "\" failed at " << __FILE__ << ":" << __LINE__ 

/**
 * A proxy macro to Google Test macro `EXPECT_NEAR`. 
 * 
 * See https://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide#Floating-Point_Macros
 * for documentation.
 * 
 * @param  expected
 *         The expected value
 * @param  actual
 *         The actual value
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_NEAR` instead.
 * 
 */
#define TESTEQUAL_FLOAT(expected, actual) \
  EXPECT_NEAR(expected, actual, 0.000001) << "assertion \"" #expected " == " #actual << "\" failed at " << __FILE__ << ":" << __LINE__ 

/*
 * This line will undef the `TEST` macro defined in Google Test, only to avoid conflict 
 * with `TEST` macro semantic in old test framework
 *
 * @todo rewrite old tests not to use `TEST`
 */
#undef TEST

/**
 * A proxy macro to Google Test macro `EXPECT_TRUE`. 
 * 
 * See https://code.google.com/p/googletest/wiki/V1_7_Primer#Basic_Assertions
 * for documentation.
 * 
 * @param  condition 
 *         The condition expected to be true
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_TRUE` instead.
 * 
 */
#define TEST(condition) \
  EXPECT_TRUE(condition) << "assertion \"" #condition "\" failed at " << __FILE__ << ":" << __LINE__ 

/**
 * A proxy macro to Google Test macro `EXPECT_TRUE`, also adds the test name, failed
 * file, line etc. 
 *  
 * See https://code.google.com/p/googletest/wiki/V1_7_Primer#Basic_Assertions
 * for documentation.
 * 
 * @param  condition 
 *         The condition expected to be true
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_TRUE` instead.
 * 
 */
#define TEST2(name, condition) \
  EXPECT_TRUE(condition) << "assertion \"" #name ": " #condition "\" failed at " << __FILE__ << ":" << __LINE__ 


/**
 * A proxy macro to Google Test macro `EXPECT_THROW`. 
 *  
 * See http://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide#Exception_Assertions
 * for documentation.
 * 
 * @param  condition 
 *         The condition expected to be true
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_THROW` instead.
 * 
 */
#define SHOULDFAIL(statement) \
  EXPECT_THROW(statement, std::exception)

/**
 * Similar to Google Test macro `EXPECT_THROW`, but verify the exception to be type of
 * nupic::LoggingException, and also check the message.
 *  
 * See http://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide#Exception_Assertions
 * for documentation.
 * 
 * @param  condition 
 *         The condition expected to be true
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. use Google Test macro `EXPECT_THROW` instead.
 * 
 */
#define SHOULDFAIL_WITH_MESSAGE(statement, message) \
  { \
    bool caughtException = false; \
    try { \
      statement; \
    } catch(nupic::LoggingException& e) { \
      caughtException = true; \
      EXPECT_STREQ(message, e.getMessage()) << "statement '" #statement "' should fail with message \"" \
      << message << "\", but failed with message \"" << e.getMessage() << "\""; \
    } catch(...) { \
      FAIL() << "statement '" #statement "' did not generate a logging exception"; \
    } \
    EXPECT_EQ(true, caughtException) << "statement '" #statement "' should fail"; \
  }

/**
 * Add a test to Google Test suits, using `GTEST_TEST` temporarily 
 * instead of `TEST` to avoid conflict.
 * 
 * @note Internal macro, do NOT use directly. 
 */
#define ADDING_TEST(testname, postfix) \
    GTEST_TEST(testname##postfix, testname) { \
    testname t; \
    t.RunTests(); \
  }  

/**
 * Add a test to Google Test suits.
 *
 * @deprecated This macro is preserved only for compatible reasons. You should
 * not use it in new codes. 
 */
#define ADD_TEST(testname) \
   ADDING_TEST(testname, Case)

#endif // NTA_TESTER_HPP

