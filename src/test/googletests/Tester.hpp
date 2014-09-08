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
#include <nta/utils/Log.hpp>

//----------------------------------------------------------------------

namespace nta {

  /** Abstract base class for unit testers.
   *
   * This class specifies a simple interface for unit testing all Numenta classes.
   * A unit test is created by subclassing from Tester and implementing RunTests().
   * 
   */
  class Tester  {
  public:

    /** Default constructor	 */
    Tester();

    virtual ~Tester();
    
    /// This is the main method that subclasses should implement.
    /// It is expected that this method will thoroughly test the target class by calling
    /// each method and testing boundary conditions.
    /// @todo change capitalization. Must be changed in all tests. 
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
  

} // end namespace nta

#define TESTEQUAL(expected, actual) \
  EXPECT_EQ(expected, actual);

#define TESTEQUAL2(name, expected, actual) \
  EXPECT_EQ(expected, actual) << "assertion \"" #name "\" failed at " << __FILE__ << ":" << __LINE__ ;

#define TESTEQUAL_STR(expected, actual) \
  TESTEQUAL(std::string(expected), std::string(actual));

#define TESTEQUAL2_STR(name, expected, actual) \
  TESTEQUAL2(name, std::string(expected), std::string(actual));

#define TESTEQUAL_FLOAT(expected, actual) \
  ASSERT_NEAR(expected, actual, 0.000001) << "assertion \"" #expected " == " #actual << "\" failed at " << __FILE__ << ":" << __LINE__ ;


#undef TEST

#define TEST(condition) \
  EXPECT_TRUE(condition) << "assertion \"" #condition "\" failed at " << __FILE__ << ":" << __LINE__ ;
#define TEST2(name, condition) \
  EXPECT_TRUE(condition) << "assertion \"" #name ": " #condition "\" failed at " << __FILE__ << ":" << __LINE__ ;

// http://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide#Exception_Assertions
#define SHOULDFAIL(statement) \
  EXPECT_THROW(statement, std::exception);

// { \
//   if (!disableNegativeTests_) \
//   { \
//     bool caughtException = false; \
//     try { \
//       statement; \
//     } catch(std::exception& ) { \
//       caughtException = true; \
//     } \
//     testEqual("statement '" #statement "' should fail", __FILE__, __LINE__, true, caughtException); \
//   } else { \
//     disable("statement '" #statement "' should fail", __FILE__, __LINE__); \
//   } \
// }

#define SHOULDFAIL_WITH_MESSAGE(statement, message) \
  { \
    bool caughtException = false; \
    try { \
      statement; \
    } catch(nta::LoggingException& e) { \
      caughtException = true; \
      EXPECT_STREQ(message, e.getMessage()) << "statement '" #statement "' should fail with message \"" \
      << message << "\", but failed with message \"" << e.getMessage() << "\""; \
    } catch(...) { \
      FAIL() << "statement '" #statement "' did not generate a logging exception"; \
    } \
    EXPECT_EQ(true, caughtException) << "statement '" #statement "' should fail"; \
  }

#define ADDING_TEST(testname, postfix) \
    GTEST_TEST(testname##postfix, testname) { \
    testname t; \
    t.RunTests(); \
  }  

#define ADD_TEST(testname) \
   ADDING_TEST(testname, Case)

#endif // NTA_TESTER_HPP

