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
    Implementation of Tester
*/

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <nta/test/Tester.hpp>
#include <nta/utils/Log.hpp>

#include <nta/os/Directory.hpp>
#include <nta/os/Path.hpp>

using namespace std;
namespace nta {

  /* static members */
  std::string Tester::testInputDir_;
  std::string Tester::testOutputDir_;

  Tester::Tester()
  {
    
  }

  Tester::~Tester()
  {
  }

  void Tester::init() {
    /* the following functions implicitly set these options if not
     * already set. 
     */
    // TODO -- fix me! needed for finding test data
    testInputDir_ = "/does/not/exist";
    testOutputDir_ = Path::makeAbsolute("testeverything.out");

    // Create if it doesn't exist
    if (!Path::exists(testOutputDir_)) {
      std::cout << "Tester -- creating testoutput directory " << std::string(testOutputDir_) << "\n";
      // will throw if unsuccessful. 
      Directory::create(string(testOutputDir_));
    } 
  }

  std::string Tester::fromTestInputDir(const std::string& path) {
    
    Path testinputpath(testInputDir_);
    if (path != "")
      testinputpath += path;
    
    return string(testinputpath);
  }

  std::string Tester::fromTestOutputDir(const std::string& path) {
    
    Path testoutputpath(testOutputDir_);
    if (path != "")
      testoutputpath += path;
    
    return std::string(testoutputpath);
  }

} // end namespace nta

