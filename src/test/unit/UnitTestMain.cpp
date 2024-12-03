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
Google test main program
*/

#if defined(NTA_OS_WINDOWS)
// Exclude rarely-used stuff from Windows headers
#define WIN32_LEAN_AND_MEAN
#endif

#include <gtest/gtest.h>
#include <nupic/utils/Log.hpp>

using namespace std;
using namespace nupic;

// APR must be explicit initialized
#include <apr-1/apr_general.h>

int main(int argc, char **argv) {

  // initialize APR
  apr_status_t result;
  result =
      apr_app_initialize(&argc, (char const *const **)&argv, nullptr /*env*/);
  if (result)
    NTA_THROW << "error initializing APR. Err code: " << result;

  // initialize GoogleTest
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
