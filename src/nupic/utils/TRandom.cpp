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
    Random Number Generator implementation
*/

#include <cstdlib>
#include <ctime>
#include <nupic/os/Env.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/StringUtils.hpp>
#include <nupic/utils/TRandom.hpp>

using namespace nupic;

TRandom::TRandom(std::string name) {

  UInt64 seed = 0;

  std::string optionName = "set_random";
  if (name != "") {
    optionName += "_" + name;
  }

  bool seed_from_environment = false;
  if (Env::isOptionSet(optionName)) {
    seed_from_environment = true;
    std::string val = Env::getOption(optionName);
    try {
      seed = StringUtils::toUInt64(val, true);
    } catch (...) {
      NTA_WARN << "Invalid value \"" << val << "\" for NTA_SET_RANDOM. Using 1";
      seed = 1;
    }
  } else {
    // Seed the global rng from time().
    // Don't seed subsequent ones from time() because several random
    // number generators may be initialized within the same second.
    // Instead, use the global rng.
    if (theInstanceP_ == nullptr) {
      seed = (UInt64)time(nullptr);
    } else {
      seed = (*Random::getSeeder())();
    }
  }

  if (Env::isOptionSet("random_debug")) {
    if (seed_from_environment) {
      NTA_INFO << "TRandom(" << name << ") -- initializing with seed " << seed
               << " from environment";
    } else {
      NTA_INFO << "TRandom(" << name << ") -- initializing with seed " << seed;
    }
  }

  // Create the actual RNG
  // @todo to add different algorithm support, this is where we will
  // instantiate different implementations depending on the requested
  // algorithm
  reseed(seed);
}
