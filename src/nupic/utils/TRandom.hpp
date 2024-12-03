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
    Random Number Generator interface (for tests)
*/

#ifndef NTA_TRANDOM_HPP
#define NTA_TRANDOM_HPP

#include <nupic/utils/Random.hpp>
#include <string>

namespace nupic {
/**
 * @b Responsibility
 * Provides standard random number generation for testing.
 * Seed can be logged in one run and then set in another.
 * @b Rationale
 * Makes it possible to reproduce tests that are driven by random number
 * generation.
 *
 * @b Description
 * Functionality is similar to the standard random() function that is provided
 * by C.
 *
 * TRandom is a subclass of Random with an additional constructor.
 * This constructor creates a named generator -- normally self-seeded, but
 * seed may be set explicitly through an environment variable. For example:
 *       Random rng("level2TP");
 * If NTA_RANDOM_DEBUG is set, this object will log its self-seed
 * The seed can be explicitly set through NTA_RANDOM_SEED_level2TP
 *
 * If self-seeded, the seed comes from the same global random number generaetor
 * used for Random.
 *
 * Automated tests that use random numbers should normally use named generators.
 * This allows them to get a different seed each time, but also allows
 * reproducibility in the case that a test failure is triggered by a particular
 * seed.
 *
 * Random should not be used if cryptographic strength is required (e.g. for
 * generating a challenge in an authentication scheme).
 *
 * @todo Add ability to specify different rng algorithms.
 */

class TRandom : public Random {
public:
  TRandom(std::string name);

private:
  friend class TRandomTest;
};
} // namespace nupic

#endif // NTA_TRANDOM_HPP
