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
    Random Number Generator interface
*/

#ifndef NTA_RANDOM_HPP
#define NTA_RANDOM_HPP

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <nupic/proto/RandomProto.capnp.h>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

typedef NTA_UInt64 (*RandomSeedFuncPtr)();

namespace nupic {
/**
 * @b Responsibility
 * Provides standardized random number generation for the NuPIC Runtime Engine.
 * Seed can be logged in one run and then set in another.
 * @b Rationale
 * Makes it possible to reproduce tests that are driven by random number
 * generation.
 *
 * @b Description
 * Functionality is similar to the standard random() function that is provided
 * by C.
 *
 * Each Random object is a random number generator. There are three ways of
 * creating one:
 * 1) explicit seed
 *       Random rng(seed);
 * 2) self-seeded
 *       Random rng;
 * 3) named generator -- normally self-seeded, but seed may be
 *    set explicitly through an environment variable
 *       Random rng("level2TP");
 *    If NTA_RANDOM_DEBUG is set, this object will log its self-seed
 *    The seed can be explicitly set through NTA_RANDOM_SEED_level2TP
 *
 * Good self-seeds are generated by an internal global random number generator.
 * This global rng is seeded from the current time, but its seed may be
 * overridden with NTA_RANDOM_SEED
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
class RandomImpl;

class Random : public Serializable<RandomProto> {
public:
  /**
   * Retrieve the seeder. If seeder not set, allocates the
   * singleton and and initializes the seeder.
   */
  static RandomSeedFuncPtr getSeeder();

  Random(UInt64 seed = 0);

  // support copy constructor and operator= -- these require non-default
  // implementations because of the impl_ pointer.
  // They do a deep copy of impl_ so that an RNG and its copy generate the
  // same set of numbers.
  Random(const Random &);
  Random &operator=(const Random &);
  ~Random();

  // write serialized data
  using Serializable::write;
  void write(RandomProto::Builder &proto) const override;

  // read and deserialize data
  using Serializable::read;
  void read(RandomProto::Reader &proto) override;

  // return a value uniformly distributed between 0 and max-1
  UInt32 getUInt32(UInt32 max = MAX32);
  UInt64 getUInt64(UInt64 max = MAX64);
  // return a double uniformly distributed on 0...1.0
  Real64 getReal64();

  // populate choices with a random selection of nChoices elements from
  // population. throws exception when nPopulation < nChoices
  // templated functions must be defined in header
  template <typename T>
  void sample(T population[], UInt32 nPopulation, T choices[],
              UInt32 nChoices) {
    if (nChoices == 0) {
      return;
    }
    if (nChoices > nPopulation) {
      NTA_THROW << "population size must be greater than number of choices";
    }
    UInt32 nextChoice = 0;
    for (UInt32 i = 0; i < nPopulation; ++i) {
      if (getUInt32(nPopulation - i) < (nChoices - nextChoice)) {
        choices[nextChoice] = population[i];
        ++nextChoice;
        if (nextChoice == nChoices) {
          break;
        }
      }
    }
  }

  // randomly shuffle the elements
  template <class RandomAccessIterator>
  void shuffle(RandomAccessIterator first, RandomAccessIterator last) {
    UInt n = last - first;
    while (first != last) {
      // Pick a random position between the current and the end to swap the
      // current element with.
      UInt i = getUInt32(n);
      std::swap(*first, *(first + i));

      // Move to the next element and decrement the number of possible
      // positions remaining.
      first++;
      n--;
    }
  }

  // for STL compatibility
  UInt32 operator()(UInt32 n = MAX32) { return getUInt32(n); }

  // normally used for debugging only
  UInt64 getSeed() { return seed_; }

  // for STL
  typedef UInt32 argument_type;
  typedef UInt32 result_type;

  result_type max() { return MAX32; }
  result_type min() { return 0; }

  static const UInt32 MAX32;
  static const UInt64 MAX64;

  bool operator==(const Random &other) const;
  inline bool operator!=(const Random &other) const {
    return !operator==(other);
  }

  // called by the plugin framework so that plugins
  // get the "global" seeder
  static void initSeeder(const RandomSeedFuncPtr r);

  static void shutdown();

protected:
  // each "universe" (application/plugin/python module) has its own instance,
  // but the instance should be NULL in all but one
  static Random *theInstanceP_;
  // seeder_ is a function called by the constructor to get new random seeds
  // If not set when we call Random constructor, then the singleton is allocated
  // and seeder_ is set to a function that uses our singleton
  // initFromPlatformServices can also be used to initialize the seeder_
  static RandomSeedFuncPtr seeder_;

  void reseed(UInt64 seed);

  RandomImpl *impl_;
  UInt64 seed_;

  friend class RandomTest;
  friend std::ostream &operator<<(std::ostream &, const Random &);
  friend std::istream &operator>>(std::istream &, Random &);
  friend NTA_UInt64 GetRandomSeed();
};

// serialization/deserialization
std::ostream &operator<<(std::ostream &, const Random &);
std::istream &operator>>(std::istream &, Random &);

// This function returns seeds from the Random singleton in our
// "universe" (application, plugin, python module). If, when the
// Random constructor is called, seeder_ is NULL, then seeder_ is
// set to this function. The plugin framework can override this
// behavior by explicitly setting the seeder to the RandomSeeder
// function provided by the application.
NTA_UInt64 GetRandomSeed();

} // namespace nupic

#endif // NTA_RANDOM_HPP
