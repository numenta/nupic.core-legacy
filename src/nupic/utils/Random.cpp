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
    Random Number Generator implementation
*/
#include <iostream> // for istream, ostream

#include <nupic/utils/Log.hpp>
#include <nupic/utils/Random.hpp>

using namespace nupic;


bool Random::operator==(const Random &o) const {
  return seed_ == o.seed_ && \
	 steps_ == o.steps_ && \
	 gen == o.gen;
}

std::mt19937 static_gen;

Random::Random(UInt64 seed) {
  if (seed == 0) {
    seed_ = static_gen(); //generate random value from HW RNG
  } else {
    seed_ = seed;
  }
  // if seed is zero at this point, there is a logic error.
  NTA_CHECK(seed_ != 0);
  gen.seed(seed_); //seed the generator
  steps_ = 0;
}


namespace nupic {
std::ostream &operator<<(std::ostream &outStream, const Random &r) {
  outStream << "random-v2" << " ";
  outStream << r.seed_ << " ";
  outStream << r.steps_ << " ";
  outStream << "endrandom-v2" << " ";
  return outStream;
}


std::istream &operator>>(std::istream &inStream, Random &r) {
  std::string version;

  inStream >> version;
  NTA_CHECK(version == "random-v2") << "Random() deserializer -- found unexpected version string '"
              << version << "'";
  inStream >> r.seed_;
  r.gen.seed(r.seed_); //reseed
  inStream >> r.steps_;
  r.gen.discard(r.steps_); //advance n steps
  //FIXME we could de/serialize directly RNG gen, it should be multi-platform according to standard, 
  //but on OSX CI it wasn't (25/11/2018). So "hacking" the above instead. 
  std::string endtag;
  inStream >> endtag;
  NTA_CHECK(endtag == "endrandom-v2") << "Random() deserializer -- found unexpected end tag '" << endtag  << "'";
  inStream.ignore(1);

  return inStream;
}

// helper function for seeding RNGs across the plugin barrier
// Unless there is a logic error, should not be called if
// the Random singleton has not been initialized.
UInt32 GetRandomSeed() {
  return nupic::Random().getUInt32();
}
} // namespace nupic
