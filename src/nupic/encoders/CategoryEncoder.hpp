/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
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
 * Define the CategoryEncoder
 */

#ifndef NTA_ENCODERS_CATEGORY
#define NTA_ENCODERS_CATEGORY

#include <map>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>

namespace nupic {

/**
 * TODO DOCUMENTATION
 */
template<typename CategoryType>
class CategoryEncoder // : public Serializable // TODO Serializable unimplemented!
{
private:
  UInt   size_;
  Real   sparsity_;
  Random rng_;
  std::map<CategoryType, UInt> inputSeedMap_;

public:
  /**
   * TODO DOCUMENTATION
   */
  CategoryEncoder(UInt size, Real sparsity) {
    Random rng( 0 );
    initialize(size, sparsity, rng);
  }
  CategoryEncoder(UInt size, Real sparsity, Random rng)
    { initialize(size, sparsity, rng); }
  void initialize(UInt size, Real sparsity, Random rng) {
    size_     = size;
    sparsity_ = sparsity;
    rng_      = rng;
  }

  const UInt                         &size         = size_;
  const Real                         &sparsity     = sparsity_;
  const std::map<CategoryType, UInt> &inputSeedMap = inputSeedMap_;

  /**
   * TODO DOCUMENTATION
   */
  void encode(const CategoryType value, SDR &output) {
    if( inputSeedMap.count( value ) == 0 ) {
      // Insert new value
      UInt seed;
      do{
        seed = randomSeed_();
        encodeFromSeed_(seed, output);
      } while( maximumOverlap_( output ) >= 0.50f );
      inputSeedMap_[value] = seed;
    }
    else {
      encodeFromSeed_(inputSeedMap.at(value), output);
    }
  }

  const CategoryType decode(const SDR &encoding) {
    // TODO
  }

private:
  UInt randomSeed_() {
    UInt seed = 0;
    do {
      seed = rng_();
    } while( seed == 0 );
    return seed;
  }

  void encodeFromSeed_(UInt seed, SDR &output) {
    NTA_ASSERT( seed != 0u );
    NTA_CHECK( output.size == size );
    Random rng( seed );
    output.randomize( sparsity, rng );
  }

  Real maximumOverlap_(SDR &newCategory) {
    Real maxOvlp = 0.0f;
    const UInt n_active = std::round(size * sparsity);
    SDR X({ size });
    for( const auto &encoding : inputSeedMap ) {
      encodeFromSeed_( encoding.second, X );
      maxOvlp = std::max( maxOvlp, (Real) X.getOverlap( newCategory ) / n_active);
    }
    return maxOvlp;
  }
};     // End class CategoryEncoder
}      // End namespace nupic
#endif // End ifdef NTA_ENCODERS_CATEGORY
