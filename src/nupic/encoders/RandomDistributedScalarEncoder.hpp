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
 * Define the RandomDistributedScalarEncoder
 */

#ifndef NTA_ENCODERS_RDSE
#define NTA_ENCODERS_RDSE

#include <nupic/types/Sdr.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>
#include <nupic/utils/Random.hpp>

namespace nupic {

/**
 * TODO DOCUMENTATION
      COPY FROM NUPIC!
      MY FROM MY OLD WORK?
 */
class RandomDistributedScalarEncoder // : public Serializable // TODO Serializable unimplemented!
{
private:
  UInt size_;
  Real sparsity_;
  Real radius_;
  UInt seed_;

public:
  /**
   * TODO DOCUMENTATION
   * https://arxiv.org/pdf/1602.05925.pdf
   */
  RandomDistributedScalarEncoder(UInt size, Real sparsity, Real radius, UInt seed = 0u)
    { initialize(size, sparsity, radius, seed); }

  void initialize(UInt size, Real sparsity, Real radius, UInt seed = 0u) {
    size_       = size;
    sparsity_   = sparsity;
    radius_     = radius;
    // Use the given seed to make a better, more randomized seed.
    Random apple( seed );
    seed_ = apple();

    NTA_CHECK(sparsity >= 0.0f);
    NTA_CHECK(sparsity <= 1.0f);
    NTA_CHECK(radius > 0.0f);
  }

  const UInt &size      = size_;
  const Real &sparsity  = sparsity_;
  const Real &radius    = radius_;

  /**
   * TODO DOCUMENTATION
   */
  void encode(Real value, SDR &output) const {
    NTA_CHECK( output.size == size );
    SDR_dense_t data( size, 0 );
    const UInt n_active   = round(size * sparsity);
    const Real resolution = (Real) 2.0f * radius / n_active;
    const UInt index      = seed_ + (UInt) (value / resolution);
    hash<std::string> h;
    for(auto offset = 0u; offset < n_active; ++offset) {
      std::stringstream temp;
      temp << index + offset;
      UInt bucket = h(temp.str()) % size;
      // Don't worry about hash collisions.  Instead measure the critical
      // properties of the encoder in unit tests and quantify how significant
      // the hash collisions are.  This encoder can not fix the collisions
      // because it does not record past encodings.  Collisions cause small
      // deviations in the sparsity or semantic similarity, depending on how
      // they're handled.
      // TODO: Calculate the probability of a hash collision and account for
      // it in n_active.
      data[bucket] = 1u;
    }
    output.setDense( data );
  }
};     // End class RandomDistributedScalarEncoder
typedef RandomDistributedScalarEncoder RDSE;

}      // End namespace nupic
#endif // End ifdef NTA_ENCODERS_RDSE
