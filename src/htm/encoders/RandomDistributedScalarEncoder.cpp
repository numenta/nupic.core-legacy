/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
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
 * --------------------------------------------------------------------- */

/** @file
 * Implementation of the RandomDistributedScalarEncoder
 */

#include <htm/encoders/RandomDistributedScalarEncoder.hpp>
#include <murmurhash3/MurmurHash3.hpp>
#include <htm/utils/Random.hpp>
#include <algorithm> // fill

using namespace std;
using namespace htm;

RandomDistributedScalarEncoder::RandomDistributedScalarEncoder(
                                              const RDSE_Parameters &parameters)
  { initialize( parameters ); }

void RandomDistributedScalarEncoder::initialize( const RDSE_Parameters &parameters)
{
  // Check size parameter
  NTA_CHECK( parameters.size > 0u );

  // Initialize parent class.
  BaseEncoder<Real64>::initialize({ parameters.size });

  // Check other parameters
  UInt num_active_args = 0;
  if( parameters.activeBits > 0u)   { num_active_args++; }
  if( parameters.sparsity   > 0.0f) { num_active_args++; }
  NTA_CHECK( num_active_args != 0u )
      << "Missing argument, need one of: 'activeBits' or 'sparsity'.";
  NTA_CHECK( num_active_args == 1u )
      << "Too many arguments, choose only one of: 'activeBits' or 'sparsity'.";

  UInt num_resolution_args = 0;
  if( parameters.radius     > 0.0f) { num_resolution_args++; }
  if( parameters.category )         { num_resolution_args++; }
  if( parameters.resolution > 0.0f) { num_resolution_args++; }
  NTA_CHECK( num_resolution_args != 0u )
      << "Missing argument, need one of: 'radius', 'resolution', 'category'.";
  NTA_CHECK( num_resolution_args == 1u )
      << "Too many arguments, choose only one of: 'radius', 'resolution', 'category'.";

  args_ = parameters;
  // Finish filling in all of parameters.

  // Determine number of activeBits.
  if( args_.sparsity > 0.0f ) {
    NTA_CHECK( args_.sparsity >= 0.0f );
    NTA_CHECK( args_.sparsity <= 1.0f );
    args_.activeBits = (UInt) round( args_.size * args_.sparsity );
    NTA_CHECK( args_.activeBits > 0u );
  }
  // Determine sparsity. Always calculate this even if it was given, to correct for rounding error.
  args_.sparsity = (Real) args_.activeBits / args_.size;

  if( args_.category ) {
    args_.radius = 1.0f;
  }

  // Determine resolution.
  if( args_.radius > 0.0f ) {
    args_.resolution = args_.radius / args_.activeBits;
  }
  // Determine radius.
  else if( args_.resolution > 0.0f ) {
    args_.radius = args_.activeBits * args_.resolution;
  }

  while( args_.seed == 0u ) {
    args_.seed = Random().getUInt32();
  }
}

void RandomDistributedScalarEncoder::encode(Real64 input, SDR &output)
{
  // Check inputs
  NTA_CHECK( output.size == size );
  if( isnan(input) ) {
    output.zero();
    return;
  }
  else if( args_.category ) {
    NTA_CHECK( input == Real64(UInt64(input)))
      << "Input to category encoder must be an unsigned integer!";
  }

  auto &data = output.getDense();
  fill( data.begin(), data.end(), 0u );

  const UInt index = (UInt) (input / args_.resolution);
  for(auto offset = 0u; offset < args_.activeBits; ++offset)
  {
    UInt hash_buffer = index + offset;
    UInt32 bucket = MurmurHash3_x86_32(&hash_buffer, sizeof(hash_buffer), args_.seed);
    bucket = bucket % size;

    // Don't worry about hash collisions.  Instead measure the critical
    // properties of the encoder in unit tests and quantify how significant
    // the hash collisions are.  This encoder can not fix the collisions
    // because it does not record past encodings.  Collisions cause small
    // deviations in the sparsity or semantic similarity, depending on how
    // they're handled.

    // Exercise for the reader: Calculate the probability of a hash collision
    // and account for it in the sparsity.

    data[bucket] = 1u;
  }
  output.setDense( data );
}

std::ostream & htm::operator<<(std::ostream & out, const RandomDistributedScalarEncoder &self)
{
  out << "RDSE ";
  out << "  size:       " << self.parameters.size << ",\n";
  out << "  activeBits: " << self.parameters.activeBits << ",\n";
  out << "  resolution: " << self.parameters.resolution << ",\n";
  out << "  category:   " << self.parameters.category << ",\n";
  out << "  seed:       " << self.parameters.seed << std::endl;
  return out;
}
