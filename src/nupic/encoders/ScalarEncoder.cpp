/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.
 *               2019, David McDougall
 *
 * Unless you have an agreement with Numenta, Inc., for a separate license for
 * this software code, the following terms and conditions apply:
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
 * Implementation of the ScalarEncoder
 */

#include <algorithm> // std::min
#include <numeric>   // std::iota
#include <cmath>     // std::isnan
#include <nupic/encoders/ScalarEncoder.hpp>
using nupic::sdr::SDR;

namespace nupic {
namespace encoders {

ScalarEncoder::ScalarEncoder(ScalarEncoderParameters &parameters)
  { initialize( parameters ); }

void ScalarEncoder::initialize(ScalarEncoderParameters &parameters)
{
  // Check parameters
  NTA_CHECK( parameters.minimum < parameters.maximum );

  UInt num_active_args = 0;
  if( parameters.activeBits > 0)    { num_active_args++; }
  if( parameters.sparsity   > 0.0f) { num_active_args++; }
  NTA_CHECK( num_active_args != 0u )
      << "Missing argument, need one of: 'activeBits' or 'sparsity'.";
  NTA_CHECK( num_active_args == 1u )
      << "Too many arguments, choose only one of: 'activeBits' or 'sparsity'.";

  UInt num_size_args = 0;
  if( parameters.size       > 0u)   { num_size_args++; }
  if( parameters.radius     > 0.0f) { num_size_args++; }
  if( parameters.resolution > 0.0f) { num_size_args++; }
  NTA_CHECK( num_size_args != 0u )
      << "Missing argument, need one of: 'size', 'radius', 'resolution'.";
  NTA_CHECK( num_size_args == 1u )
      << "Too many arguments, choose only one of: 'size', 'radius', 'resolution'.";

  if( parameters.periodic ) {
    NTA_CHECK( not parameters.clipInput )
      << "Will not clip periodic inputs.  Caller must apply modulus.";
    // TODO: Instead of balking, do the modulus!
  }

  args_ = parameters;
  // Finish filling in all of parameters.

  if( args_.sparsity > 0.0f ) {
    NTA_CHECK( parameters.sparsity >= 0.0f );
    NTA_CHECK( parameters.sparsity <= 1.0f );
    NTA_CHECK( args_.size > 0u )
        << "'Sparsity' requires that the 'size' also be given.";
    args_.activeBits = (UInt) round( args_.size * args_.sparsity );
  }

  // Determine resolution & size.
  Real64 extentWidth;
  if( args_.periodic ) {
    extentWidth = args_.maximum - args_.minimum;
  }
  else {
    Real64  maxInclusive     = args_.maximum;
    UInt64* maxInclusiveData = reinterpret_cast<UInt64*>( &maxInclusive );
    (*maxInclusiveData)++; // Increase maxInclusive by the smallest possible amount.
    extentWidth = maxInclusive - args_.minimum;
  }
  if( args_.size > 0u ) {
    // Distribute the active bits along the domain [minimum, maximum], including
    // the endpoints. The resolution is the width of each band between the
    // points.
    if( args_.periodic ) {
      args_.resolution = extentWidth / args_.size;
    }
    else {
      const UInt nBuckets = args_.size - (args_.activeBits - 1);
      args_.resolution = extentWidth / (nBuckets - 1);
    }
  }
  else {
    if( args_.radius > 0.0f ) {
      args_.resolution = args_.radius / args_.activeBits;
    }

    const int neededBands = (int)ceil(extentWidth / args_.resolution);
    if( args_.periodic ) {
      args_.size = neededBands;
    }
    else {
      args_.size = neededBands + (args_.activeBits - 1);
    }
  }

  // Determine radius. Always calculate this even if it was given, to correct for rounding error.
  args_.radius = args_.activeBits * args_.resolution;

  // Determine sparsity. Always calculate this even if it was given, to correct for rounding error.
  args_.sparsity = (Real) args_.activeBits / args_.size;

  // Sanity check the parameters.
  NTA_CHECK( args_.size       > 0u );
  NTA_CHECK( args_.activeBits > 0u );
  NTA_CHECK( args_.activeBits < args_.size );

  // Initialize parent class.
  BaseEncoder<Real64>::initialize({ args_.size });
}

void ScalarEncoder::encode(Real64 input, SDR &output)
{
  // Check inputs
  NTA_CHECK( output.size == size );
  if( std::isnan(input) ) {
    output.zero();
    return;
  }
  else if( args_.clipInput ) {
    if( args_.periodic ) {
      // TODO: Apply modulus to inputs here!
      NTA_THROW << "Unimplemented";
    }
    else {
      input = std::max(input, parameters.minimum);
      input = std::min(input, parameters.maximum);
    }
  }
  else {
    NTA_CHECK(input >= parameters.minimum && input <= parameters.maximum)
        << "Input must be within range [minimum, maximum]!";
  }

  UInt start = (UInt) round((input - parameters.minimum) / parameters.resolution);

  // The endpoints of the input range are inclusive, which means that the
  // maximum value may round up to an index which is outside of the SDR. Correct
  // this by pushing the endpoint (and everything which rounds to it) onto the
  // last bit in the SDR.
  if( not parameters.periodic ) {
    start = std::min(start, output.size - parameters.activeBits);
  }

  auto &sparse = output.getSparse();
  sparse.resize( parameters.activeBits );
  std::iota( sparse.begin(), sparse.end(), start );

  if( parameters.periodic ) {
    for( auto & bit : sparse ) {
      if( bit >= output.size ) {
        bit -= output.size;
      }
    }
  }

  output.setSparse( sparse );
}

void ScalarEncoder::save(std::ostream &stream) const
{
  stream << "ScalarEncoder ";
  stream << parameters.minimum    << " ";
  stream << parameters.maximum    << " ";
  stream << parameters.clipInput  << " ";
  stream << parameters.periodic   << " ";
  stream << parameters.activeBits << " ";
  // Save the resolution instead of the size BC it's higher precision.
  stream << parameters.resolution << " ";
  stream << "~ScalarEncoder~" << std::endl;
}

void ScalarEncoder::load(std::istream &stream)
{
  std::string prelude;
  stream >> prelude;
  NTA_CHECK( prelude == "ScalarEncoder" );

  ScalarEncoderParameters p;
  stream >> p.minimum;
  stream >> p.maximum;
  stream >> p.clipInput;
  stream >> p.periodic;
  stream >> p.activeBits;
  stream >> p.resolution;

  std::string postlude;
  stream >> postlude;
  NTA_CHECK( postlude == "~ScalarEncoder~" );
  stream.ignore( 1 ); // Eat the trailing newline.

  initialize( p );
}

} // end namespace encoders
} // end namespace nupic
