/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
 *               2019, David McDougall
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
 * Implementation of the ScalarEncoder
 */

#include <algorithm> // std::min std::sort
#include <numeric>   // std::iota
#include <cmath>     // std::isnan std::nextafter
#include <htm/encoders/ScalarEncoder.hpp>

namespace htm {

ScalarEncoder::ScalarEncoder(const ScalarEncoderParameters &parameters)
  { initialize( parameters ); }

void ScalarEncoder::initialize(const ScalarEncoderParameters &parameters)
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
  if( parameters.category )         { num_size_args++; }
  if( parameters.resolution > 0.0f) { num_size_args++; }
  NTA_CHECK( num_size_args != 0u )
      << "Missing argument, need one of: 'size', 'radius', 'resolution', 'category'.";
  NTA_CHECK( num_size_args == 1u )
      << "Too many arguments, choose only one of: 'size', 'radius', 'resolution', 'category'.";

  if( parameters.periodic ) {
    NTA_CHECK( not parameters.clipInput )
      << "Will not clip periodic inputs.  Caller must apply modulus.";
    // TODO: Instead of balking, do the modulus!
  }
  if( parameters.category ) {
    NTA_CHECK( not parameters.clipInput )
      << "Incompatible arguments: category & clipInput.";
    NTA_CHECK( not parameters.periodic )
      << "Incompatible arguments: category & periodic.";
    NTA_CHECK( args_.minimum == Real64(UInt64(args_.minimum)) )
      << "Minimum input value of category encoder must be an unsigned integer!";
    NTA_CHECK( args_.maximum == Real64(UInt64(args_.maximum)) )
      << "Maximum input value of category encoder must be an unsigned integer!";
  }

  args_ = parameters;
  // Finish filling in all of parameters.

  if( args_.category ) {
    args_.radius = 1.0f;
  }

  if( args_.sparsity > 0.0f ) {
    NTA_CHECK( parameters.sparsity >= 0.0f );
    NTA_CHECK( parameters.sparsity <= 1.0f );
    NTA_CHECK( args_.size > 0u )
        << "Argument 'sparsity' requires that the 'size' also be given.";
    args_.activeBits = (UInt) round( args_.size * args_.sparsity );
  }

  // Determine resolution & size.
  Real64 extentWidth;
  if( args_.periodic ) {
    extentWidth = args_.maximum - args_.minimum;
  }
  else {
    // Increase the max by the smallest possible amount.
    Real64  maxInclusive = std::nextafter( args_.maximum, HUGE_VAL );
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
  // Sanity check the parameters.
  NTA_CHECK(args_.size > 0u);
  NTA_CHECK(args_.activeBits > 0u);
  NTA_CHECK(args_.activeBits < args_.size);

  // Determine radius. Always calculate this even if it was given, to correct for rounding error.
  args_.radius = args_.activeBits * args_.resolution;

  // Determine sparsity. Always calculate this even if it was given, to correct for rounding error.
  args_.sparsity = (Real) args_.activeBits / args_.size;


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
      NTA_THROW << "Unimplemented. This code is unreachable.";
    }
    else {
      input = std::max(input, parameters.minimum);
      input = std::min(input, parameters.maximum);
    }
  }
  else {
    if( args_.category ) {
      NTA_CHECK( input == Real64(UInt64(input)))
        << "Input to category encoder must be an unsigned integer!";
    }
    NTA_CHECK(input >= parameters.minimum && input <= parameters.maximum)
        << "Input must be within range [minimum, maximum]! " << input << " vs [" << parameters.minimum << " , " << parameters.maximum << " ]";
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
    std::sort( sparse.begin(),  sparse.end() );
  }

  output.setSparse( sparse );
}

std::ostream & operator<<(std::ostream & out, const ScalarEncoder &self)
{
  out << "ScalarEncoder \n";
  out << "  minimum:   " << self.parameters.minimum    << ",\n";
  out << "  maximum:   " << self.parameters.maximum    << ",\n";
  out << "  clipInput: " << self.parameters.clipInput  << ",\n";
  out << "  periodic:  " << self.parameters.periodic   << ",\n";
  out << "  category:  " << self.parameters.category   << ",\n";
  out << "  activeBits:" << self.parameters.activeBits << ",\n";
  out << "  resolution:" << self.parameters.resolution << std::endl;
  return out;
}

} // end namespace htm
