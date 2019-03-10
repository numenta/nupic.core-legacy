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

#include <algorithm> //std::fill
#include <cmath>

#include <nupic/encoders/ScalarEncoder.hpp>

namespace nupic {

ScalarEncoder::ScalarEncoder(ScalarEncoderParameters &parameters)
  { initialize( parameters ); }

void ScalarEncoder::initialize(ScalarEncoderParameters &parameters)
{
  // Check parameters
  NTA_CHECK( parameters.minimum < parameters.maximum );

  UInt num_active_args = 0;
  if( parameters.active     > 0)    num_active_args++;
  if( parameters.sparsity   > 0.0f) num_active_args++;
  NTA_CHECK( num_active_args != 0u )
      << "Missing argument: 'active' or 'sparsity'.";
  NTA_CHECK( num_active_args == 1u )
      << "Too many arguments, choose only one of 'active' or 'sparsity'.";

  UInt num_size_args = 0;
  if( parameters.size       > 0u)   num_size_args++;
  if( parameters.radius     > 0.0f) num_size_args++;
  if( parameters.resolution > 0.0f) num_size_args++;
  NTA_CHECK( num_size_args != 0u )
      << "Missing argument, one of: 'size', 'radius', 'resolution'.";
  NTA_CHECK( num_size_args == 1u )
      << "Too many arguments, choose only one of 'size', 'radius', 'resolution'.";

  args_ = parameters;
  // Finish filling in all of parameters.

  if( args_.sparsity > 0.0f ) {
    NTA_CHECK( parameters.sparsity >= 0.0f );
    NTA_CHECK( parameters.sparsity <= 1.0f );
    NTA_CHECK( args_.size > 0u )
        << "'Sparsity' requires that the 'size' also be given.";
    args_.active = (UInt) round( args_.size * args_.sparsity );
  }

  const double extentWidth = args_.maximum - args_.minimum;
  if( args_.size > 0u ) {
    // Distribute the active bits along the domain [minimum, maximum], including
    // the endpoints. The resolution is the width of each band between the
    // points.
    const int nBuckets = args_.size - (args_.active - 1);
    args_.resolution = extentWidth / (nBuckets - 1);
  }
  else {
    if( args_.radius > 0.0f ) {
      args_.resolution = args_.radius / args_.active;
    }

    const int neededBands   = (int)ceil(extentWidth / args_.resolution);
    const int neededBuckets = neededBands + 1;
    args_.size = neededBuckets + (args_.active - 1);
  }

  if( args_.radius == 0.0f ) {
    args_.radius = args_.size * args_.resolution;
  }

  // Always calculate the sparsity even if it was given, to correct for rounding error.
  args_.sparsity = (Real) args_.active / args_.size;

  NTA_CHECK( args_.size   > 0u );
  NTA_CHECK( args_.active > 0u );
  NTA_CHECK( args_.active < args_.size );

  // Initialize parent class.
  BaseEncoder<double>::initialize({ args_.size });
}

void ScalarEncoder::encode(double input, SDR &output)
{
  if( args_.clipInput ) {
    input = input < parameters.minimum ? parameters.minimum : input;
    input = input > parameters.maximum ? parameters.maximum : input;
  }
  else {
    NTA_CHECK(input >= parameters.minimum && input <= parameters.maximum)
        << "Input must be within range [minimum, maximum]!";
  }

  auto &dense = output.getDense();
  dense.assign( output.size, 0u );

  const UInt start = (UInt)(input - parameters.minimum) / parameters.resolution;
  const UInt end   = start + parameters.active;
  if( parameters.periodic ) {
    if( end >= output.size ) {
      std::fill(&dense[start], &dense[output.size],       1u);
      std::fill(&dense[0],     &dense[end - output.size], 1u);
    }
    else {
      std::fill(&dense[start], &dense[end], 1u);
    }
  }
  else {
    std::fill(&dense[start], &dense[end], 1u);
  }

  output.setDense( dense );
}

} // end namespace nupic
