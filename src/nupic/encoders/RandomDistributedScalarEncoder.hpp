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
 * --------------------------------------------------------------------- */

/** @file
 * Define the RandomDistributedScalarEncoder
 */

#ifndef NTA_ENCODERS_RDSE
#define NTA_ENCODERS_RDSE

#include <nupic/types/Sdr.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/encoders/BaseEncoder.hpp>

namespace nupic {
namespace encoders {

/**
 * TODO DOCUMENTATION
 * https://arxiv.org/pdf/1602.05925.pdf
      COPY FROM NUPIC!
      COPY FROM MY OLD WORK?
 */

struct RDSE_Parameters
{
  UInt size = 0u;
  UInt activeBits = 0u;
  Real sparsity = 0.0f;
  Real radius = 0.0f;
  Real resolution = 0.0f;
  UInt seed = 0u;
};

class RandomDistributedScalarEncoder : public BaseEncoder<Real64>
{
public:
  RandomDistributedScalarEncoder() {}
  RandomDistributedScalarEncoder( const RDSE_Parameters &parameters );
  void initialize( const RDSE_Parameters &parameters );

  const RDSE_Parameters &parameters = args_;

  /**
   * TODO DOCUMENTATION
   */
  void encode(Real64 input, SDR &output) override;

  void save(std::ostream &stream) const override;
  void load(std::istream &stream) override;

  ~RandomDistributedScalarEncoder() override {};

private:
  RDSE_Parameters args_;
};

typedef RandomDistributedScalarEncoder RDSE;

}      // End namespace encoders
}      // End namespace nupic
#endif // End ifdef NTA_ENCODERS_RDSE
