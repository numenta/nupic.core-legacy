/* -----------------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc. https://numenta.com
 *               2019, David McDougall
 *               2019, Brev Patterson, Lux Rota LLC, https://luxrota.com
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero Public License version 3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for
 * more details.
 *
 * You should have received a copy of the GNU Affero Public License along with
 * this program.  If not, see http://www.gnu.org/licenses.
 * -------------------------------------------------------------------------- */

/** @file
 * Implementation of the SimHashDocumentEncoder
 */

#include <algorithm> // std::min std::sort
#include <numeric>   // std::iota
#include <cmath>     // std::isnan std::nextafter
#include <htm/encoders/SimHashDocumentEncoder.hpp>


namespace htm {

  SimHashDocumentEncoder::SimHashDocumentEncoder(const SimHashDocumentEncoderParameters &parameters)
    { initialize(parameters); }

  /**
   * Initialization method.
   */
  void SimHashDocumentEncoder::initialize(const SimHashDocumentEncoderParameters &parameters)
  {
    // Pre-processing, Check parameters
    NTA_CHECK((parameters.activeBits > 0u) || (parameters.sparsity > 0.0f))
      << "Need only one argument of: 'activeBits' or 'sparsity'.";
    NTA_CHECK(parameters.size > 0u)
      << "Missing 'size' argument.";
    NTA_CHECK(parameters.tokens.size() > 0u)
      << "Missing document corpus 'tokens', expecting at least 1 string member.";

    // Make local copy of arguments for processing
    args_ = parameters;

    // process: handle 'sparsity' param instead of 'activeBits' param
    if(args_.sparsity > 0.0f) {
      NTA_CHECK((args_.sparsity >= 0.0f) && (args_.sparsity <= 1.0f))
        << "Argument 'sparsity' must be a float in the range 0.0-1.0.";
      NTA_CHECK(args_.size > 0u)
        << "Argument 'sparsity' requires that the 'size' also be given.";
      args_.activeBits = (UInt) round(args_.size * args_.sparsity);
    }

    // process: make sure corpus token list is sorted
    std::sort(args_.tokens.begin(), args_.tokens.end());

    // Post-processing, Sanity check the parameters
    NTA_CHECK(args_.size > 0u);
    NTA_CHECK(args_.activeBits > 0u);
    NTA_CHECK(args_.activeBits < args_.size);
    NTA_CHECK(args_.tokens.size() > 0u);

    // Initialize parent class with finalized params
    BaseEncoder<std::vector<std::string>>::initialize({ args_.size });
  } // end method initialize

  /**
   * Encoding method.
   */
  void SimHashDocumentEncoder::encode(std::vector<std::string> input, SDR &output)
  {
    // verify input tokens were passed in
    NTA_CHECK(input.size() > 0u)
      << "Encoding input vector array should have at least 1 string member.";
    // verify input tokens against corpus tokens (from init)
    std::sort(input.begin(), input.end());
    NTA_CHECK(std::includes(args_.tokens.begin(), args_.tokens.end(), input.begin(), input.end()))
      << "Unrecognized input token, all 'tokens' required during encoder init.";

    // LogItem::setLogLevel(htm::LogLevel_Verbose);
    // for (auto i : input)
    //  NTA_DEBUG << "hi";

  } // end method encode

  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self)
  {
    out << "SimHashDocumentEncoder \n";
    out << "  activeBits:" << self.parameters.activeBits << ",\n";
    out << "  sparsity:  " << self.parameters.sparsity   << ",\n";
    out << "  size:      " << self.parameters.size       << ",\n";
    out << "  # tokens:  " << self.parameters.tokens.size() << std::endl;
    return out;
  }

} // end namespace htm
