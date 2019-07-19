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

#include <bitset>
#include <climits> // CHAR_BIT
#include <string>
#include <htm/encoders/SimHashDocumentEncoder.hpp>

// digestpp: sha3+shake256 hash digests
#include <hasher.hpp>
#include <algorithm/sha3.hpp>
#include <algorithm/shake.hpp>


namespace htm {

  /**
   * Convert vector of bytes to longer vector of bits
   */
  SDR_dense_t hashDigestBytesToBits(std::vector<unsigned char> bytes) {
    SDR_dense_t bits;
    for (auto byte : bytes) {
      for (auto bit : std::bitset<CHAR_BIT>(byte).to_string()) {
        bits.push_back((bool) ((UInt) (bit - '0')));
      }
    }
    return bits;
  }

  /**
   * Create a SimHash SDR from a vector array of Hash SDR's
   */
  SDR simHash(std::vector<SDR> hashes) {
    SDR result;
    return result;
  }


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

    // Post-processing, Sanity check the parameters
    NTA_CHECK(args_.size > 0u);
    NTA_CHECK(args_.activeBits > 0u);
    NTA_CHECK(args_.activeBits < args_.size);

    // Initialize parent class with finalized params
    BaseEncoder<std::vector<std::string>>::initialize({ args_.size });
  } // end method initialize

  /**
   * Encoding method.
   */
  void SimHashDocumentEncoder::encode(std::vector<std::string> input, SDR &output)
  {
    NTA_CHECK(input.size() > 0u)
      << "Encoding input vector array should have at least 1 string member.";

    digestpp::shake256 hasher;

    // split token list into individual tokens
    for (auto token : input) {
      LogItem::setLogLevel(htm::LogLevel_Verbose);
      NTA_DEBUG << token;

      if (args_.tokenSimilarity) {
        // each token is itself simhashed (each letter/char is individually hashed)
        // split token into individual letter-characters
        for (auto letter : token) {
          // generate hash digest for single character
          std::vector<unsigned char> digest;
          hasher.absorb(std::string (1, letter));
          hasher.squeeze(args_.size / CHAR_BIT, back_inserter(digest));
          hasher.reset();
          SDR_dense_t bits = hashDigestBytesToBits(digest);
          SDR hash({args_.size});
          hash.setDense(bits);
          NTA_DEBUG << "  " << letter << " " << hash;
        }
      } // end if tokenSimilarity
      else {
        // generate hash digest for whole token string
        std::vector<unsigned char> digest;
        hasher.absorb(token);
        hasher.squeeze(args_.size / CHAR_BIT, back_inserter(digest));
        hasher.reset();
        SDR_dense_t bits = hashDigestBytesToBits(digest);
        SDR hash({args_.size});
        hash.setDense(bits);
        NTA_DEBUG << "  " << hash;
      } // end else tokenSimilarity
    }
  } // end method encode

  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self)
  {
    out << "SimHashDocumentEncoder \n";
    out << "  activeBits:       " << self.parameters.activeBits       << ",\n";
    out << "  sparsity:         " << self.parameters.sparsity         << ",\n";
    out << "  size:             " << self.parameters.size             << ",\n";
    out << "  tokenSimilarity:  " << self.parameters.tokenSimilarity  << ",\n";
    return out;
  }

} // end namespace htm
