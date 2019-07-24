/* -----------------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc. https://numenta.com
 *               2019, Brev Patterson, Lux Rota LLC, https://luxrota.com
 *               2019, David McDougall
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
 * SimHashDocumentEncoder.cpp
 * @author Brev Patterson, Lux Rota LLC, https://luxrota.com
 * @author David McDougall
 * @since 0.2.3
 */

#include <climits> // CHAR_BIT
#include <hasher.hpp> // digestpp: sha3+shake256 hash digests
#include <algorithm/sha3.hpp>
#include <algorithm/shake.hpp>

#include <htm/encoders/SimHashDocumentEncoder.hpp>


namespace htm {

  /**
   * Constructor
   * @see SimHashDocumentEncoder.hpp
   */
  SimHashDocumentEncoder::SimHashDocumentEncoder(const SimHashDocumentEncoderParameters &parameters)
  {
    initialize(parameters);
  }

  /**
   * Initialize
   * @see SimHashDocumentEncoder.hpp
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
   * Encode
   *
   * Depending on param "tokenSimilarity", we'll either loop through
   *  the provided document tokens, or loop through all the letters in all
   *  the tokens.
   * Each token/letter will be hashed with SHA3+SHAKE256 to get
   *  a varible-length (param "size") binary digest output. These vectors will
   *  be stored in a matrix for the next step of processing.
   * After the loop, we SimHash the matrix of hashes, resulting in an
   *  output SDR.
   *
   * @param :input: Document token strings to encode, ex: {"what","is","up"}.
   * @param :output: Result SDR to fill with encoding.
   * @see SimHashDocumentEncoder.hpp
   */
  void SimHashDocumentEncoder::encode(std::vector<std::string> input, SDR &output)
  {
    Eigen::VectorXi hashBits(args_.size);
    Eigen::MatrixXi hashes(args_.size, 0u);
    std::vector<UInt> simBits(args_.size, 0u);

    NTA_CHECK(input.size() > 0u)
      << "Encoding input vector array should have at least 1 string member.";

    output.zero();
    hashBits = Eigen::VectorXi::Zero(args_.size);

    for (auto token : input) {
      if (args_.tokenSimilarity) {
        // each token is itself simhashed (with hashes of token letter/chars)
        for (auto letter : token) {
          // generate hash digest for single character
          hashToken_(std::string(1u, letter), hashBits);
          addColumnToMatrix_(hashBits, hashes);
        }
      }
      else {
        // generate hash digest for whole token string
        hashToken_(token, hashBits);
        addColumnToMatrix_(hashBits, hashes);
      }
    }

    simHashMatrix_(hashes, simBits);
    output.setDense(simBits);
  } // end method encode

  /**
   * AddColumnToMatrix_
   * @param :column: Source eigen vector to be attached.
   * @param :matrix: Target eigen matrix that will be added to.
   */
  void SimHashDocumentEncoder::addColumnToMatrix_(Eigen::VectorXi column, Eigen::MatrixXi &matrix)
  {
    matrix.conservativeResize(Eigen::NoChange, matrix.cols() + 1u);
    matrix.col(matrix.cols() - 1u) = column;
  } // end method addColumnToMatrix_

  /**
   * HashToken_
   *
   * Hash (SHA3+SHAKE256 variable-length) a string into a byte digest.
   * Convert the byte vector to a binary vector and set output.
   *
   * @param :text: Source text to be hashed.
   * @param :bits: Eigen vector to store result binary hash digest in.
   */
  void SimHashDocumentEncoder::hashToken_(std::string text, Eigen::VectorXi &bits)
  {
    digestpp::shake256 hasher;
    std::vector<unsigned char> digest;
    hasher.absorb(text);
    hasher.squeeze((UInt) ((args_.size / CHAR_BIT) + 1u), back_inserter(digest));
    hashBytesToBits_(digest, bits);
  } // end method hashToken_

  /**
   * HashBytesToBits_
   *
   * Convert vector of hashed digest bytes to longer vector of bits.
   *
   * @param :bytes: Source hash digest eigen byte vector for binary conversion.
   * @param :bits: Eigen vector to store converted binary hash digest in.
   */
  void SimHashDocumentEncoder::hashBytesToBits_(std::vector<unsigned char> bytes, Eigen::VectorXi &bits)
  {
    UInt bitcount = 0u;
    bits = Eigen::VectorXi::Zero(args_.size);
    for (auto byte : bytes) {
      for (auto bit : std::bitset<CHAR_BIT>(byte).to_string()) {
        if (bitcount < args_.size) {
          bits(bitcount) = (UInt) (bit - '0');
          bitcount++;
        }
      }
    }
  } // end method hashBytesToBits_

  /**
   * SimHashMatrix_
   *
   * Create a SimHash SDR from Eigen vector array (matrix) of Hash digest bits.
   * Convert all binary 0 values in input matrix to integer -1 values, leaving
   *  us with "adders", instead of binary hashes.
   * Sum all these "adder" vectors to get a type of binary histogram.
   * Choose the desired number (activeBits) of max values, use their indices
   *  to set output On bits. Rest of bits are Off. We now have our result
   *  sparse SimHash. (In an ordinary dense SimHash, sums >= 0 become binary 1,
   *  the rest 0.)
   *
   * @param :hashes: Source eigen matrix of hashes to be simhashed.
   * @param :simhash: Stadard vector to store dense binary simhash result in.
   */
  void SimHashDocumentEncoder::simHashMatrix_(Eigen::MatrixXi hashes, std::vector<UInt> &simhash)
  {
    Eigen::VectorXi::Index maxIndex;  // array index of current max member
    Eigen::VectorXi sums(args_.size); // for bit sums of all hashes passed in
    Int minValue;                     // min value, used to neuter max vals during sparsify

    std::fill(simhash.begin(), simhash.end(), 0u);
    sums = Eigen::VectorXi::Zero(args_.size);

    // convert bit columns to adder columns (binary 0 => int -1)
    hashes = (hashes.array() == 0u).select(-1, hashes);
    // sum adder columns
    sums = hashes.rowwise().sum();

    // sparse simhash: top-N sums replaced with a binary 1, rest 0.
    minValue = sums.minCoeff();
    for (UInt bit = 0u; bit < args_.activeBits; bit++) {
      // get index of current max value from vector, set bit in output
      sums.maxCoeff(&maxIndex);
      simhash[maxIndex] = 1u;
      // neuter this max value so next iteration will get next highest max
      sums(maxIndex) = minValue;
    }
  } // end method simHashMatrix_

  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self)
  {
    out << "SimHashDocumentEncoder \n";
    out << "  activeBits:       " << self.parameters.activeBits       << ",\n";
    out << "  size:             " << self.parameters.size             << ",\n";
    out << "  sparsity:         " << self.parameters.sparsity         << ",\n";
    out << "  tokenSimilarity:  " << self.parameters.tokenSimilarity  << ",\n";
    return out;
  }

} // end namespace htm
