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
 * SimHashDocumentEncoder.cpp
 */

#include <algorithm>  // transform
#include <bitset>     // to_string
#include <cctype>     // tolower
#include <climits>    // CHAR_BIT
#include <regex>

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
    NTA_CHECK((parameters.activeBits > 0u) != (parameters.sparsity > 0.0f))
      << "Need only one argument of: 'activeBits' or 'sparsity'.";
    NTA_CHECK(parameters.size > 0u)
      << "Missing 'size' argument.";
    if ((parameters.tokenFrequencyCeiling > 0) && (parameters.tokenFrequencyFloor > 0)) {
      NTA_CHECK(parameters.tokenFrequencyCeiling > parameters.tokenFrequencyFloor)
        << "Argument 'tokenFrequencyCeiling' should be greater than argument 'tokenFrequencyFloor'.";
    }
    if (!parameters.tokenSimilarity) {
      NTA_CHECK(parameters.charFrequencyCeiling == 0)
        << "Argument 'charFrequencyCeiling' requires argument 'tokenSimilarity'.";
      NTA_CHECK(parameters.charFrequencyFloor == 0)
        << "Argument 'charFrequencyFloor' requires argument 'tokenSimilarity'.";
      if ((parameters.charFrequencyCeiling > 0) && (parameters.charFrequencyFloor > 0)) {
        NTA_CHECK(parameters.charFrequencyCeiling > parameters.charFrequencyFloor)
          << "Argument 'charFrequencyCeiling' should be greater than argument 'charFrequencyFloor'.";
      }
    }

    // Make local copy of arguments for processing
    args_ = parameters;

    // process: handle 'sparsity' param instead of 'activeBits' param
    if (args_.sparsity > 0.0f) {
      NTA_CHECK((args_.sparsity >= 0.0f) && (args_.sparsity <= 1.0f))
        << "Argument 'sparsity' must be a float in the range 0.0-1.0.";
      NTA_CHECK(args_.size > 0u)
        << "Argument 'sparsity' requires that the 'size' also be given.";
      args_.activeBits = (UInt) round(args_.size * args_.sparsity);
    }
    // Process: handle internal case insensitivity needs
    if (!args_.caseSensitivity) {
      // exclusions => case insensitive
      if (!args_.excludes.empty()) {
        std::vector<std::string> excludesLower;
        for (auto& token : args_.excludes) {
          transform(token.begin(), token.end(), token.begin(), ::tolower);
          excludesLower.push_back(token);
        }
        args_.excludes = excludesLower;
      }
      // vocabulary => case insensitive
      if (args_.vocabulary.size() > 0) {
        std::map<std::string, UInt> vocabLower;
        for (const auto& pair : args_.vocabulary) {
          std::string token = pair.first;
          const UInt weight = pair.second;
          transform(token.begin(), token.end(), token.begin(), ::tolower);
          vocabLower[token] = weight;
        }
        args_.vocabulary = vocabLower;
      }
    }

    // Post-processing, Sanity check the parameters
    NTA_CHECK(args_.size > 0u);
    NTA_CHECK(args_.activeBits > 0u);
    NTA_CHECK(args_.activeBits < args_.size);

    // Initialize parent class with finalized params
    BaseEncoder<std::vector<std::string>>::initialize({ args_.size });
  } // end method initialize

  /**
   * Encode (Main calling style)
   * @see SimHashDocumentEncoder.hpp
   * @see encode(std::string input, SDR &output)
   */
  void SimHashDocumentEncoder::encode(const std::vector<std::string> input, SDR &output)
  {
    NTA_CHECK(input.size() > 0u)
      << "Encoding input vector array should have at least 1 string member.";

    Eigen::MatrixXi adders(args_.size, 0u);
    Eigen::VectorXi hashBits(args_.size);
    std::map<std::string, UInt> histogramToken;
    SDR result({ args_.size });
    std::vector<UInt> simBits(args_.size, 0u);

    hashBits = Eigen::VectorXi::Zero(args_.size);
    result.zero();

    LogItem::setLogLevel(LogLevel::LogLevel_Verbose);
    NTA_DEBUG << "! input.size = " << input.size();

    for (const auto& member : input) {
      std::string token = member;
      UInt tokenWeight;
      std::map<std::string, UInt> histogramChar;

      // caseSensitivity
      if (!args_.caseSensitivity) {
        transform(token.begin(), token.end(), token.begin(), ::tolower);
      }

      // excludes
      if (!args_.excludes.empty()) {
        if(std::find(args_.excludes.begin(), args_.excludes.end(), token) != args_.excludes.end()) {
          continue; // skip this excluded token
        }
      }

      // vocabulary + encodeOrphans
      if (args_.vocabulary[token]) {
        tokenWeight = args_.vocabulary[token];
      }
      else {
        if (args_.encodeOrphans) {
          tokenWeight = 1; // encode non-vocab token with weight 1
        }
        else {
          continue; // skip non-vocab token
        }
      }

      // token frequency floor and ceiling
      if (histogramToken.count(token) <= 0) { histogramToken[token] = 1; }
      else { histogramToken[token]++; }
      LogItem::setLogLevel(LogLevel::LogLevel_Verbose);
      NTA_DEBUG << "- hist  " << token << " " << histogramToken[token];
      if ((histogramToken[token] < args_.tokenFrequencyFloor) ||
          (histogramToken[token] > args_.tokenFrequencyCeiling) ) {
        NTA_DEBUG << "-- skipping";
        continue;  // discard token
      }

      // tokenSimilarity
      if (args_.tokenSimilarity) {
        // generate hash digest for every single character individually
        for (const auto& letter : token) {
          const std::string letterStr = std::string(1u, letter);
          UInt charWeight = args_.vocabulary[letterStr] ?
                            args_.vocabulary[letterStr] : tokenWeight;

          // char frequency floor and ceiling
          // if (histogramChar.count(letterStr) <= 0) {
          //   histogramChar[letterStr] = 1;
          // }
          // else {
          //   histogramChar[letterStr]++;
          // }
          // if ((histogramChar[letterStr] < args_.charFrequencyFloor) ||
          //     (histogramChar[letterStr] > args_.charFrequencyCeiling) ) {
          //   continue;  // discard char
          // }

          LogItem::setLogLevel(LogLevel::LogLevel_Verbose);
          NTA_DEBUG << "  " << letterStr;

          // hash character
          hashToken_(letterStr, hashBits);
          bitsToWeightedAdder_(charWeight, hashBits);
          addVectorToMatrix_(hashBits, adders);
        }
        tokenWeight = (UInt) (tokenWeight * 1.5); // try to balance token with letters
      }

      // generate hash digest for whole token string
      LogItem::setLogLevel(LogLevel::LogLevel_Verbose);
      NTA_DEBUG << " " << token;
      hashToken_(token, hashBits);
      bitsToWeightedAdder_(tokenWeight, hashBits);
      addVectorToMatrix_(hashBits, adders);
    }

    // simhash
    simHashAdders_(adders, simBits);
    result.setDense(simBits);
    output.setDense(result.getDense());
  } // end method encode

  /**
   * Encode (Alternate calling style: Simple string method)
   * @see SimHashDocumentEncoder.hpp
   * @see encode(const std::vector<std::string> input, SDR &output)
   */
  void SimHashDocumentEncoder::encode(const std::string input, SDR &output)
  {
    std::regex spaces("\\s+");
    std::sregex_token_iterator iterate(input.begin(), input.end(), spaces, -1);
    std::sregex_token_iterator end;
    std::vector<std::string> inputSplit(iterate, end);
    encode(inputSplit, output);
  } // end method encode (string alternate)

  /**
   * AddVectorToMatrix_
   * @see SimHashDocumentEncoder.hpp
   */
  void SimHashDocumentEncoder::addVectorToMatrix_(const Eigen::VectorXi vector, Eigen::MatrixXi &matrix)
  {
    matrix.conservativeResize(Eigen::NoChange, matrix.cols() + 1u);
    matrix.col(matrix.cols() - 1u) = vector;
  } // end method addVectorToMatrix_

  /**
   * BitsToWeightedAdder_
   * @see SimHashDocumentEncoder.hpp
   */
  void SimHashDocumentEncoder::bitsToWeightedAdder_(const UInt weight, Eigen::VectorXi &vector)
  {
    // convert hash bit columns to int adder columns (0 => -1)
    vector = (vector.array() == 0u).select(-1, vector);
    // weight the adders
    vector *= weight;
  } // end method bitsToWeightedAdder_

  /**
   * bytesToBits_
   * @see SimHashDocumentEncoder.hpp
   */
  void SimHashDocumentEncoder::bytesToBits_(const std::vector<unsigned char> bytes, Eigen::VectorXi &bits)
  {
    UInt bitcount = 0u;
    bits = Eigen::VectorXi::Zero(args_.size);

    for (const auto& byte : bytes) {
      for (const auto& bit : std::bitset<CHAR_BIT>(byte).to_string()) {
        bits(bitcount) = (UInt) (bit - '0');
        bitcount++;
        if (bitcount >= args_.size) break;
      }
    }
  } // end method bytesToBits_

  /**
   * HashToken_
   * @see SimHashDocumentEncoder.hpp
   */
  void SimHashDocumentEncoder::hashToken_(const std::string token, Eigen::VectorXi &hashBits)
  {
    digestpp::shake256 hasher;
    std::vector<unsigned char> digest;

    hasher.absorb(token);
    hasher.squeeze((UInt) ((args_.size / CHAR_BIT) + 1u), back_inserter(digest));
    bytesToBits_(digest, hashBits);
  } // end method hashToken_

  /**
   * SimHashAdders_
   * @see SimHashDocumentEncoder.hpp
   */
  void SimHashDocumentEncoder::simHashAdders_(const Eigen::MatrixXi adders, std::vector<UInt> &simhash)
  {
    Eigen::VectorXi::Index maxIndex;  // array index of current max member
    Int minValue;                     // min value, used to neuter max vals during sparsify
    Eigen::VectorXi sums(args_.size); // for bit sums of all hashes passed in

    std::fill(simhash.begin(), simhash.end(), 0u);
    sums = Eigen::VectorXi::Zero(adders.cols());

    // sum adder columns
    sums = adders.rowwise().sum();
    // sparse simhash: top-N sums replaced with a binary 1, rest 0.
    minValue = sums.minCoeff();
    for (UInt bit = 0u; bit < args_.activeBits; bit++) {
      // get index of current max value from vector, set bit in output
      sums.maxCoeff(&maxIndex);
      simhash[maxIndex] = 1u;
      // neuter this max value so next iteration will get next highest max
      sums(maxIndex) = minValue;
    }
  } // end method simHashAdders_


  // Debug
  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self)
  {
    out << "SimHashDocumentEncoder \n";
    out << "  activeBits:            " << self.parameters.activeBits            << ",\n";
    out << "  caseSensitivity:       " << self.parameters.caseSensitivity       << ",\n";
    out << "  charFrequencyCeiling:  " << self.parameters.charFrequencyCeiling  << ",\n";
    out << "  charFrequencyFloor:    " << self.parameters.charFrequencyFloor    << ",\n";
    out << "  encodeOrphans:         " << self.parameters.encodeOrphans         << ",\n";
    out << "  excludes.size:         " << self.parameters.excludes.size()       << ",\n";
    out << "  size:                  " << self.parameters.size                  << ",\n";
    out << "  sparsity:              " << self.parameters.sparsity              << ",\n";
    out << "  tokenFrequencyCeiling: " << self.parameters.tokenFrequencyCeiling << ",\n";
    out << "  tokenFrequencyFloor:   " << self.parameters.tokenFrequencyFloor   << ",\n";
    out << "  tokenSimilarity:       " << self.parameters.tokenSimilarity       << ",\n";
    out << "  vocabulary.size:       " << self.parameters.vocabulary.size()     << ",\n";
    return out;
  }

} // end namespace htm
