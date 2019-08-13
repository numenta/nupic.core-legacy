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
 * SimHashDocumentEncoder.hpp
 */

#ifndef NTA_ENCODERS_SIMHASH_DOCUMENT
#define NTA_ENCODERS_SIMHASH_DOCUMENT

#include <Eigen/Dense>
#include <string>
#include <vector>

#include <htm/encoders/BaseEncoder.hpp>
#include <htm/types/Types.hpp>


namespace htm {

  struct SimHashDocumentEncoderParameters {
    /**
     * @param :activeBits: The number of true bits in the encoded output SDR.
     *  Specify only one of: activeBits or sparsity.
     */
    UInt activeBits = 0u;

    /**
     * @param :caseSensitivity: Should capitalized English letters (A-Z) have
     *  differing influence on our output than their lower-cased (a-z)
     *  counterparts?
     *    If True:  "DOGS" and "dogs" will have completely different encodings.
     *    If False: "DOGS" and "dogs" will share the same encoding (Default).
     */
    bool caseSensitivity = false;

    /**
     * @param :size: Total number of bits in the encoded output SDR.
     */
    UInt size = 0u;

    /**
     * @param :sparsity: An alternative way to specify the member
     *  "activeBits". Sparsity requires that the size to also be specified.
     *  Specify only one of: activeBits or sparsity.
     */
    Real sparsity = 0.0f;

    /**
     * @param :tokenSimilarity: In addition to document similarity, we can also
     *  achieve a kind of token similarity. Default is FALSE (providing better
     *  document-level similarity, at the expense of token-level similarity).
     *  This could be use to meaningfully encode plurals and mis-spellings as
     *  similar. It may also be hacked to create a complex dimensional category
     *  encoder. Results are heavily dependent on the content of your
     *  input data.
     *    If TRUE: Similar tokens ("cat", "cats") will have similar influence
     *      on the output simhash. This benefit comes with the cost of a
     *      probable reduction in document-level similarity accuracy.
     *    If FALSE: Similar tokens ("cat", "cats") will have individually unique
     *      and unrelated influence on the output simhash encoding, thus losing
     *      token-level similarity and increasing document-level similarity.
     */
    bool tokenSimilarity = false;
  }; // end struct SimHashDocumentEncoderParameters


  /**
   * SimHashDocumentEncoder
   *
   * Encodes documents and text into Sparse Distributed Representations (SDRs),
   * ready for use with Hierarchical Temporal Memory (HTM). Similar document
   * encodings will share similar representations, and vice versa. Unicode
   * is supported. No lookup tables are used.
   *
   * "Similarity" here refers to bitwise similarity (small hamming distance,
   * high overlap), not semantic similarity (encodings for "apple" and
   * "computer" will have no relation here.) For document encodings which are
   * also semantic, please try Cortical.io and their Semantic Folding encoding.
   *
   * In addition to document similarity, an option is provided to toggle if
   * token similarity "near-spellings" (such as "cat" and "cats") will receieve
   * similar encodings or not. Another option is provided for manging
   * English-language token case in/sensitivity.
   *
   * Definition of Terms:
   *    A "corpus" is a collection of "documents".
   *    A "document" is made up of "tokens" (or "words").
   *    A "token" is made up of "characters" (or "letters").
   *
   * For details on the SimHash Algorithm itself, please see nearby file:
   *    SimHashDocumentEncoder.README.md
   *
   * @code
   *    #include <htm/encoders/SimHashDocumentEncoder.hpp>
   *    #include <htm/encoders/types/Sdr.hpp>
   *
   *    SimHashDocumentEncoderParameters params;
   *    params.size = 400u;
   *    params.activeBits = 21u;
   *
   *    SDR output({ params.size });
   *    SimHashDocumentEncoder encoder(params);
   *    encoder.encode({{ "bravo", 2 }, { "delta", 1 }, { "echo", 3 }}, output)
   *    encoder.encode({ "bravo", "delta", "echo" }, output)  // weights 1
   *    encoder.encode("bravo delta echo", output)            // weights 1
   *
   * @see BaseEncoder.hpp
   * @see SimHashDocumentEncoder.cpp
   * @see SimHashDocumentEncoder.README.md
   */
  class SimHashDocumentEncoder : public BaseEncoder<std::map<std::string, UInt>> {
  public:
    /**
     * Constructor
     * @public
     * @see BaseEncoder.hpp
     */
    SimHashDocumentEncoder() {};
    SimHashDocumentEncoder(const SimHashDocumentEncoderParameters &parameters);

    /**
     * Initialize
     * @public
     * @see BaseEncoder.hpp
     */
    void initialize(const SimHashDocumentEncoderParameters &parameters);

    // Public Params
    const SimHashDocumentEncoderParameters &parameters = args_;

    /**
     * Encode (Main calling style)
     *
     * Each token/letter will be hashed with SHA3+SHAKE256 to get a binary
     *  digest output of desired `size`. These vectors will be stored in a
     *  matrix for the next step of processing.
     * Weights are added in during hashing and simhashing.
     * After the loop, we SimHash the matrix of hashes, resulting in an
     *  output SDR.
     * If param "caseSensitivity" is passed, we'll lowercase all token strings.
     * If param "tokenSimilarity" is passed, we'll additionally loop through
     *  all the letters in all the tokens.
     *
     * @param :input: Document token strings (with weights) to encode,
     *  ex: {{ "what", 3 }, { "is", 1 }, { "up", 2 }}.
     *  Documents can contain any number of tokens > 0. Token order in the
     *  document is ignored and does not effect the output encoding.
     * @param :output: Result SDR to fill with result output encoding.
     *
     * @public
     * @see SimHashDocumentEncoder.hpp
     * @see encode(const std::vector<std::string> input, SDR &output)
     * @see encode(std::string input, SDR &output)
     */
    void encode(const std::map<std::string, UInt> input, SDR &output) override;

    /**
     * Encode (Alternate calling style: Simple token list method)
     *
     * An alternate simple list calling method for Encode, no weights need to be
     * passed in with tokens (all weights assumed to be 1).
     *
     * @param :input: Document token strings to encode, ex: {"what","is","up"}.
     *  Documents can contain any number of tokens > 0. Token order in the
     *  document is ignored and does not effect the output encoding.
     * @param :output: Result SDR to fill with result output encoding.
     *
     * @public
     * @see encode(const std::map<std::string, UInt> input, SDR &output)
     * @see encode(std::string input, SDR &output)
     */
    void encode(const std::vector<std::string> input, SDR &output);

    /**
     * Encode (Alternate calling style: Simple string method)
     *
     * An alternate simple string calling method for Encode, no weights need to
     * be passed in with tokens (all weights assumed to be 1). String will be
     * split into tokens based on empty whitespace characters, after trimming.
     *
     * @param :input: Document token string to encode, ex: "what is up".
     *  String will be split into tokens based on empty whitespace characters,
     *  after trimming. Documents can contain any number of tokens > 0. Token
     *  order in the document is ignored and does not effect the
     *  output encoding.
     * @param :output: Result SDR to fill with result output encoding.
     *
     * @public
     * @see encode(const std::map<std::string, UInt> input, SDR &output)
     * @see encode(const std::vector<std::string> input, SDR &output)
     */
    void encode(const std::string input, SDR &output);

    /**
     * Serialization
     */
    CerealAdapter;
    // Cereal Serialize
    template<class Archive>
    void save_ar(Archive& ar) const {
      std::string name = "SimHashDocumentEncoder";
      ar(cereal::make_nvp("name", name));
      ar(cereal::make_nvp("activeBits", args_.activeBits));
      ar(cereal::make_nvp("caseSensitivity", args_.caseSensitivity));
      ar(cereal::make_nvp("sparsity", args_.sparsity));
      ar(cereal::make_nvp("size", args_.size));
      ar(cereal::make_nvp("tokenSimilarity", args_.tokenSimilarity));
    }
    // Cereal Deserialize
    template<class Archive>
    void load_ar(Archive& ar) {
      std::string name;
      ar(cereal::make_nvp("name", name));
      ar(cereal::make_nvp("activeBits", args_.activeBits));
      ar(cereal::make_nvp("caseSensitivity", args_.caseSensitivity));
      ar(cereal::make_nvp("sparsity", args_.sparsity));
      ar(cereal::make_nvp("size", args_.size));
      ar(cereal::make_nvp("tokenSimilarity", args_.tokenSimilarity));
      BaseEncoder<std::map<std::string, UInt>>::initialize({ args_.size });
    }

    ~SimHashDocumentEncoder() override {};
    // end public

  private:
    // Private Params
    SimHashDocumentEncoderParameters args_;

    /**
     * AddVectorToMatrix_
     *
     * @param :vector: Source eigen vector column to be attached.
     * @param :matrix: Target eigen matrix that will be added to.
     * @private
     */
    void addVectorToMatrix_(const Eigen::VectorXi vector, Eigen::MatrixXi &matrix);

    /**
     * BitsToWeightedAdder_
     *
     * Take the bits from a hash, convert 0 to -1, and multiply by provided
     *  weighting factor. For example:
     *    In Column   = { 0, 1,  0,  0, 1,  0}
     *    In Weight   = 3
     *    Out Result  = {-3, 3, -3, -3, 3, -3}
     *
     * @param :weight: Weight to add to column (positive integer, usually 1).
     * @param :vector: Target eigen vector column to add weighting to.
     * @private
     */
    void bitsToWeightedAdder_(const UInt weight, Eigen::VectorXi &vector);

    /**
     * BytesToBits_
     *
     * Convert vector of hashed digest bytes to longer vector of bits.
     *
     * @param :bytes: Source hash (eigen byte vector) for binary conversion.
     * @param :bits: Eigen vector to store converted binary hash digest in.
     * @private
     */
    void bytesToBits_(const std::vector<unsigned char> bytes, Eigen::VectorXi &bits);

    /**
     * HashToken_
     *
     * Hash (SHA3+SHAKE256) a string into a byte digest.
     * Convert the byte vector to a binary vector and set output.
     *
     * @param :token: Source text to be hashed.
     * @param :hashBits: Eigen vector to store result binary hash digest in.
     * @private
     */
    void hashToken_(const std::string token, Eigen::VectorXi &hashBits);

    /**
     * SimHashAdders_
     *
     * Create a SimHash SDR from Eigen vector array (matrix) of Hash digest bits
     *  (in slightly modified "Adder" SimHash form).
     * Sum all these "Adder" vectors to get a type of binary histogram.
     * Choose the desired number (activeBits) of max values, use their indices
     *  to set output On bits. Rest of bits are Off. We now have our result
     *  sparse SimHash. (In an ordinary dense SimHash, sums >= 0 become
     *  binary 1, the rest 0.)
     *
     * @param :hashes: Source eigen matrix of hashes to be simhashed.
     * @param :simhash: Stadard vector to store dense binary simhash result in.
     * @private
     */
    void simHashAdders_(const Eigen::MatrixXi adders, std::vector<UInt> &simhash);
    // end private

  }; // end class SimHashDocumentEncoder


  // Debug
  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self);

} // end namespace htm

#endif // end define NTA_ENCODERS_SIMHASH_DOCUMENT
