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
 * SimHashDocumentEncoder.hpp
 * @author Brev Patterson, Lux Rota LLC, https://luxrota.com
 * @author David McDougall
 * @since 0.2.3
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
     *  Results are heavily dependent on the content of your input data.
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
   * is supported.
   *
   * "Similarity" here refers to bitwise similarity (small hamming distance,
   * high overlap), not semantic similarity (encodings for "apple" and
   * "computer" will have no relation here.) For document encodings which are
   * also semantic, please try Cortical.io and their Semantic Folding encoding.
   *
   * Encoding is accomplished using SimHash, a Locality-Sensitive Hashing (LSH)
   * algorithm from the world of nearest-neighbor document similarity search.
   * As SDRs are variable-length, we use the SHA3+SHAKE256 hashing algorithm.
   * We deviate slightly from the standard SimHash algorithm in order to
   * achieve sparsity.
   *
   * In addition to document similarity, an option is provided to toggle if
   * token similarity "near-spellings" (such as "cat" and "cats") will receieve
   * similar encodings or not.
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
   *    encoder.encode({ "bravo", "delta", "echo" }, output)  // weights 1
   *    encoder.encode({{ "bravo", 2 }, { "delta", 1 }, { "echo", 3 }}, output)
   *
   * @see BaseEncoder.hpp
   * @see SimHashDocumentEncoder.cpp
   * @see Serializable.hpp
   */
  class SimHashDocumentEncoder : public BaseEncoder<std::vector<std::string>> {
  public:
    SimHashDocumentEncoder() {};
    SimHashDocumentEncoder(const SimHashDocumentEncoderParameters &parameters);
    void initialize(const SimHashDocumentEncoderParameters &parameters);

    const SimHashDocumentEncoderParameters &parameters = args_;

    void encode(const std::map<std::string, UInt> input, SDR &output);
    void encode(const std::vector<std::string> input, SDR &output) override;

    ~SimHashDocumentEncoder() override {};

    CerealAdapter;
    // Cereal Serialization
    template<class Archive>
    void save_ar(Archive& ar) const {
      std::string name = "SimHashDocumentEncoder";
      ar(cereal::make_nvp("name", name));
      ar(cereal::make_nvp("activeBits", args_.activeBits));
      ar(cereal::make_nvp("sparsity", args_.sparsity));
      ar(cereal::make_nvp("size", args_.size));
      ar(cereal::make_nvp("tokenSimilarity", args_.size));
    }
    // Cereal Deserialization
    template<class Archive>
    void load_ar(Archive& ar) {
      std::string name;
      ar(cereal::make_nvp("name", name));
      ar(cereal::make_nvp("activeBits", args_.activeBits));
      ar(cereal::make_nvp("sparsity", args_.sparsity));
      ar(cereal::make_nvp("size", args_.size));
      ar(cereal::make_nvp("tokenSimilarity", args_.size));
    }

  private:
    SimHashDocumentEncoderParameters args_;
    void addVectorToMatrix_(const Eigen::VectorXi vector, Eigen::MatrixXi &matrix);
    void bitsToWeightedAdder_(const UInt weight, Eigen::VectorXi &vector);
    void bytesToBits_(const std::vector<unsigned char> bytes, Eigen::VectorXi &bits);
    void hashToken_(const std::string token, Eigen::VectorXi &hashBits);
    void simHashAdders_(const Eigen::MatrixXi adders, std::vector<UInt> &simhash);
  }; // end class SimHashDocumentEncoder

  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self);

} // end namespace htm

#endif // end define NTA_ENCODERS_SIMHASH_DOCUMENT
