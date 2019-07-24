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

// @TODO py bind
// @TODO article


namespace htm {

  /**
   * SimHashDocumentEncoderParameters
   *
   * @param :activeBits: The number of true bits in the encoded output SDR.
   *  Specify only one of: activeBits or sparsity.
   * @param :size: Total number of bits in the encoded output SDR.
   * @param :sparsity: An alternative way to specify the member
   *  "activeBits". Sparsity requires that the size to also be specified.
   *  Specify only one of: activeBits or sparsity.
   * @param :tokenSimilarity: If True (default), similar tokens such as
   *  "cat" and "cats" will have very similar representations. If False,
   *  similar tokens ("cat", "cats") will have completely unrelated
   *  representations.
   */
  struct SimHashDocumentEncoderParameters {
    UInt activeBits = 0u;
    UInt size = 0u;
    Real sparsity = 0.0f;
    bool tokenSimilarity = true;
  }; // end struct SimHashDocumentEncoderParameters

  /**
   * SimHashDocumentEncoder
   *
   * High level API description here
   *
   * @see BaseEncoder.hpp
   * @see SimHashDocumentEncoder.cpp
   */
  class SimHashDocumentEncoder : public BaseEncoder<std::vector<std::string>> {
  public:
    SimHashDocumentEncoder() {};
    SimHashDocumentEncoder(const SimHashDocumentEncoderParameters &parameters);
    void initialize(const SimHashDocumentEncoderParameters &parameters);

    const SimHashDocumentEncoderParameters &parameters = args_;
    void encode(std::vector<std::string> input, SDR &output) override;

    CerealAdapter;  // see Serializable.hpp
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

    ~SimHashDocumentEncoder() override {};

  private:
    SimHashDocumentEncoderParameters args_;
    void addColumnToMatrix_(Eigen::VectorXi column, Eigen::MatrixXi &matrix);
    void hashBytesToBits_(std::vector<unsigned char> bytes, Eigen::VectorXi &bits);
    void hashToken_(std::string text, Eigen::VectorXi &bits);
    void simHashMatrix_(Eigen::MatrixXi hashes, std::vector<UInt> &simhash);
  }; // end class SimHashDocumentEncoder

  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self);

} // end namespace htm

#endif // end define NTA_ENCODERS_SIMHASH_DOCUMENT
