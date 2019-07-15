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
 * Define the SimHashDocumentEncoder
 */

#ifndef NTA_ENCODERS_SIMHASH_DOCUMENT
#define NTA_ENCODERS_SIMHASH_DOCUMENT

#include <htm/encoders/BaseEncoder.hpp>
#include <htm/types/Types.hpp>


namespace htm {

  /**
   * Encoder Parameters
   *  Specify only one of: activeBits or sparsity.
   */
  struct SimHashDocumentEncoderParameters {
    /**
     * Member "activeBits" is the number of true bits in the encoded output SDR.
     * Specify only one of: activeBits or sparsity.
     */
    UInt activeBits = 0u;

    /**
     * Member "sparsity" is an alternative way to specify the member
     * "activeBits". Sparsity requires that the size to also be specified.
     * Specify only one of: activeBits or sparsity.
     */
    Real sparsity = 0.0f;

    /**
     * Member "size" is the total number of bits in the encoded output SDR.
     */
    UInt size = 0u;

    /**
     * Member "tokens" is all token strings from all documents. This is the
     * corpus from which encodings will draw.
     */
    std::vector<std::string> tokens = {};
  }; // end struct SimHashDocumentEncoderParameters

  /**
   * @TODO
   */
  class SimHashDocumentEncoder : public BaseEncoder<std::vector<std::string>> {
  public:
    SimHashDocumentEncoder() {};
    SimHashDocumentEncoder(const SimHashDocumentEncoderParameters &parameters);
    void initialize(const SimHashDocumentEncoderParameters &parameters);

    const SimHashDocumentEncoderParameters &parameters = args_;

    void encode(std::vector<std::string> input, SDR &output) override;


    CerealAdapter;  // see Serializable.hpp
    // FOR Cereal Serialization
    template<class Archive>
    void save_ar(Archive& ar) const {
      std::string name = "SimHashDocumentEncoder";
      ar(cereal::make_nvp("name", name));
      ar(cereal::make_nvp("activeBits", args_.activeBits));
      ar(cereal::make_nvp("sparsity", args_.sparsity));
      ar(cereal::make_nvp("size", args_.size));
      ar(cereal::make_nvp("tokens", args_.tokens));
    }

    // FOR Cereal Deserialization
    template<class Archive>
    void load_ar(Archive& ar) {
      std::string name;
      ar(cereal::make_nvp("name", name));
      ar(cereal::make_nvp("activeBits", args_.activeBits));
      ar(cereal::make_nvp("sparsity", args_.sparsity));
      ar(cereal::make_nvp("size", args_.size));
      ar(cereal::make_nvp("tokens", args_.tokens));
    }

    ~SimHashDocumentEncoder() override {};

  private:
    SimHashDocumentEncoderParameters args_;
  }; // end class SimHashDocumentEncoder

  std::ostream & operator<<(std::ostream & out, const SimHashDocumentEncoder &self);

} // end namespace htm
#endif // end define NTA_ENCODERS_SIMHASH_DOCUMENT
