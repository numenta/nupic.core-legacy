/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
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
 * --------------------------------------------------------------------- */

/** @file
 * Define the RandomDistributedScalarEncoder
 */

#ifndef NTA_ENCODERS_RDSE
#define NTA_ENCODERS_RDSE

#include <htm/encoders/BaseEncoder.hpp>
#include <htm/utils/Log.hpp>

namespace htm {

/**
 * Parameters for the RandomDistributedScalarEncoder (RDSE)
 *
 * Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
 * of them.
 *
 * Members "radius", "resolution", & "category" are mutually exclusive, specify
 * exactly one of them.
 */
struct RDSE_Parameters
{
  /**
   * Member "size" is the total number of bits in the encoded output SDR.
   */
  UInt size = 0u;

  /**
   * Member "activeBits" is the number of true bits in the encoded output SDR.
   */
  UInt activeBits = 0u;

  /**
   * Member "sparsity" is the fraction of bits in the encoded output which this
   * encoder will activate. This is an alternative way to specify the member
   * "activeBits".
   */
  Real sparsity = 0.0f;

  /**
   * Member "radius" Two inputs separated by more than the radius have
   * non-overlapping representations. Two inputs separated by less than the
   * radius will in general overlap in at least some of their bits. You can
   * think of this as the radius of the input.
   */
  Real radius = 0.0f;

  /**
   * Member "resolution" Two inputs separated by greater than, or equal to the
   * resolution are guaranteed to have different representations.
   */
  Real resolution = 0.0f;

  /**
   * Member "category" means that the inputs are enumerated categories.
   * If true then this encoder will only encode unsigned integers, and all
   * inputs will have unique / non-overlapping representations.
   */
  bool category = false;

  /**
   * Member "seed" forces different encoders to produce different outputs, even
   * if the inputs and all other parameters are the same.  Two encoders with the
   * same seed, parameters, and input will produce identical outputs.
   *
   * The seed 0 is special.  Seed 0 is replaced with a random number.
   */
  UInt seed = 0u;
};

/**
 * Encodes a real number as a set of randomly generated activations.
 *
 * Description:
 * The RandomDistributedScalarEncoder (RDSE) encodes a numeric scalar (floating
 * point) value into an SDR.  The RDSE is more flexible than the ScalarEncoder.
 * This encoder does not need to know the minimum and maximum of the input
 * range.  It does not assign an input->output mapping at construction.  Instead
 * the encoding is determined at runtime.
 *
 * Note: This implementation differs from Numenta's original RDSE.  The original
 * RDSE saved all associations between inputs and active bits for the lifetime
 * of the encoder.  This allowed it to guarantee a good set of random
 * activations which didn't conflict with any previous encoding.  It also allowed
 * the encoder to decode an SDR into the input value which likely created it.
 * This RDSE does not save the association between inputs and active bits.  This
 * is faster and uses less memory.  It relies on the random & distributed nature
 * of SDRs to prevent conflicts between different encodings.  This method does
 * not allow for decoding SDRs into the inputs which likely created it.
 *
 * To inspect this run:
 * $ python -m htm.examples.encoders.rdse --help
 */
class RandomDistributedScalarEncoder : public BaseEncoder<Real64>
{
public:
  RandomDistributedScalarEncoder() {}
  RandomDistributedScalarEncoder( const RDSE_Parameters &parameters );
  void initialize( const RDSE_Parameters &parameters );

  const RDSE_Parameters &parameters = args_;

  void encode(Real64 input, SDR &output) override;


  ~RandomDistributedScalarEncoder() override {};

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    std::string name = "RandomDistributedScalarEncoder";
    ar(cereal::make_nvp("name", name));
    ar(cereal::make_nvp("size", args_.size));
    ar(cereal::make_nvp("activeBits", args_.activeBits));
    ar(cereal::make_nvp("sparsity", args_.sparsity));
    ar(cereal::make_nvp("radius", args_.radius));
    ar(cereal::make_nvp("resolution", args_.resolution));
    ar(cereal::make_nvp("category", args_.category));
    ar(cereal::make_nvp("seed", args_.seed));
  }
  
  // FOR Cereal Deserialization
  template<class Archive>
  void load_ar(Archive& ar) {
    std::vector<UInt> dim;
    std::string name;
    ar(cereal::make_nvp("name", name));
    NTA_CHECK(name == "RandomDistributedScalarEncoder");
    ar(cereal::make_nvp("size", args_.size));
    ar(cereal::make_nvp("activeBits", args_.activeBits));
    ar(cereal::make_nvp("sparsity", args_.sparsity));
    ar(cereal::make_nvp("radius", args_.radius));
    ar(cereal::make_nvp("resolution", args_.resolution));
    ar(cereal::make_nvp("category", args_.category));
    ar(cereal::make_nvp("seed", args_.seed));
    BaseEncoder<Real64>::initialize({ parameters.size });
  }
private:
  RDSE_Parameters args_;
};

typedef RandomDistributedScalarEncoder RDSE;

std::ostream & operator<<(std::ostream & out, const RandomDistributedScalarEncoder & self);

}      // End namespace htm
#endif // End ifdef NTA_ENCODERS_RDSE
