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
 * py_SimHashDocumentEncoder.cpp
 */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <htm/encoders/SimHashDocumentEncoder.hpp>

namespace py = pybind11;

using namespace htm;
using namespace std;


namespace htm_ext {

  using namespace htm;

  void init_SimHashDocumentEncoder(py::module& m)
  {
    /**
     * Parameters
     */
    py::class_<SimHashDocumentEncoderParameters>
      py_SimHashDocumentEncoderParameters(m, "SimHashDocumentEncoderParameters",
R"(
Parameters for the SimHashDocumentEncoder.
)");

    py_SimHashDocumentEncoderParameters.def(py::init<>());

    py_SimHashDocumentEncoderParameters.def_readwrite("activeBits",
      &SimHashDocumentEncoderParameters::activeBits,
R"(
This is the number of true bits in the encoded output SDR. The output encoding
will have a distribution of this many 1's. Specify only one of: activeBits
or sparsity.
)");

    py_SimHashDocumentEncoderParameters.def_readwrite("size",
      &SimHashDocumentEncoderParameters::size,
R"(
This is the total number of bits in the encoded output SDR.
)");

    py_SimHashDocumentEncoderParameters.def_readwrite("sparsity",
      &SimHashDocumentEncoderParameters::sparsity,
R"(
This is an alternate way (percentage) to specify the the number of active bits.
Specify only one of: activeBits or sparsity.
)");

    py_SimHashDocumentEncoderParameters.def_readwrite("caseSensitivity",
      &SimHashDocumentEncoderParameters::caseSensitivity,
R"(
Should capitalized English letters (A-Z) have different influence on our output
than their lower-cased (a-z) counterparts? Or the same influence on output?
  If TRUE:  "DOGS" and "dogs" will have completely different encodings.
  If FALSE: "DOGS" and "dogs" will share the same encoding (Default).
)");

    py_SimHashDocumentEncoderParameters.def_readwrite("tokenSimilarity",
      &SimHashDocumentEncoderParameters::tokenSimilarity,
R"(
This allows similar tokens ("cat", "cats") to also be represented similarly,
at the cost of document similarity accuracy. Default is FALSE (providing better
document-level similarity, at the expense of token-level similarity).

Results are heavily dependent on the content of your input data.

If TRUE: Similar tokens ("cat", "cats") will have similar influence on the
  output simhash. This benefit comes with the cost of a reduction in
  document-level similarity accuracy.

If FALSE: Similar tokens ("cat", "cats") will have individually unique and
  unrelated influence on the output simhash encoding, thus losing token-level
  similarity and increasing document-level similarity.
)");


    /**
     * Class
     */
    py::class_<SimHashDocumentEncoder> py_SimHashDocumentEncoder(m,
      "SimHashDocumentEncoder",
R"(
Encodes a document text into a distributed spray of 1's.

The SimHashDocumentEncoder encodes a document (array of strings) value into an
array of bits. The output is 0's except for a sparse distribution spray of 1's.
Similar document encodings will share similar representations, and vice versa.
Unicode is supported. No lookup tables are used.

"Similarity" here refers to bitwise similarity (small hamming distance,
high overlap), not semantic similarity (encodings for "apple" and
"computer" will have no relation here.) For document encodings which are
also semantic, please try Cortical.io and their Semantic Folding tech.

Definition of Terms:
  - A "corpus" is a collection of "documents".
  - A "document" is made up of "tokens" (or "words").
  - A "token" is made up of "characters" (or "letters").

Encoding is accomplished using SimHash, a Locality-Sensitive Hashing (LSH)
algorithm from the world of nearest-neighbor document similarity search.
As SDRs can be of any size, we use the SHA3 hashing algorithm with SHAKE256
Extendible Output Function (XOF). We deviate slightly from the standard
SimHash algorithm in order to achieve sparsity.

To inspect this run:
$ python -m htm.encoders.simhash_document_encoder --help

Python Code Example:
    from htm.bindings.encoders import SimHashDocumentEncoder
    from htm.bindings.encoders import SimHashDocumentEncoderParameters
    from htm.bindings.sdr import SDR

    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.activeBits = 21

    output = SDR(params.size)
    encoder = SimHashDocumentEncoder(params)

    # call style: output is reference
    encoder.encode({ "brevo": 3, "delta" : 1, "echo" : 2 }, output)
    encoder.encode([ "bravo", "delta", "echo" ], output)  # weights 1
    encoder.encode("bravo delta echo", output)            # weights 1

    # call style: output is returned
    other = encoder.encode({ "brevo": 3, "delta" : 1, "echo" : 2 })
    other = encoder.encode([ "bravo", "delta", "echo" ])  # weights 1
    other = encoder.encode("bravo delta echo")            # weights 1
)");

    py_SimHashDocumentEncoder.def(py::init<SimHashDocumentEncoderParameters&>());

    py_SimHashDocumentEncoder.def_property_readonly("parameters",
      [](SimHashDocumentEncoder &self) { return self.parameters; },
R"(
Contains the parameter structure which this encoder uses internally. All fields
are filled in automatically.
)");

    py_SimHashDocumentEncoder.def_property_readonly("dimensions",
      [](SimHashDocumentEncoder &self) { return self.dimensions; });

    py_SimHashDocumentEncoder.def_property_readonly("size",
      [](SimHashDocumentEncoder &self) { return self.size; });

    // Handle case of class method overload + class method override
    // https://pybind11.readthedocs.io/en/master/classes.html#overloaded-methods
    py_SimHashDocumentEncoder.def("encode",
      (void (SimHashDocumentEncoder::*)(std::map<std::string, htm::UInt>, htm::SDR &))
        &SimHashDocumentEncoder::encode);
    py_SimHashDocumentEncoder.def("encode", // alt: simple list w/o weights
      (void (SimHashDocumentEncoder::*)(std::vector<std::string>, htm::SDR &))
        &SimHashDocumentEncoder::encode);
    py_SimHashDocumentEncoder.def("encode", // alt: simple string w/o weights
      (void (SimHashDocumentEncoder::*)(std::string, htm::SDR &))
        &SimHashDocumentEncoder::encode);

    py_SimHashDocumentEncoder.def("encode",
      [](SimHashDocumentEncoder &self, std::map<std::string, htm::UInt> value) {
        auto output = new SDR({ self.size });
        self.encode( value, *output );
        return output;
      },
R"(
Takes input in a python map of strings (tokens) => integer (weights).
  Ex: { "alpha": 2, "bravo": 1, "delta": 1, "echo": 3 }.
Documents can contain any number of tokens > 0. Token order in the document is
  ignored and does not effect the output encoding.
)");
    py_SimHashDocumentEncoder.def("encode", // alt: simple list w/o weights
      [](SimHashDocumentEncoder &self, std::vector<std::string> value) {
        auto output = new SDR({ self.size });
        self.encode( value, *output );
        return output;
      },
R"(
Simple alternate calling pattern using only list of small strings, with no
weights (assumed to be 1). Takes input in a python list of strings (tokens).
  Ex: [ "alpha", "bravo", "delta", "echo" ].
Documents can contain any number of tokens > 0. Token order in the document is
  ignored and does not effect the output encoding.
)");
    py_SimHashDocumentEncoder.def("encode", // alt: simple string w/o weights
      [](SimHashDocumentEncoder &self, std::string value) {
        auto output = new SDR({ self.size });
        self.encode( value, *output );
        return output;
      },
R"(
Simple alternate calling pattern using only a single longer string, with no
weights (assumed to be 1). Takes input as a long python string, which will
automatically be tokenized (split on whitespace).
  Ex: "alpha bravo delta echo".
Documents can contain any number of tokens > 0. Token order in the document is
  ignored and does not effect the output encoding.
)");

    // Serialization
    // string in
    py_SimHashDocumentEncoder.def("loadFromString", [](SimHashDocumentEncoder& self, const py::bytes& inString) {
      std::stringstream inStream(inString.cast<std::string>());
      self.load(inStream);
    });
    // string out
    py_SimHashDocumentEncoder.def("writeToString", [](const SimHashDocumentEncoder& self) {
      std::ostringstream os;
      os.flags(ios::scientific);
      os.precision(numeric_limits<double>::digits10 + 1);
      self.save(os);
      return py::bytes( os.str() );
    });
    // pickle
    py_SimHashDocumentEncoder.def(py::pickle(
      // pickle out
      [](const SimHashDocumentEncoder& self) {
        std::stringstream ss;
        self.save(ss);
        return py::bytes( ss.str() );
      },
      // pickle in
      [](py::bytes &s) {
        std::stringstream ss( s.cast<std::string>() );
        SimHashDocumentEncoder self;
        self.load(ss);
        return self;
      }
    ));

  }
}
