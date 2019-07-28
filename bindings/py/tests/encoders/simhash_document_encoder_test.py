# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc. https://numenta.com
#               2019, Brev Patterson, Lux Rota LLC, https://luxrota.com
#               2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero Public License version 3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for
# more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

"""Unit tests for Scalar Encoder."""

import pickle
import pytest
import unittest

from htm.bindings.encoders import SimHashDocumentEncoder, SimHashDocumentEncoderParameters
from htm.bindings.sdr import SDR, Metrics


# Shared Test Strings
testDoc1 = [ "abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"  ]
testDoc2 = [ "klmno", "pqrst",  "uvwxy",  "z1234",  "56789"  ]
testDoc3 = [ "z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"  ]
testDoc4 = [ "z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP" ]
testDocUni1 = [
  "\u0395\u0396\u0397\u0398\u0399",
  "\u0400\u0401\u0402\u0403\u0404",
  "\u0405\u0406\u0407\u0408\u0409"
]
testDocUni2 = [
  "\u0395\u0396\u0397\u0398\u0399\u0410",
  "\u0400\u0401\u0402\u0403\u0404\u0410",
  "\u0405\u0406\u0407\u0408\u0409\u0410"
]
testDocMap1 = { "aaa": 4, "bbb": 2, "ccc": 2, "ddd": 4, "sss": 1 }
testDocMap2 = { "eee": 2, "bbb": 2, "ccc": 2, "fff": 2, "sss": 1 }
testDocMap3 = { "aaa": 4, "eee": 2, "fff": 2, "ddd": 4  }


#
# TESTS
#

class SimHashDocumentEncoder_Test(unittest.TestCase):

  # Test a basic construction with defaults
  def testConstructor(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.activeBits = 20

    encoder = SimHashDocumentEncoder(params)

    assert(encoder.dimensions == [params.size])
    assert(encoder.size == params.size)
    assert(encoder.parameters.size == params.size)
    assert(encoder.parameters.activeBits == params.activeBits)
    assert(not encoder.parameters.tokenSimilarity)

  # Test a basic construction using 'sparsity' param instead of 'activeBits'
  def testConstructorParamSparsity(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.05

    encoder = SimHashDocumentEncoder(params)

    assert(encoder.dimensions == [params.size])
    assert(encoder.size == params.size)
    assert(encoder.parameters.size == params.size)
    assert(encoder.parameters.activeBits == 20)
    assert(not encoder.parameters.tokenSimilarity)

  # Test a basic encoding, try a few failure cases
  def testEncoding(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.activeBits = 20

    encoder = SimHashDocumentEncoder(params)
    output = SDR(params.size)
    encoder.encode(testDoc1, output)

    assert(encoder.size == params.size)
    assert(output.size == params.size)
    assert(output.getSum() == params.activeBits)
    with self.assertRaises(RuntimeError):
      encoder.encode([], output)
    with self.assertRaises(RuntimeError):
      encoder.encode({}, output)

  # Test encoding simple corpus with 'tokenSimilarity' On. Tokens of similar
  # spelling will affect the output in shared manner.
  def testTokenSimilarityOn(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33
    params.tokenSimilarity = True

    encoder1 = SimHashDocumentEncoder(params)
    encoder2 = SimHashDocumentEncoder(params)
    encoder3 = SimHashDocumentEncoder(params)
    encoder4 = SimHashDocumentEncoder(params)
    output1 = SDR(params.size)
    output2 = SDR(params.size)
    output3 = SDR(params.size)
    output4 = SDR(params.size)
    encoder1.encode(testDoc1, output1)
    encoder2.encode(testDoc2, output2)
    encoder3.encode(testDoc3, output3)
    encoder4.encode(testDoc4, output4)

    assert(encoder1.parameters.tokenSimilarity)

    assert(output3.getOverlap(output4) > output2.getOverlap(output3))
    assert(output2.getOverlap(output3) > output1.getOverlap(output3))
    assert(output1.getOverlap(output3) > output1.getOverlap(output4))

  # Test encoding a simple corpus with 'tokenSimilarity' Off (default). Tokens
  # of similar spelling will NOT affect the output in shared manner, but apart.
  def testTokenSimilarityOff(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33
    params.tokenSimilarity = False

    encoder1 = SimHashDocumentEncoder(params)
    encoder2 = SimHashDocumentEncoder(params)
    encoder3 = SimHashDocumentEncoder(params)
    encoder4 = SimHashDocumentEncoder(params)
    output1 = SDR(params.size)
    output2 = SDR(params.size)
    output3 = SDR(params.size)
    output4 = SDR(params.size)
    encoder1.encode(testDoc1, output1)
    encoder2.encode(testDoc2, output2)
    encoder3.encode(testDoc3, output3)
    encoder4.encode(testDoc4, output4)

    assert(output1.getOverlap(output2) > output2.getOverlap(output3))
    assert(output2.getOverlap(output3) > output3.getOverlap(output4))
    assert(output3.getOverlap(output4) > output1.getOverlap(output3))

  # Test encoding with weighted tokens. Make sure output changes accordingly.
  def testTokenWeightMap(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33

    encoder1 = SimHashDocumentEncoder(params)
    encoder2 = SimHashDocumentEncoder(params)
    encoder3 = SimHashDocumentEncoder(params)
    output1 = SDR(params.size)
    output2 = SDR(params.size)
    output3 = SDR(params.size)
    encoder1.encode(testDocMap1, output1)
    encoder2.encode(testDocMap2, output2)
    encoder3.encode(testDocMap3, output3)

    assert(output1.getOverlap(output3) > output1.getOverlap(output2))
    assert(output1.getOverlap(output2) > output2.getOverlap(output3))

  # Test encoding unicode text with 'tokenSimilarity' on
  def testUnicodeSimilarityOn(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33
    params.tokenSimilarity = True

    encoder1 = SimHashDocumentEncoder(params)
    encoder2 = SimHashDocumentEncoder(params)
    output1 = SDR(params.size)
    output2 = SDR(params.size)
    encoder1.encode(testDocUni1, output1)
    encoder2.encode(testDocUni2, output2)

    assert(output1.getOverlap(output2) > 65)

  # Test encoding unicode text with 'tokenSimilarity' Off
  def testUnicodeSimilarityOff(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33
    params.tokenSimilarity = False

    encoder1 = SimHashDocumentEncoder(params)
    encoder2 = SimHashDocumentEncoder(params)
    output1 = SDR(params.size)
    output2 = SDR(params.size)
    encoder1.encode(testDocUni1, output1)
    encoder2.encode(testDocUni2, output2)

    assert(output1.getOverlap(output2) < 65)

  # Test serialization and deserialization
  @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/htm.core/issues/160")
  def testPickle(self):
    assert(False)  # @TODO: Serialization Unimplemented
