# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc. https://numenta.com
#               2019, David McDougall
#               2019, Brev Patterson, Lux Rota LLC, https://luxrota.com
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
import random
import sys
import unittest

from htm.bindings.encoders import SimHashDocumentEncoder, SimHashDocumentEncoderParameters
from htm.bindings.sdr import SDR, Metrics


### Shared Test Strings

# Human-readable example use case strings (see `testBasicExampleUseCase` below):
#   1 vs 2 = very similar and should receive similar encodings
#   2 vs 3 = very different and should receive differeing encodings
exampleDoc1 = [ "The", "sky", "is", "beautiful", "today" ]
exampleDoc2 = [ "The", "sun", "is", "beautiful", "today" ] # similar above, differ below
exampleDoc3 = [ "Who", "did", "my", "homework",  "today" ]
# Basic test strings
testDoc1 = [ "abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"  ]
testDoc2 = [ "klmno", "pqrst",  "uvwxy",  "z1234",  "56789"  ]
testDoc3 = [ "z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"  ]
testDoc4 = [ "z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP" ]
# Case-sensitivite strings
testDocCase1 = [ "alpha", "bravo",  "delta",  "echo",  "foxtrot", "hotel" ]
testDocCase2 = [ "ALPHA", "BRAVO",  "DELTA",  "ECHO",  "FOXTROT", "HOTEL" ]
# Weighted strings
testDocMap1 = { "aaa": 4, "bbb": 2, "ccc": 2, "ddd": 4, "sss": 1 }
testDocMap2 = { "eee": 2, "bbb": 2, "ccc": 2, "fff": 2, "sss": 1 }
testDocMap3 = { "aaa": 4, "eee": 2, "fff": 2, "ddd": 4  }
# Unicode test strings
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
# 100 random simple English words run mass encoding stats against
testCorpus = [ "find", "any", "new", "work", "part", "take", "get", "place",
  "made", "live", "where", "after", "back", "little", "only", "round", "man",
  "year", "came", "show", "every", "good", "me", "give", "our", "under",
  "name", "very", "through", "just", "form", "sentence", "great", "think",
  "say", "help", "low", "line", "differ", "turn", "cause", "much", "mean",
  "before", "move", "right", "boy", "old", "too", "same", "tell", "does",
  "set", "three", "want", "air", "well", "also", "play", "small", "end", "put",
  "home", "read", "hand", "port", "large", "spell", "add", "even", "land",
  "here", "must", "big", "high", "such", "follow", "act", "why", "ask", "men",
  "change", "went", "light", "kind", "off", "need", "house", "picture", "try",
  "us", "again", "animal", "point", "mother", "world", "near", "build",
  "self", "earth" ]


### TESTS

class SimHashDocumentEncoder_Test(unittest.TestCase):

  # Test a basic use-case in human-readable form.
  #  Documents (from shared test strings above):
  #    1: "The sky is beautiful today"
  #    2: "The sun is beautiful today"  (similar above, differ below)
  #    3: "Who did my homework  today"
  #  Test Expectations:
  #    1 vs 2 = very similar and should receive similar encodings
  #    2 vs 3 = very different and should receive differing encodings
  def testBasicExampleUseCase(self):
    # setup params
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33

    # init encoder
    encoder = SimHashDocumentEncoder(params)

    # encode!
    output1 = encoder.encode(exampleDoc1)
    output2 = encoder.encode(exampleDoc2)
    output3 = encoder.encode(exampleDoc3)

    # encodings for Docs 1 and 2 should be more similar than the encodings
    #   for Docs 2 and 3 (which should be more disparate).
    assert(output1.getOverlap(output2) > output2.getOverlap(output3))

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

  # Test binary properties of a variety of output encodings
  def testStatistics(self):
    num_samples = 1000 # number of documents to run
    num_tokens = 10 # tokens per document

    # Case 1 = tokenSimilarity OFF
    params1 = SimHashDocumentEncoderParameters()
    params1.size = 400
    params1.sparsity = 0.33
    params1.tokenSimilarity = False
    encoder1 = SimHashDocumentEncoder(params1)

    # Case 2 = tokenSimilarity ON
    params2 = params1
    params2.tokenSimilarity = True
    encoder2 = SimHashDocumentEncoder(params2)

    sdrs1 = []
    sdrs2 = []
    for _ in range(num_samples):
      document = []
      for _ in range(num_tokens - 1):
        document.append(testCorpus[random.randint(0, len(testCorpus) - 1)])
      sdrs1.append(encoder1.encode(document))
      sdrs2.append(encoder2.encode(document))

    report1 = Metrics([encoder1.size], len(sdrs1) + 1)
    report2 = Metrics([encoder2.size], len(sdrs2) + 1)

    for sdr in sdrs1:
      report1.addData(sdr)
    for sdr in sdrs2:
      report2.addData(sdr)

    # Assertions for Case 1 = tokenSimilarity OFF
    assert(report1.activationFrequency.entropy() > 0.87)
    assert(report1.activationFrequency.min() > 0.01)
    assert(report1.activationFrequency.max() < 0.99)
    assert(report1.activationFrequency.mean() > params1.sparsity - 0.005)
    assert(report1.activationFrequency.mean() < params1.sparsity + 0.005)
    assert(report1.overlap.min() > 0.21)
    assert(report1.overlap.max() > 0.53)
    assert(report1.overlap.mean() > 0.38)
    assert(report1.sparsity.min() > params1.sparsity - 0.01)
    assert(report1.sparsity.max() < params1.sparsity + 0.01)
    assert(report1.sparsity.mean() > params1.sparsity - 0.005)
    assert(report1.sparsity.mean() < params1.sparsity + 0.005)

    # Assertions for Case 2 = tokenSimilarity ON
    assert(report2.activationFrequency.entropy() > 0.59)
    assert(report2.activationFrequency.min() >= 0)
    assert(report2.activationFrequency.max() <= 1)
    assert(report2.activationFrequency.mean() > params2.sparsity - 0.005)
    assert(report2.activationFrequency.mean() < params2.sparsity + 0.005)
    assert(report2.overlap.min() > 0.40)
    assert(report2.overlap.max() > 0.79)
    assert(report2.overlap.mean() > 0.61)
    assert(report2.sparsity.min() > params2.sparsity - 0.01)
    assert(report2.sparsity.max() < params2.sparsity + 0.01)
    assert(report2.sparsity.mean() > params2.sparsity - 0.005)
    assert(report2.sparsity.mean() < params2.sparsity + 0.005)

  # Test encoding without case sensitivity
  def testTokenCaseInsensitivity(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33

    encoder1 = SimHashDocumentEncoder(params)
    output1 = encoder1.encode(testDocCase1)

    encoder2 = SimHashDocumentEncoder(params)
    output2 = encoder2.encode(testDocCase2)

    assert(output1 == output2)

  # Test encoding with case sensitivity
  def testTokenCaseSensitivity(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33
    params.caseSensitivity = True

    encoder1 = SimHashDocumentEncoder(params)
    output1 = encoder1.encode(testDocCase1)

    encoder2 = SimHashDocumentEncoder(params)
    output2 = encoder2.encode(testDocCase2)

    assert(output1 != output2)

  # Test encoding simple corpus with 'tokenSimilarity' On. Tokens of similar
  # spelling will affect the output in shared manner.
  def testTokenSimilarityOn(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.sparsity = 0.33
    params.caseSensitivity = True
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
    params.caseSensitivity = True
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

  # Test de/serialization via Pickle method
  @pytest.mark.skip("pickle deserialization getting corrupted @TODO")
  @pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails for python2 with segmentation fault")
  def testSerializePickle(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.activeBits = 21
    encoder1 = SimHashDocumentEncoder(params)

    pickled = pickle.dumps(encoder1)
    encoder2 = pickle.loads(pickled)

    assert(encoder1.size == encoder2.size)
    assert(encoder1.parameters.size == encoder2.parameters.size)
    assert(encoder1.parameters.activeBits == encoder2.parameters.activeBits)

  # Test de/serialization via String
  def testSerializeString(self):
    params = SimHashDocumentEncoderParameters()
    params.size = 400
    params.activeBits = 21
    encoder1 = SimHashDocumentEncoder(params)

    params.size = 40
    params.activeBits = 2
    encoder2 = SimHashDocumentEncoder(params)

    assert(encoder1.size != encoder2.size)
    assert(encoder1.parameters.size != encoder2.parameters.size)
    assert(encoder1.parameters.activeBits != encoder2.parameters.activeBits)

    s = encoder1.writeToString()
    encoder2.loadFromString(s)

    assert(encoder1.size == encoder2.size)
    assert(encoder1.parameters.size == encoder2.parameters.size)
    assert(encoder1.parameters.activeBits == encoder2.parameters.activeBits)

