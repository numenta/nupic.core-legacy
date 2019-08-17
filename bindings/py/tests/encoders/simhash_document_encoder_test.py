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

"""Unit tests for SimHashDocumentEncoder."""

import pickle
import pytest
import random
import sys
import unittest

from htm.bindings.encoders import SimHashDocumentEncoder, \
    SimHashDocumentEncoderParameters
from htm.bindings.sdr import SDR, Metrics


# Shared Test Strings
testDoc1 = ["abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"]
testDoc2 = ["klmno", "pqrst",  "uvwxy",  "z1234",  "56789"]
testDoc3 = ["z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"]
testDoc4 = ["z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP"]


# TESTS

class SimHashDocumentEncoder_Test(unittest.TestCase):

    # Test a basic use-case in human-readable form.
    #  Documents:
    #    1: "The sky is beautiful today"
    #    2: "The sun is beautiful today"  (similar above, differ below)
    #    3: "Who did my homework  today"
    #  Test Expectations:
    #    1 vs 2 = very similar and should receive similar encodings
    #    2 vs 3 = very different and should receive differing encodings
    def testBasicExampleUseCase(self):
        testDocEasy1 = "The sky is beautiful today"
        testDocEasy2 = "The sun is beautiful today"  # similar up, differ down
        testDocEasy3 = "Who did my homework  today"

        # setup params
        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33

        # init encoder
        encoder = SimHashDocumentEncoder(params)

        # encode!
        output1 = encoder.encode(testDocEasy1)
        output2 = encoder.encode(testDocEasy2)
        output3 = encoder.encode(testDocEasy3)

        # encodings for Docs 1 and 2 should be more similar than the encodings
        #   for Docs 2 and 3 (which should be more disparate).
        assert(output1.getOverlap(output2) > output2.getOverlap(output3))

    # Test a basic construction with defaults
    def testConstructor(self):
        params1 = SimHashDocumentEncoderParameters()
        params1.size = 400
        params1.activeBits = 20
        encoder1 = SimHashDocumentEncoder(params1)
        assert(encoder1)
        assert(encoder1.dimensions == [params1.size])
        assert(encoder1.size == params1.size)
        assert(encoder1.parameters.size == params1.size)
        assert(encoder1.parameters.activeBits == params1.activeBits)
        assert(not encoder1.parameters.tokenSimilarity)

        # test bad encoder params - both activeBits and sparsity
        params2 = SimHashDocumentEncoderParameters()
        params2.size = 400
        params2.activeBits = 20
        params2.sparsity = 0.666
        encoder2 = None
        assert(not encoder2)
        with self.assertRaises(RuntimeError):
            encoder2 = SimHashDocumentEncoder(params2)

        # test bad encoder params - neither activeBits or sparsity
        params3 = SimHashDocumentEncoderParameters()
        params3.size = 400
        encoder3 = None
        assert(not encoder3)
        with self.assertRaises(RuntimeError):
            encoder3 = SimHashDocumentEncoder(params3)

        # test good encoder param - using 'sparsity' instead of 'activeBits'
        params4 = SimHashDocumentEncoderParameters()
        params4.size = 400
        params4.sparsity = 0.05
        encoder4 = SimHashDocumentEncoder(params4)
        assert(encoder4)
        assert(encoder4.dimensions == [params4.size])
        assert(encoder4.size == params4.size)
        assert(encoder4.parameters.size == params4.size)
        assert(encoder4.parameters.activeBits == 20)
        assert(not encoder4.parameters.tokenSimilarity)

    # Make sure bits stay the same across uses and envrionments
    def testDeterminism(self):
        GOLD = SDR(1000)
        GOLD.sparse = [
            16, 29, 60, 76, 117, 144, 149, 198, 221, 242, 291, 315, 340, 347,
            368, 373, 384, 400, 402, 412, 431, 433, 447, 459, 471, 475, 488,
            552, 587, 601, 613, 624, 639, 668, 721, 724, 726, 753, 799, 805,
            807, 821, 822, 843, 879, 912, 929, 943, 956, 970]
        params = SimHashDocumentEncoderParameters()
        params.size = GOLD.size
        params.sparsity = 0.05
        encoder = SimHashDocumentEncoder(params)
        current = encoder.encode("I came to the fork in the road")
        assert(current == GOLD)

    # Test a basic encoding, try a few failure cases
    def testEncoding(self):
        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.activeBits = 20

        encoder = SimHashDocumentEncoder(params)
        output = encoder.encode(testDoc1)

        assert(encoder.size == params.size)
        assert(output.size == params.size)
        assert(output.getSum() == params.activeBits)
        with self.assertRaises(RuntimeError):
            encoder.encode([], output)
        with self.assertRaises(RuntimeError):
            encoder.encode("", output)

    # Test excludes param
    def testExcludes(self):
        keepList = ["but", "it", "all", "stays", "the", "same"]
        nopeList = ["seasons", "change", "mad", "things", "rearrange"]
        fullList = keepList + nopeList

        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33

        encoder1 = SimHashDocumentEncoder(params)
        output1 = encoder1.encode(fullList)

        encoder2 = SimHashDocumentEncoder(params)
        output2 = encoder2.encode(keepList)

        params.excludes = nopeList
        encoder3 = SimHashDocumentEncoder(params)
        output3 = encoder3.encode(fullList)

        assert(output1 != output2)  # full != part
        assert(output1 != output3)  # full != (full - nope)
        assert(output2 == output3)  # part == (full - nope)

    # Test de/serialization via Pickle method
    @pytest.mark.skip("pickle deserialization getting corrupted somehow @TODO")
    @pytest.mark.skipif(
        sys.version_info < (3, 6),
        reason="Fails for python2 with segmentation fault")
    def testSerializePickle(self):
        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.activeBits = 21
        enc1 = SimHashDocumentEncoder(params)

        pickled = pickle.dumps(enc1)
        enc2 = pickle.loads(pickled)

        assert(enc1.size == enc2.size)
        assert(enc1.parameters.size == enc2.parameters.size)
        assert(enc1.parameters.activeBits == enc2.parameters.activeBits)

    # Test de/serialization via String
    def testSerializeString(self):
        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.activeBits = 21
        enc1 = SimHashDocumentEncoder(params)

        params.size = 40
        params.activeBits = 2
        enc2 = SimHashDocumentEncoder(params)

        assert(enc1.size != enc2.size)
        assert(enc1.parameters.size != enc2.parameters.size)
        assert(enc1.parameters.activeBits != enc2.parameters.activeBits)

        s = enc1.writeToString()
        enc2.loadFromString(s)

        assert(enc1.size == enc2.size)
        assert(enc1.parameters.size == enc2.parameters.size)
        assert(enc1.parameters.activeBits == enc2.parameters.activeBits)

    # Test binary properties of a variety of output encodings
    def testStatistics(self):
        # 100 random simple English words run mass encoding stats against
        testCorpus = [
            "find", "any", "new", "work", "part", "take", "get", "place",
            "made", "live", "where", "after", "back", "little", "only",
            "round", "man", "year", "came", "show", "every", "good", "me",
            "give", "our", "under", "name", "very", "through", "just", "form",
            "sentence", "great", "think", "say", "help", "low", "line",
            "differ", "turn", "cause", "much", "mean", "before", "move",
            "right", "boy", "old", "too", "same", "tell", "does", "set",
            "three", "want", "air", "well", "also", "play", "small", "end",
            "put", "home", "read", "hand", "port", "large", "spell", "add",
            "even", "land", "here", "must", "big", "high", "such", "follow",
            "act", "why", "ask", "men", "change", "went", "light", "kind",
            "off", "need", "house", "picture", "try", "us", "again", "animal",
            "point", "mother", "world", "near", "build", "self", "earth"]
        num_samples = 1000  # number of documents to run
        num_tokens = 10     # tokens per document

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
                token = testCorpus[random.randint(0, len(testCorpus) - 1)]
                document.append(token)
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
        assert(report2.overlap.min() > 0.38)
        assert(report2.overlap.max() > 0.79)
        assert(report2.overlap.mean() > 0.61)
        assert(report2.sparsity.min() > params2.sparsity - 0.01)
        assert(report2.sparsity.max() < params2.sparsity + 0.01)
        assert(report2.sparsity.mean() > params2.sparsity - 0.005)
        assert(report2.sparsity.mean() < params2.sparsity + 0.005)

    # Test encoding with case in/sensitivity
    def testTokenCaseSensitivity(self):
        # Case-sensitivite strings
        testDocCase1 = [
            "alpha", "bravo",  "delta",  "echo",  "foxtrot", "hotel"]
        testDocCase2 = [
            "ALPHA", "BRAVO",  "DELTA",  "ECHO",  "FOXTROT", "HOTEL"]
        part = ["eCHo", "foXTROt", "hOtEl"]
        discard = ["AlPHa", "BRaVo", "dELTa"]
        vocab = {"EcHo": 1, "FOxtRoT": 1, "HoTeL": 1}

        # caseSensitivity ON
        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33
        params.caseSensitivity = True
        encoder1 = SimHashDocumentEncoder(params)
        output1 = encoder1.encode(testDocCase1)
        output2 = encoder1.encode(testDocCase2)
        assert(output1 != output2)

        # caseSensitivity OFF
        params.caseSensitivity = False
        encoder2 = SimHashDocumentEncoder(params)
        output1 = encoder2.encode(testDocCase1)
        output2 = encoder2.encode(testDocCase2)
        assert(output1 == output2)

        # caseSensitivity=OFF +excludes
        params.excludes = discard
        encoder3 = SimHashDocumentEncoder(params)
        output3a = encoder3.encode(testDocCase1)
        output3b = encoder3.encode(part)
        assert(output3a == output3b)

        # caseSensitivity=OFF +vocabulary
        params4 = SimHashDocumentEncoderParameters()
        params4.size = 400
        params4.sparsity = 0.33
        params4.caseSensitivity = False
        params4.encodeOrphans = False
        params4.vocabulary = vocab
        encoder4 = SimHashDocumentEncoder(params4)
        output4a = encoder4.encode(testDocCase1)
        output4b = encoder4.encode(part)
        assert(output4a == output4b)

    # Test encoding simple corpus with 'tokenSimilarity' On/Off. If ON, Tokens
    # of similar spelling will affect the output in shared manner. If OFF, then
    # Tokens of similar spelling will NOT affect the output in shared manner,
    # but apart (Default).
    def testTokenSimilarity(self):
        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33
        params.caseSensitivity = True

        # tokenSimilarity ON
        params.tokenSimilarity = True
        encoder1 = SimHashDocumentEncoder(params)
        output1 = SDR(params.size)
        output2 = SDR(params.size)
        output3 = SDR(params.size)
        output4 = SDR(params.size)
        encoder1.encode(testDoc1, output1)
        encoder1.encode(testDoc2, output2)
        encoder1.encode(testDoc3, output3)
        encoder1.encode(testDoc4, output4)
        assert(output3.getOverlap(output4) > output2.getOverlap(output3))
        assert(output2.getOverlap(output3) > output1.getOverlap(output3))
        assert(output1.getOverlap(output3) > output1.getOverlap(output4))

        # tokenSimilarity OFF
        params.tokenSimilarity = False
        encoder2 = SimHashDocumentEncoder(params)
        output1.zero()
        output2.zero()
        output3.zero()
        output4.zero()
        encoder2.encode(testDoc1, output1)
        encoder2.encode(testDoc2, output2)
        encoder2.encode(testDoc3, output3)
        encoder2.encode(testDoc4, output4)
        assert(output1.getOverlap(output2) > output2.getOverlap(output3))
        assert(output2.getOverlap(output3) > output3.getOverlap(output4))
        assert(output3.getOverlap(output4) > output1.getOverlap(output3))

    # test vocabulary
    def testTokenVocabulary(self):
        vocabulary = {
          "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
          "g": 1, "h": 2, "i": 3, "j": 4, "k": 5, "l": 6}
        input1 = "a b c d e f"
        input2 = "a b c d e f t u w x y z"

        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33
        params.vocabulary = vocabulary

        # vocabulary +encodeOrphans
        params.encodeOrphans = True
        encoder1 = SimHashDocumentEncoder(params)
        output1a = encoder1.encode(input1)
        output1b = encoder1.encode(input2)
        assert(output1a != output1b)

        # vocabulary -encodeOrphans
        params.encodeOrphans = False
        encoder2 = SimHashDocumentEncoder(params)
        output2a = encoder2.encode(input1)
        output2b = encoder2.encode(input2)
        assert(output2a == output2b)

    # Test encoding with weighted tokens. Make sure output changes accordingly.
    def testTokenWeightMap(self):
        weights = {
          "aaa": 4, "bbb": 2, "ccc": 2, "ddd": 4, "eee": 2, "fff": 2, "sss": 1}
        doc1 = ["aaa", "bbb", "ccc", "ddd", "sss"]
        doc2 = ["eee", "bbb", "ccc", "fff", "sss"]
        doc3 = ["aaa", "eee", "fff", "ddd"]

        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33
        params.tokenSimilarity = False
        params.vocabulary = weights
        encoder = SimHashDocumentEncoder(params)

        output1 = encoder.encode(doc1)
        output2 = encoder.encode(doc2)
        output3 = encoder.encode(doc3)

        assert(output1.getOverlap(output3) > output1.getOverlap(output2))
        assert(output1.getOverlap(output2) > output2.getOverlap(output3))

    # Test encoding unicode text, including with 'tokenSimilarity' On/Off
    def testUnicode(self):
        testDocUni1 = [
          "\u0395\u0396\u0397\u0398\u0399",
          "\u0400\u0401\u0402\u0403\u0404",
          "\u0405\u0406\u0407\u0408\u0409"]
        testDocUni2 = [
          "\u0395\u0396\u0397\u0398\u0399\u0410",
          "\u0400\u0401\u0402\u0403\u0404\u0410",
          "\u0405\u0406\u0407\u0408\u0409\u0410"]

        params = SimHashDocumentEncoderParameters()
        params.size = 400
        params.sparsity = 0.33

        # unicode 'tokenSimilarity' ON
        params.tokenSimilarity = True
        encoder1 = SimHashDocumentEncoder(params)
        output1 = SDR(params.size)
        output2 = SDR(params.size)
        encoder1.encode(testDocUni1, output1)
        encoder1.encode(testDocUni2, output2)
        assert(output1.getOverlap(output2) > 65)

        # unicode 'tokenSimilarity' OFF
        params.tokenSimilarity = False
        encoder2 = SimHashDocumentEncoder(params)
        output1.zero()
        output2.zero()
        encoder2.encode(testDocUni1, output1)
        encoder2.encode(testDocUni2, output2)
        assert(output1.getOverlap(output2) < 65)
