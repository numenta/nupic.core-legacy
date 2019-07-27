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
 * FITNESS FOR A PARTICULAR PURPOSencoder. See the GNU Affero Public License for
 * more details.
 *
 * You should have received a copy of the GNU Affero Public License along with
 * this program.  If not, see http://www.gnu.org/licenses.
 * -------------------------------------------------------------------------- */

/** @file
 * SimHashDocumentEncoderTest.cpp
 * @author Brev Patterson, Lux Rota LLC, https://luxrota.com
 * @author David McDougall
 * @since 0.2.3
 */

#include "gtest/gtest.h"

#include <htm/encoders/SimHashDocumentEncoder.hpp>


namespace testing {

  using namespace htm;

  // Shared Test Strings
  const std::vector<std::string> testDoc1 =
    { "abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"  };
  const std::vector<std::string> testDoc2 =
    { "klmno", "pqrst",  "uvwxy",  "z1234",  "56789"  };
  const std::vector<std::string> testDoc3 =
    { "z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"  };
  const std::vector<std::string> testDoc4 =
    { "z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP" };
  const std::vector<std::string> testDocUni1 = {
    u8"\u0395\u0396\u0397\u0398\u0399",
    u8"\u0400\u0401\u0402\u0403\u0404",
    u8"\u0405\u0406\u0407\u0408\u0409"
  };
  const std::vector<std::string> testDocUni2 = {
    u8"\u0395\u0396\u0397\u0398\u0399\u0410",
    u8"\u0400\u0401\u0402\u0403\u0404\u0410",
    u8"\u0405\u0406\u0407\u0408\u0409\u0410"
  };


  // TESTS

  TEST(SimHashDocumentEncoder, testConstruct) {
    SimHashDocumentEncoderParameters params;
    params.size = 10u;
    params.activeBits = 2u;

    SimHashDocumentEncoder encoder(params);

    ASSERT_EQ(encoder.parameters.size, 10u);
    ASSERT_EQ(encoder.parameters.activeBits, 2u);
  }

  TEST(SimHashDocumentEncoder, testConstructParamSparsity) {
    SimHashDocumentEncoderParameters params;
    params.size = 10u;
    params.sparsity = 0.20f;

    SimHashDocumentEncoder encoder(params);

    ASSERT_EQ(encoder.parameters.size, 10u);
    ASSERT_EQ(encoder.parameters.activeBits, 2u);
  }

  TEST(SimHashDocumentEncoder, testEncoding) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.activeBits = 21u;

    SDR output({ params.size });
    SimHashDocumentEncoder encoder(params);

    const std::vector<std::string> value({});
    EXPECT_ANY_THROW(encoder.encode(value, output));   // empty
    EXPECT_NO_THROW(encoder.encode(testDoc1, output)); // full

    ASSERT_EQ(output.size, params.size);
    ASSERT_EQ(output.getSum(), params.activeBits);
  }

  TEST(SimHashDocumentEncoder, testSerialize) {
    SimHashDocumentEncoderParameters params;
    params.size = 1000u;
    params.sparsity = 0.25f;

    SimHashDocumentEncoder encoderA(params);
    std::stringstream buffer;
    encoderA.save(buffer);
    SDR outputA({ encoderA.parameters.size });
    encoderA.encode(testDoc1, outputA);

    SimHashDocumentEncoder encoderB;
    encoderB.load(buffer);
    SDR outputB({ encoderB.parameters.size });
    encoderB.encode(testDoc1, outputB);

    ASSERT_EQ(outputA, outputB);
  }

  TEST(SimHashDocumentEncoder, testTokenSimilarityOn) {
    SimHashDocumentEncoderParameters params;
    params.size = 100u;
    params.sparsity = 0.33f;
    params.tokenSimilarity = true;

    SDR output1({ params.size });
    SDR output2({ params.size });
    SDR output3({ params.size });
    SDR output4({ params.size });
    SimHashDocumentEncoder encoder1(params);
    SimHashDocumentEncoder encoder2(params);
    SimHashDocumentEncoder encoder3(params);
    SimHashDocumentEncoder encoder4(params);
    encoder1.encode(testDoc1, output1);
    encoder2.encode(testDoc2, output2);
    encoder3.encode(testDoc3, output3);
    encoder4.encode(testDoc4, output4);

    ASSERT_GT(output3.getOverlap(output4), output2.getOverlap(output3));
    ASSERT_GT(output2.getOverlap(output3), output1.getOverlap(output3));
    ASSERT_GT(output1.getOverlap(output3), output1.getOverlap(output4));
  }

  TEST(SimHashDocumentEncoder, testTokenSimilarityOff) {
    SimHashDocumentEncoderParameters params;
    params.size = 100u;
    params.sparsity = 0.33f;
    params.tokenSimilarity = false;

    SDR output1({ params.size });
    SDR output2({ params.size });
    SDR output3({ params.size });
    SDR output4({ params.size });
    SimHashDocumentEncoder encoder1(params);
    SimHashDocumentEncoder encoder2(params);
    SimHashDocumentEncoder encoder3(params);
    SimHashDocumentEncoder encoder4(params);
    encoder1.encode(testDoc1, output1);
    encoder2.encode(testDoc2, output2);
    encoder3.encode(testDoc3, output3);
    encoder4.encode(testDoc4, output4);

    ASSERT_GT(output1.getOverlap(output2), output2.getOverlap(output3));
    ASSERT_GT(output2.getOverlap(output3), output3.getOverlap(output4));
    ASSERT_GT(output3.getOverlap(output4), output1.getOverlap(output3));
  }

  TEST(SimHashDocumentEncoder, testTokenWeightMap) {
    const std::map<std::string, UInt> testDocMap1 =
      {{ "aaa", 4 }, { "bbb", 2 }, { "ccc", 2 }, { "ddd", 4 }, { "sss", 1 }};
    const std::map<std::string, UInt> testDocMap2 =
      {{ "eee", 2 }, { "bbb", 2 }, { "ccc", 2 }, { "fff", 2 }, { "sss", 1 }};
    const std::map<std::string, UInt> testDocMap3 =
      {{ "aaa", 4 }, { "eee", 2 }, { "fff", 2 }, { "ddd", 4 }};

    SimHashDocumentEncoderParameters params;
    params.size = 100u;
    params.sparsity = 0.33f;
    params.tokenSimilarity = false;

    SimHashDocumentEncoder encoderA(params);
    SDR outputA({ encoderA.parameters.size });
    encoderA.encode(testDocMap1, outputA);

    SimHashDocumentEncoder encoderB(params);
    SDR outputB({ encoderB.parameters.size });
    encoderB.encode(testDocMap2, outputB);

    SimHashDocumentEncoder encoderC(params);
    SDR outputC({ encoderC.parameters.size });
    encoderC.encode(testDocMap3, outputC);

    ASSERT_GT(outputA.getOverlap(outputC), outputA.getOverlap(outputB));
    ASSERT_GT(outputA.getOverlap(outputB), outputB.getOverlap(outputC));
  }

  TEST(SimHashDocumentEncoder, testUnicodeSimilarityOn) {
    SimHashDocumentEncoderParameters params;
    params.size = 100u;
    params.sparsity = 0.33f;
    params.tokenSimilarity = true;

    SimHashDocumentEncoder encoderA(params);
    SDR outputA({ encoderA.parameters.size });
    encoderA.encode(testDocUni1, outputA);

    SimHashDocumentEncoder encoderB(params);
    SDR outputB({ encoderB.parameters.size });
    encoderB.encode(testDocUni2, outputB);

    ASSERT_GT(outputA.getOverlap(outputB), 25u);
  }

  TEST(SimHashDocumentEncoder, testUnicodeSimilarityOff) {
    SimHashDocumentEncoderParameters params;
    params.size = 100u;
    params.sparsity = 0.33f;
    params.tokenSimilarity = false;

    SimHashDocumentEncoder encoderA(params);
    SDR outputA({ encoderA.parameters.size });
    encoderA.encode(testDocUni1, outputA);

    SimHashDocumentEncoder encoderB(params);
    SDR outputB({ encoderB.parameters.size });
    encoderB.encode(testDocUni2, outputB);

    ASSERT_LT(outputA.getOverlap(outputB), 25u);
  }

} // end namespace testing
