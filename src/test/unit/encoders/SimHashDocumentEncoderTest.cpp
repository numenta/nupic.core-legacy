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
 * FITNESS FOR A PARTICULAR PURPOSencoder. See the GNU Affero Public License for
 * more details.
 *
 * You should have received a copy of the GNU Affero Public License along with
 * this program.  If not, see http://www.gnu.org/licenses.
 * -------------------------------------------------------------------------- */

/** @file
 * SimHashDocumentEncoderTest.cpp
 */

#include "gtest/gtest.h"

#include <htm/encoders/SimHashDocumentEncoder.hpp>


namespace testing {

  using namespace htm;

  /* Shared Test Strings */
  // Human-readable basic use-case strings (see `testBasicExampleUseCase` below)
  //  * 1 vs 2 = very similar and should receive similar encodings
  //  * 2 vs 3 = very different and should receive differeing encodings
  std::string testDocEasy1 = "The sky is beautiful today";
  std::string testDocEasy2 = "The sun is beautiful today"; // similar above, differ below
  std::string testDocEasy3 = "Who did my homework  today";
  // Basic test strings
  const std::vector<std::string> testDoc1 =
    { "abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"  };
  const std::vector<std::string> testDoc2 =
    { "klmno", "pqrst",  "uvwxy",  "z1234",  "56789"  };
  const std::vector<std::string> testDoc3 =
    { "z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"  };
  const std::vector<std::string> testDoc4 =
    { "z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP" };
  // Case-sensitivite strings
  const std::vector<std::string> testDocCase1 =
    { "alpha", "bravo",  "delta",  "echo",  "foxtrot", "hotel" };
  const std::vector<std::string> testDocCase2 =
    { "ALPHA", "BRAVO",  "DELTA",  "ECHO",  "FOXTROT", "HOTEL" };
  // Weighted strings
  const std::map<std::string, UInt> testDocMap1 =
    {{ "aaa", 4 }, { "bbb", 2 }, { "ccc", 2 }, { "ddd", 4 }, { "sss", 1 }};
  const std::map<std::string, UInt> testDocMap2 =
    {{ "eee", 2 }, { "bbb", 2 }, { "ccc", 2 }, { "fff", 2 }, { "sss", 1 }};
  const std::map<std::string, UInt> testDocMap3 =
    {{ "aaa", 4 }, { "eee", 2 }, { "fff", 2 }, { "ddd", 4 }};
  // Unicode test strings
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


  /**
   * TESTS
   */

  // Test a basic use-case in human-readable form.
  //  Documents (from shared test strings above):
  //    1: "The sky is beautiful today"
  //    2: "The sun is beautiful today"  (similar above, differ below)
  //    3: "Who did my homework  today"
  //  Test Expectations:
  //    1 vs 2 = very similar and should receive similar encodings
  //    2 vs 3 = very different and should receive differing encodings
  TEST(SimHashDocumentEncoder, testBasicExampleUseCase) {
    // setup params
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;

    // init encoder
    SimHashDocumentEncoder encoder(params);

    // init document encoding outputs
    SDR output1({ params.size });
    SDR output2({ params.size });
    SDR output3({ params.size });

    // encode!
    encoder.encode(testDocEasy1, output1);
    encoder.encode(testDocEasy2, output2);
    encoder.encode(testDocEasy3, output3);

    // encodings for Docs 1 and 2 should be more similar than the encodings
    //  for Docs 2 and 3 (which should be more disparate).
    ASSERT_GT(output1.getOverlap(output2), output2.getOverlap(output3));
  }

  // Test a basic construction with defaults
  TEST(SimHashDocumentEncoder, testConstructor) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.activeBits = 20u;

    SimHashDocumentEncoder encoder(params);

    ASSERT_EQ(encoder.parameters.size, params.size);
    ASSERT_EQ(encoder.parameters.activeBits, params.activeBits);
  }

  // Test a basic construction using 'sparsity' param instead of 'activeBits'
  TEST(SimHashDocumentEncoder, testConstructorParamSparsity) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.05f;

    SimHashDocumentEncoder encoder(params);

    ASSERT_EQ(encoder.parameters.size, params.size);
    ASSERT_EQ(encoder.parameters.activeBits, 20u);
  }

  // Test a basic encoding, try a few failure cases
  TEST(SimHashDocumentEncoder, testEncoding) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.activeBits = 20u;

    SDR output({ params.size });
    SimHashDocumentEncoder encoder(params);

    const std::vector<std::string> value({});
    EXPECT_ANY_THROW(encoder.encode(value, output));   // empty
    EXPECT_NO_THROW(encoder.encode(testDoc1, output)); // full

    ASSERT_EQ(output.size, params.size);
    ASSERT_EQ(output.getSum(), params.activeBits);
  }

  // Test Serialization and Deserialization
  TEST(SimHashDocumentEncoder, testSerialize) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;

    SimHashDocumentEncoder encoder1(params);
    std::stringstream buffer;
    encoder1.save(buffer);
    SDR output1({ encoder1.size });
    encoder1.encode(testDoc1, output1);

    SimHashDocumentEncoder encoder2;
    encoder2.load(buffer);
    SDR output2({ encoder2.size });
    encoder2.encode(testDoc1, output2);

    ASSERT_EQ(output1.size, output2.size);
    ASSERT_EQ(output1.getDense(), output2.getDense());
  }

  // Test encoding with case in/sensitivity
  TEST(SimHashDocumentEncoder, testTokenCaseSensitivity) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;

    // caseSensitivity ON
    params.caseSensitivity = true;
    SimHashDocumentEncoder encoder1(params);
    SDR output1({ params.size });
    SDR output2({ params.size });
    encoder1.encode(testDocCase1, output1);
    encoder1.encode(testDocCase2, output2);
    ASSERT_NE(output1, output2);

    // caseSensitivity OFF
    params.caseSensitivity = false;
    SimHashDocumentEncoder encoder2(params);
    output1.zero();
    output2.zero();
    encoder2.encode(testDocCase1, output1);
    encoder2.encode(testDocCase2, output2);
    ASSERT_EQ(output1, output2);
  }

  // Test encoding simple corpus with 'tokenSimilarity' On/Off. If ON, tokens of
  //  similar spelling will affect the output in shared manner. If OFF, tokens
  //  of similar spelling will NOT affect the output in shared manner,
  //  but apart (Default).
  TEST(SimHashDocumentEncoder, testTokenSimilarity) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    params.caseSensitivity = true;

    // tokenSimilarity ON
    params.tokenSimilarity = true;
    SimHashDocumentEncoder encoder1(params);
    SDR output1({ params.size });
    SDR output2({ params.size });
    SDR output3({ params.size });
    SDR output4({ params.size });
    encoder1.encode(testDoc1, output1);
    encoder1.encode(testDoc2, output2);
    encoder1.encode(testDoc3, output3);
    encoder1.encode(testDoc4, output4);
    ASSERT_GT(output3.getOverlap(output4), output2.getOverlap(output3));
    ASSERT_GT(output2.getOverlap(output3), output1.getOverlap(output3));
    ASSERT_GT(output1.getOverlap(output3), output1.getOverlap(output4));

    // tokenSimilarity OFF
    params.tokenSimilarity = false;
    SimHashDocumentEncoder encoder2(params);
    output1.zero();
    output2.zero();
    output3.zero();
    output4.zero();
    encoder2.encode(testDoc1, output1);
    encoder2.encode(testDoc2, output2);
    encoder2.encode(testDoc3, output3);
    encoder2.encode(testDoc4, output4);
    ASSERT_GT(output1.getOverlap(output2), output2.getOverlap(output3));
    ASSERT_GT(output2.getOverlap(output3), output3.getOverlap(output4));
    ASSERT_GT(output3.getOverlap(output4), output1.getOverlap(output3));
  }

  // Test encoding with weighted tokens. Make sure output changes accordingly.
  TEST(SimHashDocumentEncoder, testTokenWeightMap) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    params.tokenSimilarity = false;
    SimHashDocumentEncoder encoder(params);
    SDR output1({ params.size });
    SDR output2({ params.size });
    SDR output3({ params.size });

    encoder.encode(testDocMap1, output1);
    encoder.encode(testDocMap2, output2);
    encoder.encode(testDocMap3, output3);
    ASSERT_GT(output1.getOverlap(output3), output1.getOverlap(output2));
    ASSERT_GT(output1.getOverlap(output2), output2.getOverlap(output3));
  }

  // Test encoding unicode text, including with 'tokenSimilarity' on/off
  TEST(SimHashDocumentEncoder, testUnicode) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;

    // unicode tokenSimilarity ON
    params.tokenSimilarity = true;
    SimHashDocumentEncoder encoder1(params);
    SDR output1({ params.size });
    SDR output2({ params.size });
    encoder1.encode(testDocUni1, output1);
    encoder1.encode(testDocUni2, output2);
    ASSERT_GT(output1.getOverlap(output2), 65u);

    // unicode tokenSimilarity OFF
    params.tokenSimilarity = false;
    SimHashDocumentEncoder encoder2(params);
    output1.zero();
    output2.zero();
    encoder2.encode(testDocUni1, output1);
    encoder2.encode(testDocUni2, output2);
    ASSERT_LT(output1.getOverlap(output2), 65u);
  }

} // end namespace testing
