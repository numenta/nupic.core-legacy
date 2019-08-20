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

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include <htm/encoders/SimHashDocumentEncoder.hpp>


namespace testing {

  using namespace htm;

  /* Shared Test Strings */
  const std::vector<std::string> testDoc1 = {
    "abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"  };
  const std::vector<std::string> testDoc2 = {
    "klmno", "pqrst",  "uvwxy",  "z1234",  "56789"  };
  const std::vector<std::string> testDoc3 = {
    "z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"  };
  const std::vector<std::string> testDoc4 = {
    "z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP" };


  /**
   * TESTS
   */

  // Test a basic use-case in human-readable form.
  //  Documents:
  //    1: "The sky is beautiful today"
  //    2: "The sun is beautiful today"  (similar above, differ below)
  //    3: "Who did my homework  today"
  //  Test Expectations:
  //    1 vs 2 = very similar and should receive similar encodings
  //    2 vs 3 = very different and should receive differing encodings
  TEST(SimHashDocumentEncoder, testBasicExampleUseCase) {
    std::string testDocEasy1 = "The sky is beautiful today";
    std::string testDocEasy2 = "The sun is beautiful today"; // similar above, differ below
    std::string testDocEasy3 = "Who did my homework  today";

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
    SimHashDocumentEncoderParameters params1;
    params1.size = 400u;
    params1.activeBits = 20u;

    SimHashDocumentEncoder encoder1(params1);
    ASSERT_EQ(encoder1.parameters.size, params1.size);
    ASSERT_EQ(encoder1.parameters.activeBits, params1.activeBits);

    // test bad encoder params - both activeBits and sparsity
    SimHashDocumentEncoderParameters params2;
    params2.size = 400u;
    params2.activeBits = 20u;
    params2.sparsity = 0.666f;
    EXPECT_ANY_THROW(SimHashDocumentEncoder encoder2(params2));

    // test bad encoder params - neither activeBits or sparsity
    SimHashDocumentEncoderParameters params3;
    params3.size = 400u;
    EXPECT_ANY_THROW(SimHashDocumentEncoder encoder3(params3));

    // test good encoder params - using 'sparsity' instead of 'activeBits'
    SimHashDocumentEncoderParameters params4;
    params4.size = 400u;
    params4.sparsity = 0.05f;
    SimHashDocumentEncoder encoder4(params4);
    ASSERT_EQ(encoder4.parameters.size, params4.size);
    ASSERT_EQ(encoder4.parameters.activeBits, 20u);

    // test bad encoder params - frequency should be ceiling > floor
    SimHashDocumentEncoderParameters params5;
    params4.size = 400u;
    params4.sparsity = 0.05f;
    params5.frequencyCeiling = 3;
    params5.frequencyFloor = 6;
    EXPECT_ANY_THROW(SimHashDocumentEncoder encoder5(params5));
  }

  // Test a basic encoding, try a few failure cases
  TEST(SimHashDocumentEncoder, testEncoding) {
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.activeBits = 20u;

    // test basic list calling style
    SDR output({ params.size });
    SimHashDocumentEncoder encoder(params);
    EXPECT_NO_THROW(encoder.encode(testDoc1, output));
    ASSERT_EQ(output.size, params.size);
    ASSERT_EQ(output.getSum(), params.activeBits);

    // make sure simple alternate string calling style matches
    SDR output2({ params.size });
    SimHashDocumentEncoder encoder2(params);
    std::string value2 = "abcde fghij klmno pqrst uvwxy";
    encoder2.encode(value2, output2);                  // full str
    ASSERT_EQ(output.getSparse(), output2.getSparse());

    // test empty inputs => output of all zeros
    const std::vector<std::string> value3({});
    SDR outputZ({ params.size });
    SDR output3({ params.size });
    outputZ.zero();
    encoder.encode(value3, output3);
    ASSERT_EQ(output3.getSparse(), outputZ.getSparse());

    std::string value4 = "";
    SDR output4({ params.size });
    encoder2.encode(value4, output4);
    ASSERT_EQ(output4.getSparse(), outputZ.getSparse());
  }

  // Test excludes param
  TEST(SimHashDocumentEncoder, testExcludes) {
    std::vector<std::string> fullList = {
      "seasons", "change", "mad", "things", "rearrange",
      "but", "it", "all", "stays", "the", "same" };
    std::vector<std::string> keepList = {
      "but", "it", "all", "stays", "the", "same" };
    std::vector<std::string> nopeList = {
      "seasons", "change", "mad", "things", "rearrange" };

    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    SimHashDocumentEncoder encoder1(params);
    SDR output1({ params.size });
    encoder1.encode(fullList, output1);

    SimHashDocumentEncoder encoder2(params);
    SDR output2({ params.size });
    encoder2.encode(keepList, output2);

    params.excludes = nopeList;
    SimHashDocumentEncoder encoder3(params);
    SDR output3({ params.size });
    encoder3.encode(fullList, output3);

    ASSERT_NE(output1, output2); // full != part
    ASSERT_NE(output1, output3); // full != (full - nope)
    ASSERT_EQ(output2, output3); // part == (full - nope)
  }

  // Test token frequency floor/ceiling
  TEST(SimHashDocumentEncoder, testFrequency) {
    const std::string tokens = "a a a b b c d d d d e e f"; // min 1 max 4
    const std::string charTokens = {
      "abbbbbbcccdefg aaaaaabccchijk aaabcccccclmno" };

    // test token frequency floor and ceiling
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    SimHashDocumentEncoder encoder1(params);
    SDR output1({ encoder1.size });
    encoder1.encode(tokens, output1);

    params.frequencyFloor = 1u;
    SimHashDocumentEncoder encoder2(params);
    SDR output2({ encoder2.size });
    encoder2.encode(tokens, output2);

    params.frequencyFloor = 0;
    params.frequencyCeiling = 4u;
    SimHashDocumentEncoder encoder3(params);
    SDR output3({ encoder3.size });
    encoder3.encode(tokens, output3);

    ASSERT_NE(output1, output2);
    ASSERT_NE(output1, output3);
    ASSERT_NE(output2, output3);

    // test character frequency ceiling (only)
    SimHashDocumentEncoderParameters params2;
    params2.size = 400u;
    params2.sparsity = 0.33f;
    params2.tokenSimilarity = true;
    SimHashDocumentEncoder encoder4(params2);
    SDR output4({ encoder4.size });
    encoder4.encode(charTokens, output4);

    params2.frequencyCeiling = 3u;
    SimHashDocumentEncoder encoder5(params2);
    SDR output5({ encoder5.size });
    encoder5.encode(charTokens, output5);

    ASSERT_NE(output4, output5);
  }

  // Test Serialization and Deserialization
  TEST(SimHashDocumentEncoder, testSerialize) {
    std::map<std::string, UInt> vocab = {
      { "hear", 2 }, { "nothing", 4 }, { "but", 1 }, { "a", 1 },
      { "rushing", 4 }, { "sound", 3 } };
    std::vector<std::string> document = {
      "hear", "any", "sound", "sound", "louder", "but", "walls" };

    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    params.vocabulary = vocab;

    SimHashDocumentEncoder encoder1(params);
    std::stringstream buffer;
    encoder1.save(buffer);
    SDR output1({ encoder1.size });
    encoder1.encode(document, output1);

    SimHashDocumentEncoder encoder2;
    encoder2.load(buffer);
    SDR output2({ encoder2.size });
    encoder2.encode(document, output2);

    ASSERT_EQ(output1.size, output2.size);
    ASSERT_EQ(output1.getDense(), output2.getDense());
  }

  // Test encoding with case in/sensitivity
  TEST(SimHashDocumentEncoder, testTokenCaseSensitivity) {
    // local test strings
    const std::vector<std::string> testDocCase1 =
      { "alpha", "bravo",  "delta",  "echo",  "foxtrot", "hotel" };
    const std::vector<std::string> testDocCase2 =
      { "ALPHA", "BRAVO",  "DELTA",  "ECHO",  "FOXTROT", "HOTEL" };
    const std::vector<std::string> part = { "eCHo", "foXTROt", "hOtEl" };
    const std::vector<std::string> discard = { "AlPHa", "BRaVo", "dELTa" };
    const std::map<std::string, UInt> vocab = {
      { "EcHo", 1 }, { "FOxtRoT", 1 }, { "HoTeL", 1 }};

    // caseSensitivity=ON
    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    params.caseSensitivity = true;
    SimHashDocumentEncoder encoder1(params);
    SDR output1({ params.size });
    SDR output2({ params.size });
    encoder1.encode(testDocCase1, output1);
    encoder1.encode(testDocCase2, output2);
    ASSERT_NE(output1, output2);

    // caseSensitivity=OFF
    params.caseSensitivity = false;
    SimHashDocumentEncoder encoder2(params);
    output1.zero();
    output2.zero();
    encoder2.encode(testDocCase1, output1);
    encoder2.encode(testDocCase2, output2);
    ASSERT_EQ(output1, output2);

    // caseSensitivity=OFF +excludes
    params.excludes = discard;
    SimHashDocumentEncoder encoder3(params);
    SDR output3a({ params.size });
    SDR output3b({ params.size });
    encoder3.encode(testDocCase1, output3a);
    encoder3.encode(part, output3b);
    ASSERT_EQ(output3a, output3b);

    // caseSensitivity=OFF +vocabulary
    SimHashDocumentEncoderParameters params4;
    params4.size = 400u;
    params4.sparsity = 0.33f;
    params4.vocabulary = vocab;
    SimHashDocumentEncoder encoder4(params4);
    SDR output4a({ params4.size });
    SDR output4b({ params4.size });
    encoder4.encode(testDocCase1, output4a);
    encoder4.encode(part, output4b);
    ASSERT_EQ(output4a, output4b);
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
    const std::map<std::string, UInt> weights = {
      { "aaa", 4 }, { "bbb", 2 }, { "ccc", 2 }, { "ddd", 4 }, { "eee", 2 },
      { "fff", 2 }, { "sss", 1 } };
    const std::vector<std::string> doc1 = { "aaa", "bbb", "ccc", "ddd", "sss" };
    const std::vector<std::string> doc2 = { "eee", "bbb", "ccc", "fff", "sss" };
    const std::vector<std::string> doc3 = { "aaa", "eee", "fff", "ddd" };

    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    params.vocabulary = weights;
    SimHashDocumentEncoder encoder(params);
    SDR output1({ params.size });
    SDR output2({ params.size });
    SDR output3({ params.size });
    encoder.encode(doc1, output1);
    encoder.encode(doc2, output2);
    encoder.encode(doc3, output3);
    ASSERT_GT(output1.getOverlap(output3), output1.getOverlap(output2));
    ASSERT_GT(output1.getOverlap(output2), output2.getOverlap(output3));
  }

  // test vocabulary
  TEST(SimHashDocumentEncoder, testTokenVocabulary) {
    std::map<std::string, UInt> vocabulary = {
      { "a", 1 }, { "b", 2 }, { "c", 3 }, { "d", 4 }, { "e", 5 }, { "f", 6 },
      { "g", 1 }, { "h", 2 }, { "i", 3 }, { "j", 4 }, { "k", 5 }, { "l", 6 }};
    std::string input1 = "a b c d e f";
    std::string input2 = "a b c d e f t u w x y z";

    SimHashDocumentEncoderParameters params;
    params.size = 400u;
    params.sparsity = 0.33f;
    params.vocabulary = vocabulary;

    // vocabulary +encodeOrphans
    params.encodeOrphans = true;
    SimHashDocumentEncoder encoder1(params);
    SDR output1a({ params.size });
    SDR output1b({ params.size });
    encoder1.encode(input1, output1a);
    encoder1.encode(input2, output1b);
    ASSERT_NE(output1a.getDense(), output1b.getDense());

    // vocabulary -encodeOrphans
    params.encodeOrphans = false;
    SimHashDocumentEncoder encoder2(params);
    SDR output2a({ params.size });
    SDR output2b({ params.size });
    encoder2.encode(input1, output2a);
    encoder2.encode(input2, output2b);
    ASSERT_EQ(output2a.getDense(), output2b.getDense());
  }

  // Test encoding unicode text, including with 'tokenSimilarity' on/off
  TEST(SimHashDocumentEncoder, testUnicode) {
    const std::vector<std::string> testDocUni1 = {
      u8"\u0395\u0396\u0397\u0398\u0399",
      u8"\u0400\u0401\u0402\u0403\u0404",
      u8"\u0405\u0406\u0407\u0408\u0409" };
    const std::vector<std::string> testDocUni2 = {
      u8"\u0395\u0396\u0397\u0398\u0399\u0410",
      u8"\u0400\u0401\u0402\u0403\u0404\u0410",
      u8"\u0405\u0406\u0407\u0408\u0409\u0410" };

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
