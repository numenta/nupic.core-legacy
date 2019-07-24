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

  // Caution! Tests further below are very sensitive to these exact strings.
  const std::vector<std::string> testDoc1 = { "abcde", "fghij",  "klmno",  "pqrst",  "uvwxy"  };
  const std::vector<std::string> testDoc2 = { "klmno", "pqrst",  "uvwxy",  "z1234",  "56789"  };
  const std::vector<std::string> testDoc3 = { "z1234", "56789",  "0ABCD",  "EFGHI",  "JKLMN"  };
  const std::vector<std::string> testDoc4 = { "z1234", "56789P", "0ABCDP", "EFGHIP", "JKLMNP" };


  TEST(SimHashDocumentEncoder, testConstruct) {
    SimHashDocumentEncoderParameters params;
    params.size = 10u;
    params.activeBits = 2u;

    SimHashDocumentEncoder encoder(params);

    ASSERT_EQ(encoder.parameters.size, 10u);
    ASSERT_EQ(encoder.parameters.activeBits, 2u);
  }

  TEST(SimHashDocumentEncoder, testConstructParameterSparsity) {
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

    EXPECT_ANY_THROW(encoder.encode({}, output));
    EXPECT_NO_THROW(encoder.encode(testDoc1, output));

    ASSERT_EQ(output.size, params.size);
    ASSERT_EQ(output.getSum(), params.activeBits);
  }

  TEST(SimHashDocumentEncoder, testEncodingTokenSimilarityOn) {
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

  TEST(SimHashDocumentEncoder, testEncodingTokenSimilarityOff) {
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

  TEST(SimHashDocumentEncoder, testSerialize) {
    SimHashDocumentEncoderParameters params;
    params.size = 1000u;
    params.sparsity = 0.25f;

    SimHashDocumentEncoder encoderA(params);
    std::stringstream buffer;
    encoderA.save(buffer);

      LogItem::setLogLevel(LogLevel_Verbose);

    SDR outputA({ encoderA.parameters.size });
    encoderA.encode(testDoc1, outputA);

    SimHashDocumentEncoder encoderB;
    encoderB.load(buffer);

    SDR outputB({ encoderB.parameters.size });
    encoderB.encode(testDoc1, outputB);

    ASSERT_EQ(outputA, outputB);
  }

} // end namespace testing
