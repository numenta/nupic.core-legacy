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
 * Unit tests for the SimHashDocumentEncoder
 */

#include "gtest/gtest.h"
#include <htm/encoders/SimHashDocumentEncoder.hpp>


namespace testing {

  using namespace htm;

  // @TODO see RDSE TestSerialize
  // @TODO conform test names below to rest of codebase
  // @TODO test param tokenSimilarity states

  /**
   * Make sure our encoder can initalize with basic params.
   */
  TEST(SimHashDocumentEncoder, validParametersTest) {
    SimHashDocumentEncoderParameters p;
    p.size = 10u;
    p.activeBits = 2u;

    SimHashDocumentEncoder e(p);

    ASSERT_EQ(e.parameters.size, 10u);
    ASSERT_EQ(e.parameters.activeBits, 2u);
  }

  /**
   * Make sure our encoder can initalize with the 'sparsity' param
   * instead of the 'activeBits' param.
   */
  TEST(SimHashDocumentEncoder, validSparsityParameterTest) {
    SimHashDocumentEncoderParameters p;
    p.size = 10u;
    p.sparsity = 0.20f;

    SimHashDocumentEncoder e(p);

    ASSERT_EQ(e.parameters.size, 10u);
    ASSERT_EQ(e.parameters.activeBits, 2u);
  }

  /**
   * Make sure we can call and pass args to encode() correctly.
   */
  TEST(SimHashDocumentEncoder, validEncodingInputPassTest) {
    SimHashDocumentEncoderParameters p;
    p.size = 400u;
    p.activeBits = 21u;

    SDR output({ p.size });
    SimHashDocumentEncoder e(p);

    EXPECT_NO_THROW(e.encode({"beta", "delta"}, output));
    EXPECT_ANY_THROW(e.encode({}, output));
  }

} // end namespace testing
