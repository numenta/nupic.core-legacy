/*
 * Copyright 2014 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <sstream>
#include <utility>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <kj/std/iostream.h>

#include <nupic/math/SparseBinaryMatrix.hpp>
#include <nupic/proto/SparseBinaryMatrixProto.capnp.h>
#include <nupic/types/Types.h>

using namespace nupic;

TEST(SparseBinaryMatrixReadWrite, EmptyMatrix) {
  SparseBinaryMatrix<UInt32, UInt32> m1, m2;

  m1.resize(3, 4);

  std::stringstream ss;

  // write
  capnp::MallocMessageBuilder message1;
  SparseBinaryMatrixProto::Builder protoBuilder =
      message1.initRoot<SparseBinaryMatrixProto>();
  m1.write(protoBuilder);
  kj::std::StdOutputStream out(ss);
  capnp::writeMessage(out, message1);

  // read
  kj::std::StdInputStream in(ss);
  capnp::InputStreamMessageReader message2(in);
  SparseBinaryMatrixProto::Reader protoReader =
      message2.getRoot<SparseBinaryMatrixProto>();
  m2.read(protoReader);

  // compare
  ASSERT_EQ(m1.nRows(), m2.nRows()) << "Number of rows don't match";
  ASSERT_EQ(m1.nCols(), m2.nCols()) << "Number of columns don't match";
}

TEST(SparseBinaryMatrixReadWrite, Basic) {
  SparseBinaryMatrix<UInt, Real> m1, m2;

  m1.resize(3, 4);
  m1.set(1, 1, 1);

  std::stringstream ss;

  // write
  capnp::MallocMessageBuilder message1;
  SparseBinaryMatrixProto::Builder protoBuilder =
      message1.initRoot<SparseBinaryMatrixProto>();
  m1.write(protoBuilder);
  kj::std::StdOutputStream out(ss);
  capnp::writeMessage(out, message1);

  // read
  kj::std::StdInputStream in(ss);
  capnp::InputStreamMessageReader message2(in);
  SparseBinaryMatrixProto::Reader protoReader =
      message2.getRoot<SparseBinaryMatrixProto>();
  m2.read(protoReader);

  // compare
  ASSERT_EQ(m1.nRows(), m2.nRows()) << "Number of rows don't match";
  ASSERT_EQ(m1.nCols(), m2.nCols()) << "Number of columns don't match";

  auto m1r1 = m1.getSparseRow(1);
  ASSERT_EQ(m1r1.size(), 1) << "Invalid # of elements in original matrix";
  auto m2r1 = m2.getSparseRow(1);
  ASSERT_EQ(m2r1.size(), 1) << "Invalid # of elements in copied matrix";

  ASSERT_EQ(m1r1[0], 1) << "Invalid col index in original matrix";
  ASSERT_EQ(m1r1[0], m2r1[0]) << "Invalid col index in copied matrix";
}
