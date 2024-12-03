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

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/proto/SparseMatrixProto.capnp.h>
#include <nupic/types/Types.h>

using namespace nupic;

TEST(SparseMatrixReadWrite, EmptyMatrix) {
  SparseMatrix<UInt, Real> m1, m2;

  m1.resize(3, 4);

  std::stringstream ss;

  // write
  capnp::MallocMessageBuilder message1;
  SparseMatrixProto::Builder protoBuilder =
      message1.initRoot<SparseMatrixProto>();
  m1.write(protoBuilder);
  kj::std::StdOutputStream out(ss);
  capnp::writeMessage(out, message1);

  // read
  kj::std::StdInputStream in(ss);
  capnp::InputStreamMessageReader message2(in);
  SparseMatrixProto::Reader protoReader = message2.getRoot<SparseMatrixProto>();
  m2.read(protoReader);

  // compare
  ASSERT_EQ(m1.nRows(), m2.nRows()) << "Number of rows don't match";
  ASSERT_EQ(m1.nCols(), m2.nCols()) << "Number of columns don't match";
}

TEST(SparseMatrixReadWrite, Basic) {
  SparseMatrix<UInt, Real> m1, m2;

  m1.resize(3, 4);
  m1.setNonZero(1, 1, 3.0);

  std::stringstream ss;

  // write
  capnp::MallocMessageBuilder message1;
  SparseMatrixProto::Builder protoBuilder =
      message1.initRoot<SparseMatrixProto>();
  m1.write(protoBuilder);
  kj::std::StdOutputStream out(ss);
  capnp::writeMessage(out, message1);

  // read
  kj::std::StdInputStream in(ss);
  capnp::InputStreamMessageReader message2(in);
  SparseMatrixProto::Reader protoReader = message2.getRoot<SparseMatrixProto>();
  m2.read(protoReader);

  // compare
  ASSERT_EQ(m1.nRows(), m2.nRows()) << "Number of rows don't match";
  ASSERT_EQ(m1.nCols(), m2.nCols()) << "Number of columns don't match";

  std::vector<std::pair<UInt, Real>> m1r1(m1.nNonZerosOnRow(1));
  m1.getRowToSparse(1, m1r1.begin());
  ASSERT_EQ(m1r1.size(), 1) << "Invalid # of elements in original matrix";
  std::vector<std::pair<UInt, Real>> m2r1(m2.nNonZerosOnRow(1));
  m2.getRowToSparse(1, m2r1.begin());
  ASSERT_EQ(m2r1.size(), 1) << "Invalid # of elements in copied matrix";

  ASSERT_EQ(m1r1[0].first, 1) << "Invalid col index in original matrix";
  ASSERT_EQ(m1r1[0].first, m2r1[0].first)
      << "Invalid col index in copied matrix";
  ASSERT_EQ(m1r1[0].second, 3.0) << "Invalid value in original matrix";
  ASSERT_EQ(m1r1[0].second, m2r1[0].second) << "Invalid value in copied matrix";
}
