/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

#include <sstream>
#include <utility>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>
#include <gtest/gtest.h>

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/proto/SparseMatrixProto.capnp.h>
#include <nupic/types/Types.h>

namespace nupic
{
  namespace
  {

    TEST(SparseMatrixReadWrite, EmptyMatrix)
    {
      SparseMatrix<UInt, Real> m1, m2;

      m1.resize(3, 4);

      std::stringstream ss;

      // write
      capnp::MallocMessageBuilder message1;
      SparseMatrixProto::Builder protoBuilder = message1.initRoot<SparseMatrixProto>();
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

    TEST(SparseMatrixReadWrite, Basic)
    {
      SparseMatrix<UInt, Real> m1, m2;

      m1.resize(3, 4);
      m1.setNonZero(1, 1, 3.0);

      std::stringstream ss;

      // write
      capnp::MallocMessageBuilder message1;
      SparseMatrixProto::Builder protoBuilder = message1.initRoot<SparseMatrixProto>();
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

      std::vector<std::pair<UInt, Real> > m1r1(m1.nNonZerosOnRow(1));
      m1.getRowToSparse(1, m1r1.begin());
      ASSERT_EQ(m1r1.size(), 1) << "Invalid # of elements in original matrix";
      std::vector<std::pair<UInt, Real> > m2r1(m2.nNonZerosOnRow(1));
      m2.getRowToSparse(1, m2r1.begin());
      ASSERT_EQ(m2r1.size(), 1) << "Invalid # of elements in copied matrix";

      ASSERT_EQ(m1r1[0].first, 1) << "Invalid col index in original matrix";
      ASSERT_EQ(m1r1[0].first, m2r1[0].first) << "Invalid col index in copied matrix";
      ASSERT_EQ(m1r1[0].second, 3.0) << "Invalid value in original matrix";
      ASSERT_EQ(m1r1[0].second, m2r1[0].second) << "Invalid value in copied matrix";
    }

  }  // namespace
}  // namespace nupic
