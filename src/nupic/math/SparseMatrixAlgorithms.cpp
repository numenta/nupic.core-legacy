/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * External algorithms for operating on a sparse matrix.
 */

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseMatrixAlgorithms.hpp>

/**
 * This file contains the declarations for the two static tables that we compute
 * to speed-up log sum and log diff.
 */

namespace nupic {

// The two tables used when approximating logSum and logDiff.
std::vector<LogSumApprox::value_type> LogSumApprox::table;
std::vector<LogDiffApprox::value_type> LogDiffApprox::table;

} // end namespace nupic
