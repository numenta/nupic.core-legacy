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

/** @file */

#ifndef NTA_COND_PROB_TABLE_HPP
#define NTA_COND_PROB_TABLE_HPP

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseMatrix01.hpp>

namespace nupic {

////////////////////////////////////////////////////////////////////////////
/// Conditional Probablity Table
///
/// @b Responsibility
///  - Holds frequencies in a 2D grid of bins.
///
/// @b Resources/Ownerships:
///  - none
///
/// @b Notes:
/// Binning is not performed automatically by this class. Bin updates msut be
/// done one row at a time. This class uses nupic::SparseMatrix which is a
/// compressed sparse row matrix. Also maintains the row and column sumProp
/// distributions.
///
//////////////////////////////////////////////////////////////////////////////
class CondProbTable {
public:
  typedef enum {
    inferViterbi,
    inferMarginal,
    inferMaxProd,
    inferRowEvidence
  } inferType;

  static inferType convertInferType(const std::string &name) {
    if (name == "0")
      return inferViterbi;
    else if (name == "1")
      return inferMarginal;
    else if (name == "maxProp")
      return inferViterbi;
    else if (name == "sumProp")
      return inferMarginal;
    else {
      throw std::invalid_argument(
          "'" + name +
          "' is not a valid "
          "conditional probability table inference type.");
      return inferViterbi; // Unused.
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////
  /// Constructor
  ///
  /// Both the number of columns and the number of rows can be increased after
  /// construction by calling updateRow().
  ///
  /// @param hintNumCols  Number of columns in the table. This can be increased
  /// later
  ///                         via updateRow() but never decreased.
  /// @param hintNumRows  Number of rows in the table. This can be increased
  /// later
  ///                         via updateRow() but never decreased.
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  CondProbTable(const UInt hintNumCols = 0, const UInt hintNumRows = 0);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Destructor
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  virtual ~CondProbTable();

  /////////////////////////////////////////////////////////////////////////////////////
  /// Return the number of rows in the table
  ///
  /// @retval number of rows
  ///////////////////////////////////////////////////////////////////////////////////
  UInt numRows(void) {
    if (tableP_)
      return UInt(tableP_->nRows());
    else
      return hintNumRows_;
  }

  /////////////////////////////////////////////////////////////////////////////////////
  /// Return the number of columns in the table.
  ///
  /// @retval number of rows
  ///////////////////////////////////////////////////////////////////////////////////
  UInt numColumns(void) {
    if (tableP_)
      return tableP_->nCols();
    else
      return hintNumCols_;
  }

  /////////////////////////////////////////////////////////////////////////////////////
  /// Update a row with the given distribution.
  ///
  /// @param row          which row to update
  /// @param distribution the distribution to update the row with
  ///////////////////////////////////////////////////////////////////////////////////
  void updateRow(const UInt &row, const std::vector<Real> &distribution);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Return the probablity of the given distribution belonging to each row.
  ///
  /// Computes the sumProp probablity of each row given the input probability of
  /// each column.
  ///
  /// The semantics are as follows: If the distribution is P(col|e) where e is
  /// the evidence is col is the column, and the CPD represents P(row|col), then
  /// this calculates sum(P(col|e) P(row|col)) = P(row|e).
  ///
  /// The available inference methods are:
  /// inferMarginal -  Normalizes the distribution over the columns
  /// inferRowEvidence - Normalize the distribution over the rows.
  /// inferMaxProd - Computes the max product between each element of
  /// distribution
  ///                  and corresponding element of row.
  /// inferViterbi - works on a "clean" probability table, produced by finding
  /// the
  ///                 max element of each column, setting it to 1, and putting 0
  ///                 in all other elements of the column.
  ///
  /// @param distribution   the distribution to test - length equal to # of
  /// columns
  /// @param outScores      the return probablity of distribution belonging to
  /// each row -
  ///                         length equal to # of rows
  /// @param method         the method to use, one of either inferMarginal,
  /// inferMaxProd,
  ///                         inferRowEvidence, or inferViterbi
  ///////////////////////////////////////////////////////////////////////////////////
  void inferRow(const std::vector<Real> &distribution,
                std::vector<Real> &outScores, inferType infer = inferMarginal);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Form of inferRow that takes iterators as input
  ///
  /// @param distribution   the distribution to test - length equal to # of
  /// columns
  /// @param outScores      the return probablity of distribution belonging to
  /// each row
  ///                         length equal to # of rows
  /// @param method         the method to use, one of either inferMarginal,
  /// inferMaxProd,
  ///                         inferRowEvidence, or inferViterbi
  ///////////////////////////////////////////////////////////////////////////////////
  void inferRow(std::vector<Real>::const_iterator distribution,
                std::vector<Real>::iterator outScores,
                inferType infer = inferMarginal);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Get a row of the table out.
  ///
  /// @param row          which row to get
  /// @param contents     the row contents are written here
  ///////////////////////////////////////////////////////////////////////////////////
  void getRow(const UInt &row, std::vector<Real> &contents);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Get the entire table out as a sparse matrix
  ///
  /// @retval pointer to the table
  ///////////////////////////////////////////////////////////////////////////////////
  const SparseMatrix<UInt, Real> *getTable(void) const { return tableP_; }

  /////////////////////////////////////////////////////////////////////////////////////
  /// Save state to a stream
  ///
  /// @param state the stream to write to
  ///////////////////////////////////////////////////////////////////////////////////
  void saveState(std::ostream &state) const;

  /////////////////////////////////////////////////////////////////////////////////////
  /// Read state from a stream
  ///
  /// @param state the stream to read from
  ///////////////////////////////////////////////////////////////////////////////////
  void readState(std::istream &state);

private:
  /////////////////////////////////////////////////////////////////////////////////////
  /// Grow the matrix to have the given # of rows
  ///
  /// @rows          number of rows to grow to
  /// @cols          number of columns to grow to
  ///////////////////////////////////////////////////////////////////////////////////
  void grow(const UInt &rows, const UInt &cols);

  /////////////////////////////////////////////////////////////////////////////////////
  /// Make a "clean CPT". This is a copy of the CPT table with only the max
  /// element in each column kept and all others set to 0.
  ///
  ///////////////////////////////////////////////////////////////////////////////////
  void makeCleanCPT(void);

  UInt hintNumCols_;
  UInt hintNumRows_;
  SparseMatrix<UInt, Real> *tableP_;
  SparseMatrix01<UInt, Real> *cleanTableP_; // for inferViterbi
  bool cleanTableValid_;
  std::vector<Real> rowSums_;
  std::vector<Real> colSums_;
};

} // namespace nupic

#endif // NTA_COND_PROB_TABLE_HPP
