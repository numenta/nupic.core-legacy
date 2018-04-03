/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
 * ----------------------------------------------------------------------
 */

/** @file
 * SegmentMatrixAdapter class
 */

#ifndef NTA_SEGMENT_MATRIX_ADAPTER_HPP
#define NTA_SEGMENT_MATRIX_ADAPTER_HPP

#include <algorithm>
#include <vector>
#include <nupic/types/Types.hpp>

namespace nupic {

  /**
   * A data structure that stores dendrite segments as rows in a matrix.
   * The matrix itself is part of this class's public API. This class stores the
   * segments for each cell, and it can get the cell for each segment.
   *
   * This class is focused on Python consumers. C++ consumers could easily
   * accomplish all of this directly with a matrix class, but Python consumers
   * need a fast way of doing segment reads and writes in batches. This class
   * makes it possible to add rows in batch, maintaining mappings between cells
   * and segments, and providing batch lookups on those mappings.
   */
  template <typename Matrix>
  class SegmentMatrixAdapter {
  public:
    typedef typename Matrix::size_type size_type;

  public:
    SegmentMatrixAdapter(size_type nCells, size_type nCols)
      : matrix(0, nCols),
        segmentsForCell_(nCells)
    {
    }

    /**
     * Get the number of cells.
     */
    size_type nCells() const
    {
      return segmentsForCell_.size();
    }

    /**
     * Get the number of segments.
     */
    size_type nSegments() const
    {
      return cellForSegment_.size() - destroyedSegments_.size();
    }

    /**
     * Create a segment.
     *
     * @param cell
     * The cell that gets a new segment
     */
    size_type createSegment(size_type cell)
    {
      assert_valid_cell_(cell, "createSegment");

      if (destroyedSegments_.size() > 0)
      {
        const size_type segment = destroyedSegments_.back();
        destroyedSegments_.pop_back();
        segmentsForCell_[cell].push_back(segment);
        cellForSegment_[segment] = cell;
        return segment;
      }
      else
      {
        const size_type segment = matrix.nRows();
        matrix.resize(matrix.nRows() + 1, matrix.nCols());
        segmentsForCell_[cell].push_back(segment);
        cellForSegment_.push_back(cell);
        return segment;
      }
    }

    /**
     * Create one segment on each of the specified cells.
     *
     * @param cells
     * The cells that each get a new segment
     *
     * @param segments
     * An output array with the same size as 'cells'
     */
    template <typename InputIterator, typename OutputIterator>
    void createSegments(InputIterator cells_begin, InputIterator cells_end,
                        OutputIterator segments_begin)
    {
      assert_valid_cell_range_(cells_begin, cells_end, "createSegments");

      InputIterator cell = cells_begin;
      OutputIterator out = segments_begin;

      const size_type reclaimCount =
        std::min(destroyedSegments_.size(),
                 (size_t)std::distance(cell, cells_end));
      if (reclaimCount > 0)
      {
        for (auto segment = destroyedSegments_.end() - reclaimCount;
             segment != destroyedSegments_.end();
             ++cell, ++out, ++segment)
        {
          segmentsForCell_[*cell].push_back(*segment);
          cellForSegment_[*segment] = *cell;
          *out = *segment;
        }

        destroyedSegments_.resize(destroyedSegments_.size() - reclaimCount);
      }

      const size_type newCount = std::distance(cell, cells_end);
      if (newCount > 0)
      {
        const size_type firstNewRow = matrix.nRows();
        matrix.resize(matrix.nRows() + newCount, matrix.nCols());
        cellForSegment_.reserve(cellForSegment_.size() + newCount);

        for (size_type segment = firstNewRow;
             cell != cells_end;
             ++cell, ++out, ++segment)
        {
          segmentsForCell_[*cell].push_back(segment);
          cellForSegment_.push_back(*cell);
          *out = segment;
        }
      }
    }

    /**
     * Destroy a segment. Remove it from its cell and remove all of its synapses
     * in the Matrix.
     *
     * This doesn't remove the segment's row from the Matrix, so the other
     * segments' row numbers are unaffected.
     *
     * @param segment
     * The segment to destroy
     */
    void destroySegment(size_type segment)
    {
      assert_valid_segment_(segment, "destroySegment");

      matrix.setRowToZero(segment);

      std::vector<size_type>& ownerList =
        segmentsForCell_[cellForSegment_[segment]];
      ownerList.erase(std::find(ownerList.begin(), ownerList.end(),
                                segment));

      cellForSegment_[segment] = (size_type)-1;

      destroyedSegments_.push_back(segment);
    }

    /**
     * Destroy multiple segments.
     *
     * @param segments
     * The segments to destroy
     */
    template <typename InputIterator>
    void destroySegments(InputIterator segments_begin, InputIterator segments_end)
    {
      assert_valid_segment_range_(segments_begin, segments_end, "destroySegments");

      destroyedSegments_.reserve(destroyedSegments_.size() +
                                 std::distance(segments_begin, segments_end));

      for (InputIterator segment = segments_begin;
           segment != segments_end;
           ++segment)
      {
        destroySegment(*segment);
      }
    }

    /**
     * Get the number of segments on each of the provided cells.
     *
     * @param cells
     * The cells to check
     *
     * @param counts
     * Output array with the same length as 'cells'
     */
    template <typename InputIterator, typename OutputIterator>
    void getSegmentCounts(InputIterator cells_begin, InputIterator cells_end,
                          OutputIterator counts_begin) const
    {
      assert_valid_cell_range_(cells_begin, cells_end, "getSegmentCounts");

      OutputIterator out = counts_begin;

      for (InputIterator cell = cells_begin;
           cell != cells_end;
           ++cell, ++out)
      {
        *out = segmentsForCell_[*cell].size();
      }
    }

    /**
     * Get the segments for a cell.
     *
     * @param cell
     * The cell
     */
    const std::vector<size_type>& getSegmentsForCell(size_type cell) const
    {
      assert_valid_cell_(cell, "getSegmentsForCell");

      return segmentsForCell_[cell];
    }

    /**
     * Sort an array of segments by cell in increasing order.
     *
     * @param segments
     * The segment array. It's sorted in-place.
     */
    template <typename InputIterator>
    void sortSegmentsByCell(InputIterator segments_begin,
                            InputIterator segments_end) const
    {
      assert_valid_segment_range_(segments_begin, segments_end,
                                  "sortSegmentsByCell");

      std::sort(segments_begin, segments_end,
                [&](size_type a, size_type b)
                {
                  return cellForSegment_[a] < cellForSegment_[b];
                });
    }

    /**
     * Return the subset of segments that are on the provided cells.
     *
     * @param segments
     * The segments to filter. Must be sorted by cell.
     *
     * @param cells
     * The cells whose segments we want to keep. Must be sorted.
     */
    template <typename InputIterator1, typename InputIterator2>
    std::vector<size_type> filterSegmentsByCell(
      InputIterator1 segments_begin, InputIterator1 segments_end,
      InputIterator2 cells_begin, InputIterator2 cells_end) const
    {
      assert_valid_sorted_segment_range_(segments_begin, segments_end,
                                         "filterSegmentsByCell");
      assert_valid_sorted_cell_range_(cells_begin, cells_end,
                                      "filterSegmentsByCell");

      std::vector<size_type> filteredSegments;

      InputIterator1 segment = segments_begin;
      InputIterator2 cell = cells_begin;

      bool finished = (segment == segments_end) || (cell == cells_end);

      while (!finished)
      {
        while (cellForSegment_[*segment] < *cell)
        {
          finished = (++segment == segments_end);
          if (finished) break;
        }

        if (finished) break;

        if (cellForSegment_[*segment] == *cell)
        {
          filteredSegments.push_back(*segment);
          finished = (++segment == segments_end);
          if (finished) break;
        }

        while (*cell < cellForSegment_[*segment])
        {
          finished = (++cell == cells_end);
          if (finished) break;
        }
      }

      return filteredSegments;
    }

    /**
     * Get the cell for each provided segment.
     *
     * @param segments
     * The segments to query
     *
     * @param cells
     * Output array with the same length as 'segments'
     */
    template <typename InputIterator, typename OutputIterator>
    void mapSegmentsToCells(
      InputIterator segments_begin, InputIterator segments_end,
      OutputIterator cells_begin) const
    {
      assert_valid_segment_range_(segments_begin, segments_end,
                                  "mapSegmentsToCells");

      OutputIterator out = cells_begin;

      for (InputIterator segment = segments_begin;
           segment != segments_end;
           ++segment, ++out)
      {
        *out = cellForSegment_[*segment];
      }
    }

  public:

    /**
     * The underlying Matrix. Each row is a segment.
     *
     * Don't add or remove rows directly. Use createSegment / destroySegment.
     */
    Matrix matrix;

  private:

    void assert_valid_segment_(size_type segment, const char *where) const
    {
#ifdef NTA_ASSERTIONS_ON
      NTA_ASSERT(segment < matrix.nRows())
        << "SegmentMatrixAdapter " << where << ": Invalid segment: " << segment
        << " - Should be < " << matrix.nRows();

      NTA_ASSERT(cellForSegment_[segment] != (size_type)-1)
        << "SegmentMatrixAdapter " << where << ": Invalid segment: " << segment
        << " -- This segment has been destroyed.";
#endif
    }

    template <typename Iterator>
    void assert_valid_segment_range_(Iterator segments_begin,
                                     Iterator segments_end,
                                     const char *where) const
    {
#ifdef NTA_ASSERTIONS_ON
      for (Iterator segment = segments_begin;
           segment != segments_end;
           ++segment)
      {
        assert_valid_segment_(*segment, where);
      }
#endif
    }

    template <typename Iterator>
    void assert_valid_sorted_segment_range_(Iterator segments_begin,
                                            Iterator segments_end,
                                            const char *where) const
    {
#ifdef NTA_ASSERTIONS_ON
      for (Iterator segment = segments_begin;
           segment != segments_end;
           ++segment)
      {
        assert_valid_segment_(*segment, where);

        if (segment != segments_begin)
        {
          NTA_ASSERT(cellForSegment_[*(segment - 1)] <=
                     cellForSegment_[*segment])
            << "SegmentMatrixAdapter " << where << ": Segments must be sorted "
            << "by cell. Found cell " << cellForSegment_[*(segment - 1)]
            << " before cell " << cellForSegment_[*segment];
        }
      }
#endif
    }

    void assert_valid_cell_(size_type cell, const char *where) const {
#ifdef NTA_ASSERTIONS_ON
      NTA_ASSERT(cell < nCells())
        << "SegmentMatrixAdapter " << where << ": Invalid cell: " << cell
        << " - Should be < " << nCells();
#endif
    }

    template <typename Iterator>
    void assert_valid_cell_range_(Iterator cells_begin,
                                  Iterator cells_end,
                                  const char *where) const
    {
#ifdef NTA_ASSERTIONS_ON
      for (Iterator cell = cells_begin; cell != cells_end; ++cell)
      {
        assert_valid_cell_(*cell, where);
      }
#endif
    }

    template <typename Iterator>
    void assert_valid_sorted_cell_range_(Iterator cells_begin,
                                         Iterator cells_end,
                                         const char *where) const
    {
#ifdef NTA_ASSERTIONS_ON
      for (Iterator cell = cells_begin; cell != cells_end; ++cell)
      {
        assert_valid_cell_(*cell, where);

        if (cell != cells_begin)
        {
          NTA_ASSERT(*(cell - 1) <= *cell)
            << "SegmentMatrixAdapter " << where << ": Cells must be sorted. "
            << "Found cell " << *(cell - 1) << " before cell " << *cell;
        }
      }
#endif
    }

  private:

    // One-to-one mapping: segment -> cell
    std::vector<size_type> cellForSegment_;

    // One-to-many mapping: cell -> segments
    std::vector<std::vector<size_type> > segmentsForCell_;

    // Rather that deleting rows from the matrix, keep a list of rows that can
    // be reused. Otherwise the segment numbers in the 'cellForSegment' and
    // 'segmentsForCell' vectors would be invalidated every time a segment gets
    // destroyed.
    std::vector<size_type> destroyedSegments_;
  };

} // end namespace nupic

#endif // NTA_SEGMENT_MATRIX_ADAPTER_HPP
