# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


"""Common numpy operations for Numenta's algorithms"""


import numpy as np


def setCompare(a, b, aKey=None, bKey=None, leftMinusRight=False, rightMinusLeft=False):
    """
    Compute the intersection and differences between two arrays, comparing
    elements by their key.

    @param a (numpy array)
    The left set to compare.

    @param b (numpy array)
    The right set to compare.

    @param aKey (numpy array or None)
    If specified, elements in "a" are compared by their corresponding entry in
    "aKey".

    @param bKey (numpy array or None)
    If specified, elements in "b" are compared by their corresponding entry in
    "bKey".

    @param leftMinusRight
    If True, also calculate the set difference (a - b)

    @param rightMinusLeft
    If True, also calculate the set difference (b - a)

    @return (numpy array or tuple)
    Always returns the intersection of "a" and "b". The elements of this
    intersection are values from "a" (which may be different from the values of
    "b" or "aKey").

    If leftMinusRight or rightMinusLeft are True, it returns a tuple:
    - intersection (numpy array)
        See above
    - leftMinusRight (numpy array)
        The elements in a that are not in b
    - rightMinusLeft (numpy array)
        The elements in b that are not in a
    """
    aKey = aKey if aKey is not None else a
    bKey = bKey if bKey is not None else b

    aWithinBMask = np.in1d(aKey, bKey)

    if rightMinusLeft:
        bWithinAMask = np.in1d(bKey, aKey)

        if leftMinusRight:
            return (a[aWithinBMask], a[~aWithinBMask], b[bWithinAMask])
        else:
            return (a[aWithinBMask], b[~bWithinAMask])
    elif leftMinusRight:
        return (a[aWithinBMask], a[~aWithinBMask])
    else:
        return a[aWithinBMask]


def argmaxMulti(a, groupKeys, assumeSorted=False):
    """
    This is like numpy's argmax, but it returns multiple maximums.

    It gets the indices of the max values of each group in 'a', grouping the
    elements by their corresponding value in 'groupKeys'.

    @param a (numpy array)
    An array of values that will be compared

    @param groupKeys (numpy array)
    An array with the same length of 'a'. Each entry identifies the group for
    each 'a' value.

    @param assumeSorted (bool)
    If true, group keys must be organized together (e.g. sorted).

    @return (numpy array)
    The indices of one maximum value per group

    @example
        _argmaxMulti([5, 4, 7, 2, 9, 8],
                                 [0, 0, 0, 1, 1, 1])
    returns
        [2, 4]
    """

    if not assumeSorted:
        # Use a stable sort algorithm
        sorter = np.argsort(groupKeys, kind="mergesort")
        a = a[sorter]
        groupKeys = groupKeys[sorter]

    _, indices, lengths = np.unique(groupKeys, return_index=True, return_counts=True)

    maxValues = np.maximum.reduceat(a, indices)
    allMaxIndices = np.flatnonzero(np.repeat(maxValues, lengths) == a)

    # Break ties by finding the insertion points of the the group start indices
    # and using the values currently at those points. This approach will choose
    # the first occurrence of each max value.
    indices = allMaxIndices[np.searchsorted(allMaxIndices, indices)]

    if assumeSorted:
        return indices
    else:
        return sorter[indices]


def getAllCellsInColumns(columns, cellsPerColumn):
    """
    Calculate all cell indices in the specified columns.

    @param columns (numpy array)
    @param cellsPerColumn (int)

    @return (numpy array)
    All cells within the specified columns. The cells are in the same order as the
    provided columns, so they're sorted if the columns are sorted.
    """

    # Add
    #     [[beginningOfColumn0],
    #        [beginningOfColumn1],
    #         ...]
    # to
    #     [0, 1, 2, ..., cellsPerColumn - 1]
    # to get
    #     [beginningOfColumn0 + 0, beginningOfColumn0 + 1, ...
    #        beginningOfColumn1 + 0, ...
    #        ...]
    # then flatten it.
    return ((columns * cellsPerColumn).reshape((-1, 1)) + np.arange(cellsPerColumn, dtype="uint32")).flatten()
