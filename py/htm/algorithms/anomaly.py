# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Zbysek Zapadlik
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
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

"""

There are currently several ways to get anomaly score:
1) C++ Anomaly.hpp
2) C++ & python TM.anomaly
3) C++ & python AnomalyLikelihood
4) this class

This class is pure python code calculating raw anomaly score.
Was created for reason resulting from situation when TM.compute call is break down to individual steps.

Simple calculates what is the overlap ratio between active columns
and columns with predictive cells.

anomaly 1.0 = all active columns were predicted (every active column is overlapping with column with predictive cell),
anomaly 0.0 = none of active columns were predicted (none of active columns is overlapping with column with predictive cell)

"""
from htm.bindings.sdr import SDR

class Anomaly:


  @staticmethod
  def calculateRawAnomaly(activeColSDR, predictiveCellsSDR ):
    """

    :param activeColSDR: SDR with active columns - one dimensional - like SDR(100)
    :param predictiveCellsSDR: SDR with predictive cells - two dimensional - like SDR(100,30), means 30 cells per column
    :return:
    """
    if len(activeColSDR.dimensions) !=1:
      raise ValueError("activeColumns SDR must be of dimension 1")
    if len(predictiveCellsSDR.dimensions) != 2:
      raise ValueError("predictiveCells SDR must be of dimension 2")

    if activeColSDR.getSum() != 0:
      intersect = SDR(activeColSDR.dimensions)

      intersect.intersection(activeColSDR, Anomaly._cellsToColumns(predictiveCellsSDR))
      rawAnomaly = (activeColSDR.getSum() - intersect.getSum()) / activeColSDR.getSum()
    else:
      rawAnomaly = 0

    return rawAnomaly

  # converts cells SDR to columns SDR
  # expects two dimensional SDR, like SDR(100,30) - means 100 columns with 30 cells for each column
  @staticmethod
  def _cellsToColumns(cells):
    nOfColumns = cells.dimensions[0]
    cellsPerColumn = cells.dimensions[1]

    arr = []
    for cell in cells.sparse:
      a = (int)(cell / cellsPerColumn)
      if a not in arr:
        arr.append(a)

    columns = SDR(nOfColumns)
    columns.sparse = arr

    return columns