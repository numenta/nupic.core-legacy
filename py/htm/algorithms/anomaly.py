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


There are two ways to use the code: using the
:class:`.anomaly_li

"""


class Anomaly:
  @staticmethod
  def calculateRawAnomaly(activeColSDR, predictiveCellsSDR ):

    intersect = SDR(activeColSDR.dimensions)
    intersect.intersection(activeColSDR, cellsToColumns(predictiveCellsSDR))

    if activeColSDR.getSum() != 0:
      rawAnomaly = (activeColSDR.getSum() - intersect.getSum()) / activeColSDR.getSum()
    else:
      rawAnomaly = 0

    return rawAnomaly

  # converts cells SDR to columns SDR
  @staticmethod
  def cellsToColumns(cells):
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