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

anomaly 0.0 = all active columns were predicted (every active column is overlapping with column with predictive cell),
anomaly 1.0 = none of active columns were predicted (none of active columns is overlapping with column with predictive cell)

"""
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import TemporalMemory
class Anomaly:


  @staticmethod
  def calculateRawAnomaly(activeCols, predictiveCols ):
    """

    :param activeColSDR: SDR with active columns
    :param predictiveColsSDR: SDR with predictive columns, means columns where some of the cells are predictive
    :return: Raw anomaly score in range <0.0, 1.0>
    """
    if activeCols.dimensions != predictiveCols.dimensions:
      raise ValueError("activeColumns must have same dimension as predictiveCellsSDR!")

    if activeCols.getSum() != 0:
      intersect = SDR(activeCols.dimensions)

      intersect.intersection(activeCols, predictiveCols)
      rawAnomaly = (activeCols.getSum() - intersect.getSum()) / activeCols.getSum()

      if rawAnomaly<0 or rawAnomaly>1.0:
        raise ValueError("rawAnomaly out of bounds! <0.0, 1.0>")
    else:
      rawAnomaly = 0

    return rawAnomaly