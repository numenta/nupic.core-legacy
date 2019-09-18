# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2014, Numenta, Inc. 
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

import hashlib
import itertools

import numpy as np
from htm.bindings.math import Random


class CoordinateEncoder():
    """
    Given a coordinate in an N-dimensional space, and a radius around
    that coordinate, the Coordinate Encoder returns an SDR representation
    of that position.

    The Coordinate Encoder uses an N-dimensional integer coordinate space.
    For example, a valid coordinate in this space is (150, -49, 58), whereas
    an invalid coordinate would be (55.4, -5, 85.8475).

    It uses the following algorithm:

    1. Find all the coordinates around the input coordinate, within the
         specified radius.
    2. For each coordinate, use a uniform hash function to
         deterministically map it to a real number between 0 and 1. This is the
         "order" of the coordinate.
    3. Of these coordinates, pick the top W by order, where W is the
         number of active bits desired in the SDR.
    4. For each of these W coordinates, use a uniform hash function to
         deterministically map it to one of the bits in the SDR. Make this bit
         active.
    5. This results in a final SDR with exactly W bits active (barring chance hash
         collisions).

    """

    def __init__(self, w=21, n=1000, name=None, verbosity=0):
        # Validate inputs
        if (w <= 0) or (w % 2 == 0):
            raise ValueError("w must be an odd positive integer")

        if (n <= 6 * w) or (not isinstance(n, int)):
            raise ValueError("n must be an int strictly greater than 6*w. For "
                             "good results we recommend n be strictly greater "
                             "than 11*w")

        self.w = w
        self.n = n
        self.verbosity = verbosity
        self.encoders = None

        if name is None:
            name = "[%s:%s]" % (self.n, self.w)
        self.name = name


    def getWidth(self):
        """
        Should return the output width, in bits.
    
        :return: (int) output width in bits
        """
        return self.n


    def getDescription(self):
        """
        This returns a list of tuples, each containing (``name``, ``offset``).
        The ``name`` is a string description of each sub-field, and ``offset`` is
        the bit offset of the sub-field for that encoder.
    
        :return: list of tuples containing (name, offset)
        """
        return [('coordinate', 0), ('radius', 1)]


    def getScalars(self, inputData):
        """
        Returns a numpy array containing the sub-field scalar value(s) for
        each sub-field of the ``inputData``. To get the associated field names for
        each of the scalar values, call :meth:`.getScalarNames()`.
    
        The intent of the scalar representation of a sub-field is to provide a
        baseline for measuring error differences. You can compare the scalar value
        of the inputData with the scalar value returned from :meth:`.topDownCompute`
        on a top-down representation to evaluate prediction accuracy, for example.
    
        :param inputData: The data from the source. This is typically an object with
                     members
        :return: array of scalar values
        """
        return np.array([0]*len(inputData))


    def encode(self, inputData, output):
        """
        Encodes inputData and puts the encoded value into the output SDR.

        @param inputData (tuple) Contains coordinate (np.array, N-dimensional integer coordinate) and radius (int)
        @param output (SDR) Stores encoded SDR
        """
        (coordinate, radius) = inputData

        assert isinstance(radius, int), ("Expected integer radius, got: {} ({})".format(radius, type(radius)))

        neighbors = self._neighbors(coordinate, radius)
        winners = self._topWCoordinates(neighbors, self.w)

        bitFn = lambda coordinate: self._bitForCoordinate(coordinate, self.n)
        indices = np.array([bitFn(w) for w in winners])

        output.sparse = list(set(indices))


    @staticmethod
    def _neighbors(coordinate, radius):
        """
        Returns coordinates around given coordinate, within given radius.
        Includes given coordinate.

        @param coordinate (np.array) N-dimensional integer coordinate
        @param radius (int) Radius around `coordinate`

        @return (np.array) List of coordinates
        """
        ranges = (range(n-radius, n+radius+1) for n in coordinate.tolist())
        return np.array(list(itertools.product(*ranges)))


    @classmethod
    def _topWCoordinates(cls, coordinates, w):
        """
        Returns the top W coordinates by order.

        @param coordinates (np.array) A 2D np array, where each element is a coordinate
        @param w (int) Number of top coordinates to return
        @return (np.array) A subset of `coordinates`, containing only the top ones by order
        """
        orders = np.array([cls._orderForCoordinate(c) for c in coordinates.tolist()])
        indices = np.argsort(orders)[-w:]
        return coordinates[indices]


    @staticmethod
    def _hashCoordinate(coordinate):
        """
        Hash a coordinate to a 64 bit integer.
        """
        coordinateStr = ",".join(str(v) for v in coordinate).encode('utf-8')
        # Compute the hash and convert to 64 bit int.
        coord_hash = int(int(hashlib.md5(coordinateStr).hexdigest(), 16) % (2 ** 64))
        return coord_hash


    @classmethod
    def _orderForCoordinate(cls, coordinate):
        """
        Returns the order for a coordinate.

        @param coordinate (np.array) Coordinate
        @return (float) A value in the interval [0, 1), representing the order of the coordinate
        """
        seed = cls._hashCoordinate(coordinate)
        rng = Random(seed)
        return rng.getReal64()


    @classmethod
    def _bitForCoordinate(cls, coordinate, n):
        """
        Maps the coordinate to a bit in the SDR.

        @param coordinate (np.array) Coordinate
        @param n (int) The number of available bits in the SDR
        @return (int) The index to a bit in the SDR
        """
        seed = cls._hashCoordinate(coordinate)
        rng = Random(seed)
        return rng.getUInt32(n)


    def __str__(self):
        string = "CoordinateEncoder:"
        string += "\n  w: {w}".format(w=self.w)
        string += "\n  n: {n}".format(n=self.n)
        return string

