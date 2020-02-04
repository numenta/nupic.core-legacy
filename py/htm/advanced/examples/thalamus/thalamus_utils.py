# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Numenta, Inc. 
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

"""

Utility functions

"""



import numpy as np
from htm.bindings.sdr import SDR

# from nupic.encoders.base import defaultDtype
from htm.encoders.coordinate import CoordinateEncoder


def createLocationEncoder(thalamus, w=15):
    """
    A default coordinate encoder for encoding locations into sparse
    distributed representations.
    """
    encoder = CoordinateEncoder(name="positionEncoder", n=thalamus.l6CellCount, w=w)
    return encoder


def encodeLocation(encoder, x, y, output, radius=5):
    # Radius of 7 or 8 gives an overlap of about 8 or 9 with neighoring pixels
    # Radius of 5 about 3
    encoder.encode((np.array([x * radius, y * radius]), radius), output)
    return output


def getUnionLocations(encoder, x, y, r, step=1):
    """
    Return a union of location encodings that correspond to the union of all locations
    within the specified circle.
    """
    output = SDR(encoder.getWidth())
    locations = set()
    for dx in range(-r, r+1, step):
        for dy in range(-r, r+1, step):
            if dx*dx + dy*dy <= r*r:
                e = encodeLocation(encoder, x+dx, y+dy, output)
                locations = locations.union(set(e.sparse))

    output.sparse = list(locations)
    return output
    
def trainThalamusLocations(thalamus, encoder):
    print("Training TRN cells on location SDRs")
    output = SDR(encoder.getWidth())

    # Train the TRN cells to respond to SDRs representing locations
    for y in range(0, thalamus.trnHeight):
        for x in range(0, thalamus.trnWidth):                
            thalamus.learnL6Pattern(encodeLocation(encoder, x, y, output), [(x, y)])


def trainThalamusLocationsTMP(thalamus, encoder, windowSize=5):
    print("Training TRN cells on location SDRs")
    output = SDR(encoder.getWidth())

    # Train the TRN cells to respond to SDRs representing locations
    for wy in range(0, thalamus.trnHeight):
        print(wy)
        for wx in range(0, thalamus.trnWidth):
            e = encodeLocation(encoder, wx, wy, output)
            for x in range(wx-windowSize, wx+windowSize):
                for y in range(wy - windowSize, wy + windowSize):
                    if x >= 0 and x < thalamus.trnWidth and y >= 0 and y < thalamus.trnHeight:
                        thalamus.learnL6Pattern(e, [(x, y)])

    
