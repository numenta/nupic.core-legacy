# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from htm.bindings.algorithms import Connections


def main(connections, show=True):
    print("")
    print( connections )
    segmentsPerCell( connections, show=False)
    potentialSynapsesPerSegment( connections, show=False)
    connectedSynapsesPerSegment( connections, show=False)
    permanences( connections, show=False)
    if show:
        plt.show()


def segmentsPerCell(connections, show=True):
    # Histogram of segments per cell
    data = []
    for cell in range( connections.numCells() ):
        datum = len( connections.segmentsForCell(cell) )
        data.append(datum)
    plt.figure("Histogram of Segments per Cell")
    plt.hist( data )
    plt.title("Histogram of Segments per Cell")
    plt.ylabel("Number of Cells")
    plt.xlabel("Number of Segments")
    if show:
        plt.show()


def potentialSynapsesPerSegment(connections, show=True):
    # Histogram of synapses per segment
    data = []
    for cell in range( connections.numCells() ):
        for seg in connections.segmentsForCell(cell):
            datum = len( connections.synapsesForSegment(seg) )
            data.append(datum)
    plt.figure("Histogram of Potential Synapses per Segment")
    plt.hist( data, bins = 50 )
    plt.title("Histogram of Potential Synapses per Segment")
    plt.ylabel("Number of Segments")
    plt.xlabel("Number of Potential Synapses")
    if show:
        plt.show()


def connectedSynapsesPerSegment(connections, show=True):
    # Histogram of synapses per segment
    data = []
    for cell in range( connections.numCells() ):
        for seg in connections.segmentsForCell(cell):
            datum = connections.numConnectedSynapses(seg)
            data.append(datum)
    plt.figure("Histogram of Connected Synapses per Segment")
    plt.hist( data, bins = 50 )
    plt.title("Histogram of Connected Synapses per Segment")
    plt.ylabel("Number of Segments")
    plt.xlabel("Number of Connected Synapses")
    if show:
        plt.show()


def permanences(connections, show=True):
    # Histogram of synapse permanences
    # Draw vertical line at the connected synapse permanece threshold
    data = []
    for cell in range( connections.numCells() ):
        for seg in connections.segmentsForCell(cell):
            for syn in connections.synapsesForSegment(seg):
                datum = connections.permanenceForSynapse(syn)
                data.append(datum)
    plt.figure("Histogram of Synapse Permanences")
    plt.hist( data, bins = 100, range = (0, 1) )
    plt.axvline( connections.connectedThreshold, color='red' )
    plt.title("Histogram of Synapse Permanences")
    plt.ylabel("Number of Synapses")
    plt.xlabel("Permanence of Synapse")
    if show:
        plt.show()


if __name__ == '__main__':
    import argparse
    # Accept location of data dump from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('connections_save_file')
    args = parser.parse_args()

    # Load and run the connections object.
    with open(args.connections_save_file, 'rb') as data_file:
        data = data_file.read()

    # TODO: Try both pickle and load, use whichever works.
    C = Connections.load( data )
    main(C)
