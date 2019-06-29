# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2018-2019, David McDougall
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
import math
import hexy

from htm.bindings.sdr import SDR
from htm.bindings.math import Random

class GridCellEncoder:
    """
    This Encoder converts a 2-D coordinate into plausible grid cell activity.
    The output SDR is divided into modules.  Each module is a distinct groups of
    cells with a common grid spacing and orientation.  Different modules have
    different spacings & orientations.

    For example usage and to inspect the output of this encoder run:
    $ python3 -m htm.encoders.grid_cell_encoder
    """
    def __init__(self,
            size,
            sparsity,
            periods,
            seed = 0,):
        """
        Argument size is the total number of bits in the encoded output SDR.

        Argument sparsity is fraction of bits which this encoder activates in
        the output SDR.

        Argument periods is a list of distances.  The period of a module is the
        distance between the centers of a grid cells receptive fields.  The
        length of this list defines the number of distinct modules.

        Argument seed controls the pseudo-random-number-generator which this
        encoder uses.  This encoder produces deterministic output.  The seed
        zero is special, seed zero is replaced with a truly random seed.
        """
        self.size       = size
        self.dimensions = (size,)
        self.sparsity   = sparsity
        self.periods    = tuple(sorted(float(p) for p in periods))
        assert(len(self.periods) > 0)
        assert(min(self.periods) >= 4)
        assert(self.sparsity >= 0)
        assert(self.sparsity <= 1)

        # Assign each module a range of cells in the output SDR.
        partitions       = np.linspace(0, self.size, num=len(self.periods) + 1)
        partitions       = list(zip(partitions, partitions[1:]))
        self.partitions_ = [(int(round(start)), int(round(stop)))
                                        for start, stop in partitions]

        # Assign each module a random offset and orientation.
        rng            = np.random.RandomState(seed = Random(seed).getUInt32())
        self.offsets_  = rng.uniform(0, max(self.periods)*9, size=(self.size, 2))
        self.angles_   = []
        self.rot_mats_ = []
        for period in self.periods:
            angle = rng.uniform() * 2 * math.pi
            self.angles_.append(angle)
            c, s = math.cos(angle), math.sin(angle)
            R    = np.array(((c,-s), (s, c)))
            self.rot_mats_.append(R)

    def reset(self):
        """ Does nothing, GridCellEncoder holds no state. """
        pass

    def encode(self, location, grid_cells=None):
        """
        Transform a 2-D coordinate into an SDR.

        Argument location: pair of coordinates, such as "[X, Y]"

        Argument grid_cells: Optional, the SDR object to store the results in.
                             Its dimensions must be "[GridCellEncoder.size]"

        Returns grid_cells, an SDR object.  This will be created if not given.
        """
        location = list(location)
        assert(len(location) == 2)
        if grid_cells is None:
            grid_cells = SDR((self.size,))
        else:
            assert(isinstance(grid_cells, SDR))
            assert(grid_cells.dimensions == [self.size])
        if any(math.isnan(x) for x in location):
            grid_cells.zero()
            return grid_cells

        # Find the distance from the location to each grid cells nearest
        # receptive field center.
        # Convert the units of location to hex grid with angle 0, scale 1, offset 0.
        displacement = location - self.offsets_
        radius       = np.empty(self.size)
        for mod_idx in range(len(self.partitions_)):
            start, stop = self.partitions_[mod_idx]
            R           = self.rot_mats_[mod_idx]
            displacement[start:stop] = R.dot(displacement[start:stop].T).T
            radius[start:stop] = self.periods[mod_idx] / 2
        # Convert into and out of hexagonal coordinates, which rounds to the
        # nearest hexagons center.
        nearest = hexy.cube_to_pixel(hexy.pixel_to_cube(displacement, radius), radius)
        # Find the distance between the location and the RF center.
        distances = np.hypot(*(nearest - displacement).T)
        # Activate the closest grid cells in each module.
        index = []
        for start, stop in self.partitions_:
            z = int(round(self.sparsity * (stop - start)))
            index.extend( np.argpartition(distances[start : stop], z)[:z] + start )
        grid_cells.sparse = index
        return grid_cells


if __name__ == '__main__':
    import argparse
    from htm.bindings.sdr import Metrics
    import textwrap

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent(GridCellEncoder.__doc__ + "\n\n" +
                                      GridCellEncoder.__init__.__doc__))
    parser.add_argument('--arena_size', type=int, default=100,
                        help='')
    parser.add_argument('--sparsity', type=float, default=.25,
                        help='')
    parser.add_argument('--periods', type=float, nargs='+',
                        default = [6 * (2**.5)**i for i in range(5)],
                        help='')

    args = parser.parse_args()
    print('Module Periods', args.periods)

    gc = GridCellEncoder(
        size     = 100,
        sparsity = args.sparsity,
        periods  = args.periods,)

    gc_sdr = SDR( gc.dimensions )

    gc_statistics = Metrics(gc_sdr, args.arena_size ** 2)

    assert( args.arena_size >= 10 )
    rf = np.empty((gc.size, args.arena_size, args.arena_size))
    for x in range(args.arena_size):
        for y in range(args.arena_size):
            gc.encode([x, y], gc_sdr)
            rf[:, x, y] = gc_sdr.dense.ravel()

    print(gc_statistics)

    rows       = 4
    cols       = 5
    n_subplots = rows * cols
    assert(gc.size > n_subplots)
    samples    = np.linspace( 0, gc.size-1, n_subplots, dtype=np.int )
    import matplotlib.pyplot as plt
    plt.figure('Grid Cell Receptive Fields')
    plt.suptitle("Grid Cell Receptive Fields.\n" +
        "Each figure is a map of a room, showing the areas in the room where a grid cell is active.")
    i = 0
    for row in range(rows):
        for col in range(cols):
            cell_idx = samples[i]
            plt.subplot(rows, cols, i + 1)
            plt.title("Cell #%d"%cell_idx)
            plt.imshow(rf[cell_idx], interpolation='nearest')
            i += 1
    plt.show()
