# ------------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
#
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
#
# http://numenta.org/licenses/
# ------------------------------------------------------------------------------


import numpy as np
import math
from nupic.encoders import hexy
from nupic.bindings.sdr import SDR
from nupic.bindings.math import Random

class GridCellEncoder:
    """ TODO: DOCUMENTATION """
    def __init__(self,
            size,
            sparsity,
            periods,
            seed = 0,):
        """ TODO: DOCUMENTATION """
        self.size       = size
        self.dimensions = (size,)
        self.sparsity   = sparsity
        self.periods    = tuple(sorted(periods))
        assert(min(self.periods) >= 4)

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
        # Find the distance from the location to each grid cells nearest
        # receptive field center.
        # Convert the units of location to hex grid with angle 0, scale 1, offset 0.
        assert(len(location) == 2)
        displacement = list(location) - self.offsets_
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
        if grid_cells is None:
            grid_cells = SDR((self.size,))
        grid_cells.sparse = index
        return grid_cells


if __name__ == '__main__':
    import argparse
    from nupic.bindings.sdr import Metrics

    parser = argparse.ArgumentParser(description=GridCellEncoder.__doc__)
    parser.add_argument('--arena_size', type=int, default=100,
                        help='')
    parser.add_argument('--sparsity', type=float, default=.25,
                        help='')
    parser.add_argument('--periods', type=list,
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

    rf = np.empty((gc.size, args.arena_size, args.arena_size))
    for x in range(args.arena_size):
        for y in range(args.arena_size):
            gc.encode([x, y], gc_sdr)
            rf[:, x, y] = gc_sdr.dense.ravel()

    print(gc_statistics)

    rows       = 5
    cols       = 6
    n_subplots = rows * cols
    assert(gc.size > n_subplots)
    samples    = np.linspace( 0, gc.size-1, n_subplots, dtype=np.int )
    import matplotlib.pyplot as plt
    plt.figure('Grid Cell Receptive Fields')
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            plt.subplot(rows, cols, i + 1)
            plt.imshow(rf[samples[i]], interpolation='nearest')
    plt.show()
