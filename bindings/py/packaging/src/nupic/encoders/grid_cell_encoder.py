# Written by David McDougall, 2018
# TODO: LICENSE

import numpy as np
import math
import random # TODO: Use nupic's random
from nupic.encoders import hexy
from nupic.bindings.sdr import SDR

class GridCellEncoder:
    """ TODO: DOCUMENTATION """
    def __init__(self,
            size     = 100,
            sparsity = .25,
            periods  = [6 * (2**.5)**i for i in range(5)], ):
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
        self.offsets_  = np.random.uniform(0, max(self.periods)*9, size=(self.size, 2))
        self.angles_   = []
        self.rot_mats_ = []
        for period in self.periods:
            angle = random.random() * 2 * math.pi
            self.angles_.append(angle)
            c, s = math.cos(angle), math.sin(angle)
            R    = np.array(((c,-s), (s, c)))
            self.rot_mats_.append(R)

    def reset(self):
        """ Does nothing, GridCellEncoder holds no state. """
        pass

    def encode(self, location):
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
        grid_cells = SDR((self.size,))
        grid_cells.sparse = index
        return grid_cells

if __name__ == '__main__':
    """ TODO: DOCUMENTATION """
    # TODO: Move this to a new file in examples, for better visiblility!
    # TODO: Arguments for periods, sparsity, arena_size
    gc = GridCellEncoder()
    print('Module Periods', gc.periods)

    sz = 100
    rf = np.empty((gc.size, sz, sz))
    for x in range(sz):
        for y in range(sz):
            r = gc.encode([x, y])
            rf[:, x, y] = r.dense.ravel()

    rows       = 5
    cols       = 6
    n_subplots = rows * cols
    samples    = np.arange( 0, gc.size, int(gc.size / n_subplots))
    import matplotlib.pyplot as plt
    plt.figure('Grid Cell Receptive Fields')
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            plt.subplot(rows, cols, i + 1)
            plt.imshow(rf[samples[i]], interpolation='nearest')
    plt.show()
