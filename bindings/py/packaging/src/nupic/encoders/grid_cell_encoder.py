# Written by David McDougall, 2018

import numpy as np
import math
import random
from nupic.encoders import hexy
from nupic.bindings.sdr import SDR

class GridCellEncoder:
    """ TODO: DOCUMENTATION """
    def __init__(self,
            size           = 100,
            sparsity       = .25,
            module_periods = [6 * (2**.5)**i for i in range(5)], ):
        """ TODO: DOCUMENTATION """
        assert(min(module_periods) >= 4)
        self.size           = size
        self.sparsity       = sparsity
        self.module_periods = sorted(module_periods)
        self.offsets = np.random.uniform(0, max(self.module_periods)*9, size=(self.size, 2))
        module_partitions      = np.linspace(0, self.size, num=len(self.module_periods) + 1)
        module_partitions      = list(zip(module_partitions, module_partitions[1:]))
        self.module_partitions = [(int(round(start)), int(round(stop)))
                                        for start, stop in module_partitions]
        self.scales   = []
        self.angles   = []
        self.rot_mats = []
        for period in self.module_periods:
            self.scales.append(period)
            angle = random.random() * 2 * math.pi
            self.angles.append(angle)
            c, s = math.cos(angle), math.sin(angle)
            R    = np.array(((c,-s), (s, c)))
            self.rot_mats.append(R)
        self.reset()

    def reset(self):
        pass

    def encode(self, location):
        # Find the distance from the location to each grid cells nearest
        # receptive field center.
        # Convert the units of location to hex grid with angle 0, scale 1, offset 0.
        assert(len(location) == 2)
        displacement = list(location) - self.offsets
        radius       = np.empty(self.size)
        for mod_idx in range(len(self.module_partitions)):
            start, stop = self.module_partitions[mod_idx]
            R           = self.rot_mats[mod_idx]
            displacement[start:stop] = R.dot(displacement[start:stop].T).T
            radius[start:stop] = self.scales[mod_idx] / 2
        # Convert into and out of hexagonal coordinates, which rounds to the
        # nearest hexagons center.
        nearest = hexy.cube_to_pixel(hexy.pixel_to_cube(displacement, radius), radius)
        # Find the distance between the location and the RF center.
        distances = np.hypot(*(nearest - displacement).T)
        # Activate the closest grid cells in each module.
        index = []
        for start, stop in self.module_partitions:
            z = int(round(self.sparsity * (stop - start)))
            index.extend( np.argpartition(distances[start : stop], z)[:z] + start )
        grid_cells = SDR((self.size,))
        grid_cells.sparse = index
        return grid_cells

if __name__ == '__main__':
    # TODO: Arguments for module_periods, sparsity, arena_size
    gc = GridCellEncoder()
    print('Module Periods', gc.module_periods)

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
