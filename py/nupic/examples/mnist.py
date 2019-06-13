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
# ------------------------------------------------------------------------------
""" An MNIST classifier using Spatial Pooler."""

import argparse
import random
import gzip
import numpy as np
import os

from nupic.bindings.algorithms import SpatialPooler, Classifier
from nupic.bindings.sdr import SDR, Metrics


def load_mnist(path):
    """See: http://yann.lecun.com/exdb/mnist/ for MNIST download and binary file format spec."""
    def int32(b):
        i = 0
        for char in b:
            i *= 256
            # i += ord(char)    # python2
            i += char
        return i

    def load_labels(file_name):
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2049)  # Magic number
            labels = []
            for char in raw[8:]:
                # labels.append(ord(char))      # python2
                labels.append(char)
        return labels

    def load_images(file_name):
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2051)    # Magic number
            num_imgs   = int32(raw[4:8])
            rows       = int32(raw[8:12])
            cols       = int32(raw[12:16])
            assert(rows == 28)
            assert(cols == 28)
            img_size   = rows*cols
            data_start = 4*4
            imgs = []
            for img_index in range(num_imgs):
                vec = raw[data_start + img_index*img_size : data_start + (img_index+1)*img_size]
                # vec = [ord(c) for c in vec]   # python2
                vec = list(vec)
                vec = np.array(vec, dtype=np.uint8)
                vec = np.reshape(vec, (rows, cols))
                imgs.append(vec)
            assert(len(raw) == data_start + img_size * num_imgs)   # All data should be used.
        return imgs

    train_labels = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
    train_images = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))
    test_labels  = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))
    test_images  = load_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))

    return train_labels, train_images, test_labels, test_images


default_parameters = {
    'boostStrength': 7.80643753517375,
    'columnDimensions': (35415,1),
    'dutyCyclePeriod': 1321,
    'localAreaDensity': 0.05361688506086096,
    'minPctOverlapDutyCycle': 0.0016316043362658,
    'potentialPct': 0.06799785776775163,
    'stimulusThreshold': 8,
    'synPermActiveInc': 0.01455789388651146,
    'synPermConnected': 0.021649964738697944,
    'synPermInactiveDec': 0.006442691852205935
}


def main(parameters=default_parameters, argv=None, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
        default = os.path.join( os.path.dirname(__file__), 'MNIST_data'))
    args = parser.parse_args(args = argv)

    # Load data.
    train_labels, train_images, test_labels, test_images = load_mnist(args.data_dir)
    training_data = list(zip(train_images, train_labels))
    test_data     = list(zip(test_images, test_labels))
    random.shuffle(training_data)
    random.shuffle(test_data)

    # Setup the AI.
    enc = SDR((train_images[0].shape))
    sp = SpatialPooler(
        inputDimensions            = enc.dimensions,
        columnDimensions           = parameters['columnDimensions'],
        potentialRadius            = 99999999,
        potentialPct               = parameters['potentialPct'],
        globalInhibition           = True,
        localAreaDensity           = parameters['localAreaDensity'],
        numActiveColumnsPerInhArea = -1,
        stimulusThreshold          = int(round(parameters['stimulusThreshold'])),
        synPermInactiveDec         = parameters['synPermInactiveDec'],
        synPermActiveInc           = parameters['synPermActiveInc'],
        synPermConnected           = parameters['synPermConnected'],
        minPctOverlapDutyCycle     = parameters['minPctOverlapDutyCycle'],
        dutyCyclePeriod            = int(round(parameters['dutyCyclePeriod'])),
        boostStrength              = parameters['boostStrength'],
        seed                       = 0,
        spVerbosity                = 99,
        wrapAround                 = False)
    columns = SDR( sp.getColumnDimensions() )
    columns_stats = Metrics( columns, 99999999 )
    sdrc = Classifier()

    # Training Loop
    for i in range(len(train_images)):
        img, lbl = random.choice(training_data)
        enc.dense = img >= np.mean(img) # Convert greyscale image to binary.
        sp.compute( enc, True, columns )
        sdrc.learn( columns, lbl )

    print(str(sp))
    print(str(columns_stats))

    # Testing Loop
    score = 0
    for img, lbl in test_data:
        enc.dense = img >= np.mean(img) # Convert greyscale image to binary.
        sp.compute( enc, False, columns )
        if lbl == np.argmax( sdrc.infer( columns ) ):
            score += 1

    print('Score:', 100 * score / len(test_data), '%')
    return score / len(test_data)


if __name__ == '__main__':
    main()
