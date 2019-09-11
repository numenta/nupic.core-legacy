# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc. 
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
Functions for generating objects intended for the union_path_integration
project.
"""

import math
import random

import numpy as np


def getRandomFeature(probabilityByFeature):
    # Each feature owns an interval in the range [0, 1]. Generate a random
    # floating point index and find which feature's interval it lands within.
    selectedIndex = np.random.random()

    intervalEnd = probabilityByFeature[0]
    for i in range(1, len(probabilityByFeature)):
        if selectedIndex <= intervalEnd:
            return i - 1
        intervalEnd += probabilityByFeature[i]

    return len(probabilityByFeature) - 1


def arrangeFeatures(allObjectFeatures, objectWidth):
    featureScale = 20

    objects = []
    for o, objectFeatures in enumerate(allObjectFeatures):
        positions = random.sample(range(objectWidth**2), len(objectFeatures))
        objects.append({
            "features": [{"left": (pos % objectWidth)*featureScale,
                                    "top": int(pos / objectWidth)*featureScale,
                                    "width": featureScale,
                                    "height": featureScale,
                                    "name": str(feat)}
                                for pos, feat in zip(positions, objectFeatures)],
            "name": str(o)})

    return objects


def generateObjectFeatures(numObjects, featuresPerObject, numFeatures, distribution):
    if distribution == "AllFeaturesEqual_Replacement":
        return np.random.randint(numFeatures, size=(numObjects, featuresPerObject), dtype=np.int32)
    elif distribution == "AllFeaturesEqual_NoReplacement":
        totalNumLocations = numObjects * featuresPerObject
        numFeatureRepetitions = int(math.ceil(totalNumLocations / float(numFeatures)))
        allFeatures = np.tile(np.arange(numFeatures), numFeatureRepetitions)
        allFeatures = allFeatures[:totalNumLocations]
        np.random.shuffle(allFeatures)

        return allFeatures.reshape(numObjects, featuresPerObject)
    elif distribution == "TwoPools_Replacement":
        probabilityByFeature = np.ones(numFeatures, dtype="float")
        probabilityByFeature[numFeatures/2:] = 2.5
        probabilityByFeature /= np.sum(probabilityByFeature)

        return np.array([[getRandomFeature(probabilityByFeature)
                                            for _ in range(featuresPerObject)]
                                         for _ in range(numObjects)])
    elif distribution == "TwoPools_Structured":
        # Divide numFeatures into two pools. Every object gets one feature from the
        # first pool and every other feature from the second.
        secondBinBegin = int(math.ceil(numFeatures / 2.))
        return np.array([[np.random.randint(secondBinBegin)]
                                         + [np.random.randint(secondBinBegin, numFeatures)
                                            for _ in range(1, featuresPerObject)]
                                         for _ in range(numObjects)])
    elif distribution == "Random":
        probabilityByFeature = np.random.random(numFeatures)
        probabilityByFeature /= np.sum(probabilityByFeature)

        return np.array([[getRandomFeature(probabilityByFeature)
                                            for _ in range(featuresPerObject)]
                                         for _ in range(numObjects)])
    else:
        raise ValueError("Unknown distribution", distribution)



def generateObjects(numObjects, featuresPerObject, objectWidth, numFeatures, distribution="AllFeaturesEqual_Replacement"):
    assert featuresPerObject <= (objectWidth ** 2)

    objectFeatures = generateObjectFeatures(numObjects, featuresPerObject, numFeatures, distribution)

    return arrangeFeatures(objectFeatures, objectWidth)
