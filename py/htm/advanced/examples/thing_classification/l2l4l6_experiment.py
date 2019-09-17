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
    This script creates simple experiment to compute the object classification
    accuracy of L2-L4-L6 network using objects from YCB dataset and "Thing" sensor
"""
import glob
import json
import logging
import os
import random
from collections import defaultdict, OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

from htm.advanced.frameworks.location.location_network_creation import L246aNetwork
from htm.advanced.support.expsuite import PyExperimentSuite
from htm.advanced.support.register_regions import registerAllAdvancedRegions

logging.basicConfig(level=logging.ERROR)



def loadThingData(dataDir="data", n=150, w=11):
    """
    Load Thing sensation data. There is one file per object, each row contains one
    feature, location pairs. The format is as follows:
        [(-33.6705, 75.5003, 2.4207)/10] => [[list of active bits of location],
                                                                                 [list of active bits of feature]]
    The content before "=>" is the true 3D location / sensation
    We ignore the encoded values after "=>" and use :class:`ScalarEncoder` to
    encode the sensation in a way that is compatible with the experiment network.

    :param dataDir: The location data files
    :type dataDir: str
    :param n: The number of bits in the feature SDR. Usually L4 column count
    :type n: int
    :param w: Number of 'on' bits in the feature SDR. Usually L4 sample size
    :type w: int
    :return: Dictionary mapping objects to sensations that can be used directly by
                     class L246aNetwork 'infer' and 'learn' methods
    :rtype: dict[str,list]
    """
    objects = defaultdict(list)

    # Thing features are scalar values ranging from 1-25 inclusive
    p = ScalarEncoderParameters()
    p.size = n
    p.activeBits = w
    p.minimum = 1
    p.maximum = 25
    encoder = ScalarEncoder(p)

    dataPath = os.path.dirname(os.path.realpath(__file__))
    dataPath = os.path.join(dataPath, dataDir)
    objFiles = glob.glob1(dataPath, "*.log")

    for filename in objFiles:
        obj, _ = os.path.splitext(filename)

        # Read raw sensations from log file. Ignore SDRs after "=>"
        sensations = []
        with open(os.path.join(dataPath, filename)) as f:
            for line in f.readlines():
                # Parse raw location/feature values
                line = line.split("=>")[0]
                line = ''.join(i for i in line if not i in "[,]()")
                locationStr, featureStr = line.split("/")
                location = list(map(float, locationStr.split()))
                feature = list(encoder.encode(int(featureStr)).sparse)

                sensations.append((location, feature))

        # Assume single column
        objects[obj] = [sensations]

    return objects



class L2L4L6aExperiment(PyExperimentSuite):
    """
    Compute the object classification accuracy of L2-L4-L6 network using objects
    from YCB dataset and a single column "Thing" sensor.
    """

    def reset(self, params, repetition):
        """
        Take the steps necessary to reset the experiment before each repetition:
            - Make sure random seed is different for each repetition
            - Create the L2-L4-L6a network
            - Load objects used by the experiment
            - Learn all objects used by the experiment
        """
        print(params["name"], ":", repetition)
        self.debug = params.get("debug", False)
        self.numLearningPoints = params["num_learning_points"]
        self.numOfSensations = params["num_sensations"]

        L2Params = json.loads('{' + params["l2_params"] + '}')
        L4Params = json.loads('{' + params["l4_params"] + '}')
        L6aParams = json.loads('{' + params["l6a_params"] + '}')

        self.sdrSize = L2Params["sdrSize"]

        # Make sure random seed is different for each repetition
        seed = params.get("seed", 42)
        np.random.seed(seed + repetition)
        random.seed(seed + repetition)
        L2Params["seed"] = seed + repetition
        L4Params["seed"] = seed + repetition
        L6aParams["seed"] = seed + repetition

        # Configure L6a params
        numModules = params["num_modules"]
        L6aParams["scale"] = [params["scale"]] * numModules
        angle = params["angle"] // numModules
        orientation = list(range(angle // 2, angle * numModules, angle))
        L6aParams["orientation"] = np.radians(orientation).tolist()
        L6aParams["cellsPerAxis"] = params["cells_per_axis"]

        # Create single column L2-L4-L6a network
        self.network = L246aNetwork(numColumns=1, 
                                    L2Params=L2Params,
                                    L4Params=L4Params, 
                                    L6aParams=L6aParams,
                                    repeat=self.numLearningPoints,
                                    logCalls=self.debug)

        # Load Thing Objects
        sampleSize = L4Params["sampleSize"]
        columnCount = L4Params["columnCount"]

        # Make sure w is odd per encoder requirement
        sampleSize = sampleSize if sampleSize % 2 != 0 else sampleSize + 1

        self.objects = loadThingData(dataDir=params["data_path"], w=sampleSize, n=columnCount)

        # Number of iterations must match the number of objects. This will allow us
        # to execute one iteration per object and use the "iteration" parameter as
        # the object index
        assert params["iterations"] == len(self.objects)

        # Learn objects
        self.network.learn(self.objects)

    def iterate(self, _, repetition, iteration):
        """
        For each iteration try to infer the object represented by the 'iteration'
        parameter returning Whether or not the object was unambiguously classified.
        :param params: Specific parameters for this iteration. See 'experiments.cfg'
                                     for list of parameters
        :param repetition: Current repetition
        :param iteration: Use the iteration to select the object to infer
        :return: Whether or not the object was classified
        """
        objname, sensations = list(self.objects.items())[iteration]

        # Select sensations to infer
        np.random.shuffle(sensations[0])
        sensations = [sensations[0][:self.numOfSensations]]

        self.network.sendReset()

        # Collect all statistics for every inference.
        # See L246aNetwork._updateInferenceStats
        stats = defaultdict(list)
        self.network.infer(sensations=sensations, stats=stats, objname=objname)
        stats.update({"name": objname})
        return stats



def plotAccuracy(suite, name):
    """
    Plots classification accuracy
    """
    path = suite.cfgparser.get(name, "path")
    path = os.path.join(path, name)

    accuracy = defaultdict(list)
    sensations = defaultdict(list)

    for exp in suite.get_exps(path=path):
        params = suite.get_params(exp)
        maxTouches = params["num_sensations"]
        cells = params["cells_per_axis"]

        res = suite.get_history(exp, 0, "Correct classification")
        classified = [any(x) for x in res]
        accuracy[cells] = float(sum(classified)) / float(len(classified))
        touches = [np.argmax(x) or maxTouches for x in res]
        sensations[cells] = [np.mean(touches), np.max(touches)]

    plt.title("Classification Accuracy")

    accuracy = OrderedDict(sorted(list(accuracy.items()), key=lambda t: t[0]))
    _, ax1 = plt.subplots()
    ax1.plot(list(accuracy.keys()), list(accuracy.values()), "b")
    ax1.set_xlabel("Cells per axis")
    ax1.set_ylabel("Accuracy", color="b")
    ax1.tick_params("y", colors="b")

    sensations = OrderedDict(sorted(list(sensations.items()), key=lambda t: t[0]))
    ax2 = ax1.twinx()
    ax2.set_prop_cycle(linestyle=["-", "--"])
    ax2.plot(list(sensations.keys()), list(sensations.values()), "r")
    ax2.set_ylabel("Sensations", color="r")
    ax2.tick_params("y", colors="r")
    ax2.legend(("Mean", "Max"))

    # save
    path = suite.cfgparser.get(name, "path")
    plotPath = os.path.join(path, "{}.pdf".format(name))
    plt.savefig(plotPath)
    plt.close()



if __name__ == "__main__":
    registerAllAdvancedRegions()

    suite = L2L4L6aExperiment()
    suite.start()

    experiments = suite.options.experiments
    if experiments is None:
        experiments = suite.cfgparser.sections()

    for exp in experiments:
        plotAccuracy(suite, exp)
