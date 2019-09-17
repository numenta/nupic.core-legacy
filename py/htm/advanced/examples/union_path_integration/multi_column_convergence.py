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
This file computes number of observations needed to unambiguously recognize an
object with multi-column L2-L4-L6a networks as the number of columns increases.
"""
import collections
import json
import os
import random

import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htm.advanced.frameworks.location.location_network_creation import L246aNetwork
from htm.advanced.support.object_generation import generateObjects
from htm.advanced.support.expsuite import PyExperimentSuite
from htm.advanced.support.register_regions import registerAllAdvancedRegions


class MultiColumnExperiment(PyExperimentSuite):
    """
    Number of observations needed to unambiguously recognize an object with
    multi-column networks as the set of columns increases. We train each network
    with a set number of objects and plot the average number of sensations
    required to unambiguously recognize an object.
    """

    def reset(self, params, repetition):
        """
        Take the steps necessary to reset the experiment before each repetition:
            - Make sure random seed is different for each repetition
            - Create the L2-L4-L6a network
            - Generate objects used by the experiment
            - Learn all objects used by the experiment
        """
        print(params["name"], ":", repetition)

        self.debug = params.get("debug", False)

        L2Params = json.loads('{' + params["l2_params"] + '}')
        L4Params = json.loads('{' + params["l4_params"] + '}')
        L6aParams = json.loads('{' + params["l6a_params"] + '}')

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

        # Create multi-column L2-L4-L6a network
        self.numColumns = params["num_cortical_columns"]
        self.network = L246aNetwork(numColumns=self.numColumns, 
                                    L2Params=L2Params,
                                    L4Params=L4Params, 
                                    L6aParams=L6aParams,
                                    repeat=0,
                                    logCalls=self.debug)

        # Use the number of iterations as the number of objects. This will allow us
        # to execute one iteration per object and use the "iteration" parameter as
        # the object index
        numObjects = params["iterations"]

        # Generate feature SDRs
        numFeatures = params["num_features"]
        numOfMinicolumns = L4Params["columnCount"]
        numOfActiveMinicolumns = params["num_active_minicolumns"]
        self.featureSDR = [{
            str(f): sorted(np.random.choice(numOfMinicolumns, numOfActiveMinicolumns))
            for f in range(numFeatures)
        } for _ in range(self.numColumns)]

        # Generate objects used in the experiment
        self.objects = generateObjects(numObjects=numObjects,
                                         featuresPerObject=params["features_per_object"],
                                         objectWidth=params["object_width"],
                                         numFeatures=numFeatures,
                                         distribution=params["feature_distribution"])

        self.sdrSize = L2Params["sdrSize"]

        # Learn objects
        self.numLearningPoints = params["num_learning_points"]
        self.numOfSensations = params["num_sensations"]
        self.learnedObjects = {}
        self.learn()

    def iterate(self, params, repetition, iteration):
        """
        For each iteration try to infer the object represented by the 'iteration'
        parameter returning the number of touches required to unambiguously
        classify the object.
        :param params: Specific parameters for this iteration. See 'experiments.cfg'
                                     for list of parameters
        :param repetition: Current repetition
        :param iteration: Use the iteration to select the object to infer
        :return: number of touches required to unambiguously classify the object
        """
        objectToInfer = self.objects[iteration]
        stats = collections.defaultdict(list)
        touches = self.infer(objectToInfer, stats)
        results = {'touches': touches}
        results.update(stats)

        return results

    def setLearning(self, learn):
        """
        Set all regions in every column into the given learning mode
        """
        self.network.setLearning(learn)


    def sendReset(self):
        """
        Sends a reset signal to all regions in the network.
        It should be called before changing objects.
        """
        self.network.sendReset()

    def learn(self):
        """
        Learn all objects on every column. Each column will learn all the features
        of every object and store the the object's L2 representation to be later
        used in the inference stage
        """
        self.setLearning(True)

        for obj in self.objects:
            self.sendReset()

            previousLocation = [None] * self.numColumns
            displacement = [0., 0.]
            features = obj["features"]
            numOfFeatures = len(features)

            # Randomize touch sequences
            touchSequence = np.random.permutation(numOfFeatures)

            for sensation in range(numOfFeatures):
                for col in range(self.numColumns):
                    # Shift the touch sequence for each column
                    colSequence = np.roll(touchSequence, col)
                    feature = features[colSequence[sensation]]
                    # Move the sensor to the center of the object
                    locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
                    # Calculate displacement from previous location
                    if previousLocation[col] is not None:
                        displacement = locationOnObject - previousLocation[col]
                    previousLocation[col] = locationOnObject

                    # learn each pattern multiple times
                    activeColumns = self.featureSDR[col][feature["name"]]
                    for _ in range(self.numLearningPoints):
                        # Sense feature at location
                        self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
                        self.network.sensorInput[col].executeCommand('addDataToQueue', activeColumns, False, 0)
                        # Only move to the location on the first sensation.
                        displacement = [0, 0]

            self.network.network.run(numOfFeatures * self.numLearningPoints)

            # update L2 representations for the object
            self.learnedObjects[obj["name"]] = self.getL2Representations()

    def infer(self, objectToInfer, stats=None):
        """
        Attempt to recognize the specified object with the network. Randomly move
        the sensor over the object until the object is recognized.
        """
        self.setLearning(False)
        self.sendReset()

        touches = None
        previousLocation = [None] * self.numColumns
        displacement = [0., 0.]
        features = objectToInfer["features"]
        objName = objectToInfer["name"]
        numOfFeatures = len(features)

        # Randomize touch sequences
        touchSequence = np.random.permutation(numOfFeatures)

        for sensation in range(self.numOfSensations):
            # Add sensation for all columns at once
            for col in range(self.numColumns):
                # Shift the touch sequence for each column
                colSequence = np.roll(touchSequence, col)
                feature = features[colSequence[sensation]]
                # Move the sensor to the center of the object
                locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
                # Calculate displacement from previous location
                if previousLocation[col] is not None:
                    displacement = locationOnObject - previousLocation[col]
                previousLocation[col] = locationOnObject

                # Sense feature at location
                self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
                self.network.sensorInput[col].executeCommand('addDataToQueue', self.featureSDR[col][feature["name"]], False, 0)
            self.network.network.run(1)
            if self.debug:
                self.network.updateInferenceStats(stats, objectName=objName)

            if touches is None and self.network.isObjectClassified(objName, minOverlap=30):
                touches = sensation + 1
                if not self.debug:
                    return touches

        return self.numOfSensations if touches is None else touches

    def getL2Representations(self):
        """
        Returns the active representation in L2.
        """
        return self.network.getL2Representations()

def plotSensationByColumn(suite, name):
    """
    Plots the convergence graph: touches by columns.
    """
    path = suite.cfgparser.get(name, "path")
    path = os.path.join(path, name)

    touches = {}
    for exp in suite.get_exps(path=path):
        params = suite.get_params(exp)
        cols = params["num_cortical_columns"]
        features = params["num_features"]
        if not features in touches:
            touches[features] = {}

        histories = suite.get_histories_over_repetitions(exp, "touches", np.mean)
        touches[features][cols] = np.mean(histories)

    ax = plt.figure(tight_layout={"pad": 0}).gca()
    colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
    for i, features in enumerate(sorted(touches)):
        cols = touches[features]
        plt.plot(list(cols.keys()), list(cols.values()), "-", label="Unique features={}".format(features), color=colorList[i])

    # format
    plt.xlabel("Number of columns")
    plt.ylabel("Average number of touches")
    plt.title("Number of touches to recognize one object (multiple columns)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(framealpha=1.0)

    # save
    path = suite.cfgparser.get(name, "path")
    plotPath = os.path.join(path, "{}.pdf".format(name))
    plt.savefig(plotPath)
    plt.close()



def plotDebugStatistics(suite, name):
    path = suite.cfgparser.get(name, "path")
    path = os.path.join(path, name)
    for exp in suite.get_exps(path=path):
        params = suite.get_params(exp)
        if not params["debug"]:
            continue

        cols = params["num_cortical_columns"]
        features = params["num_features"]
        L2Params = json.loads('{' + params["l2_params"] + '}')
        cellCount = L2Params["cellCount"]

        # Multi column metrics. See _updateInferenceStats
        metrics = ["L2 Representation",
                 "Overlap L2 with object",
                 "L6a Representation",
                 "L6a LearnableCells",
                 "L6a SensoryAssociatedCells",
                 "L4 Representation",
                 "L4 Predicted"]

        keys = []
        for col in range(cols):
            keys.extend(["{} C{}".format(metric, col) for metric in metrics])
            keys.append("Full L2 SDR C{}".format(col))

        # Just Plot the first repetition
        history = suite.get_history(exp, 0, keys)

        # Plot metrics
        for metric in metrics:
            ax = plt.figure().gca()
            for c in range(cols):
                key = "{} C{}".format(metric, c)
                data = history[key]
                if not None in data:
                    mean_data = np.mean(data, axis=0)
                    plt.plot(range(1, len(mean_data) + 1), mean_data, label=key)

            # format
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel("Number of sensations")
            plt.ylabel(metric)
            plt.title("{} by sensation ({} features, {} columns)".format(metric, features, cols))

            # save
            plotPath = os.path.join(path, "{}_{}_{}.pdf".format(metric, features, cols))
            plt.savefig(plotPath, tight_layout={"pad": 0})
            plt.close()

        # Plot L2 SDR
        for c in range(cols):
            fig = plt.figure()
            data = history["Full L2 SDR C{}".format(c)]

            # One SDR per object for every touch
            for obj, touches in enumerate(data):
                ax = fig.add_axes([0.1 + 0.015 * obj, 0.1, 0.01, .8], frameon=False, xticks=[], yticks=[])
                values = []
                for sdr in touches:
                    bits = np.zeros(cellCount)
                    bits[sdr] = 1
                    values.append(bits)

                ax.imshow(np.array(values).T, aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
            # format
            plt.title("L2 SDR for col {} ({} features, {} columns)".format(c, features, cols), loc='right')
            # save
            plotPath = os.path.join(path, "Full L2 SDR_{}_{}_{}.pdf".format(features, cols, c))
            plt.savefig(plotPath, tight_layout={"pad": 0})
            plt.close()



if __name__ == "__main__":
    registerAllAdvancedRegions()

    suite = MultiColumnExperiment()
    suite.start()

    experiments = suite.options.experiments
    if experiments is None:
        experiments = suite.cfgparser.sections()

    for exp in experiments:
        plotSensationByColumn(suite, exp)
        plotDebugStatistics(suite, exp)
