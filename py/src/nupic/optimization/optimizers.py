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

from nupic.optimization.parameter_set import ParameterSet
import itertools
import random

class BaseOptimizer:
    """
    TODO
    """
    def addArguments(parser):
        """
        TODO
        """
        pass

    def useThisOptimizer(args):
        """
        TODO
        """
        return False

    def __init__(self, labReport, args):
        """
        TODO
        """
        self.lab  = labReport
        self.args = args

    def suggestExperiment(self):
        """
        TODO
        """
        pass

    def collectResults(self, experiment, result):
        """
        TODO
        """
        pass


class EvaluateDefaultParameters(BaseOptimizer):
    def addArguments(parser):
        parser.add_argument('--default_parameters', action='store_true',)

    def useThisOptimizer(args):
        return args.default_parameters

    def suggestExperiment(self):
        return self.lab.default_parameters


class EvaluateAllExperiments(BaseOptimizer):
    def addArguments(parser):
        parser.add_argument('--all_experiments', action='store_true',
            help='Evaluate all experiments in the lab report, don\'t start new experiments')

    def useThisOptimizer(args):
        return args.all_experiments

    def suggestExperiment(self):
        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(self.lab.experiments, key=lambda x: x.attempts + rnd()).parameters


class EvaluateBestExperiment(BaseOptimizer):
    def addArguments(parser):
        parser.add_argument('--best', action='store_true',
            help='Evaluate the best set of parameters on file.')

    def useThisOptimizer(args):
        return args.best

    def suggestExperiment(self):
        best = max(self.lab.experiments, key = lambda X: X.mean() )
        return best.parameters


class EvaluateHashes(BaseOptimizer):
    def addArguments(parser):
        parser.add_argument('--hashes', type=str,)

    def useThisOptimizer(args):
        return args.hashes

    def __init__(self, labReport, args):
        hashes = [int(h, base=16) for h in args.hashes.split(',')]
        try:
            self.experiments = [lab.experiment_ids[h] for h in hashes]
        except KeyError:
            unknown = [h for h in hashes if h not in lab.experiment_ids]
            raise ValueError('Hash not recognized: %X'%unknown[0])

    def suggestExperiment(self):
        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(self.experiments, key=lambda x: x.attempts + rnd()).parameters


class GridSearch(BaseOptimizer):
    # TODO: Make these into a CLI argument?
    mod_funcs = [
        lambda v: v *  .40,
        lambda v: v *  .60,
        lambda v: v *  .80,
        lambda v: v * 1.20,
        lambda v: v * 1.40,
        lambda v: v * 1.60,
    ]

    def addArguments(parser):
        parser.add_argument('--grid_search', type=str,
            help="TODO CLI argument help for GridSearch")

    def useThisOptimizer(args):
        return args.grid_search

    def __init__(self, labReport, args):
        self.lab = labReport

        # Get a list of every parameter to experiment with.
        target_parameters = []
        for start in args.grid_search.split(','):
            node = eval("lab.default_parameters" + start)
            target_parameters.extend(start + end for end in paths(node))

        # Suggest modifications to each parameter.
        self.experiments = []
        for path in target_parameters:
            value = lab.default_parameters.get(path)
            for mod in self.mod_funcs:
                params = deepcopy(lab.default_parameters)
                params.apply( path, mod(value) )
                try:
                    self.experiments.append(
                        ExperimentSummary(lab, parameters=params))
                except ValueError:
                    # ExperimentSummary raises ValueError if it detects
                    # duplicate entry in the database.
                    self.experiments.append(
                        lab.experiment_ids[hash(params)])

        lab.save() # Write all of the new grid-search experiments to the lab report.

    def suggestExperiment(self):
        # Start with a good baseline of the default parameters.
        if self.lab.experiment_ids[hash(self.lab.default_parameters)].attempts < 7:
            return self.lab.default_parameters

        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(self.experiments, key=lambda x: x.attempts + rnd()).parameters


class CombineBest:
    def addArguments(parser):
        parser.add_argument('--combine', type=int, default=0,
            help='Combine the NUM best experiments.')

    def useThisOptimizer(args):
        return args.combine

    def merge(self, lab, ideas):
        """ Take several experiments and return the best combination of them. """
        # Marshal all of the modifications together.
        ideas  = sorted(ideas, key = lambda x: -x.mean())
        paths  = []
        values = []
        for x in ideas:
            for path, value in x.modifications:
                if path in paths:
                    continue # Higher scoring experiments take precedence.
                paths.append(path)
                values.append(value)
        # Create or get the experiment object.
        mods = list(zip(paths, values))
        try:
            return ExperimentSummary(lab, modifications=mods)
        except ValueError:
            # ExperimentSummary raises ValueError if it detects duplicate entry
            # in the database.
            params = deepcopy(lab.default_parameters)
            for p, v in mods:
                params.apply(p, v)
            return lab.experiment_ids[hash(params)]

    def suggestExperiment(self):
        suggest = [] # Retval accumulator

        # Ignore all underperforming experiments.
        null = self.lab.experiment_ids[hash(self.lab.default_parameters)]
        ex   = [x for x in self.lab.experiments if x.mean() > null.mean()]

        # Limit to the top/best experiments.
        ex = sorted(ex, key = lambda x: -x.mean())[ : self.args.combine]

        # Keep trying experiments which are not yet significant.  Experiments
        # with a single datum have a significance of NaN...
        trymore = [x for x in ex if (x.significance() > .50 or math.isnan(x.significance()))]
        ex = [x for x in ex if x not in trymore]
        suggest.extend(trymore)
        # Suggests combinations
        for ideas in itertools.combinations(ex, 2):
            suggest.append( self.merge(self.lab, ideas) )

        if False: # Dump the suggestions for debugging
            for x in suggest:
                for p, v in x.modifications:
                    print(p , v)
                print()
            1/0

        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(suggest, key=lambda x: x.attempts + rnd()).parameters

