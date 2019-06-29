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

from htm.optimization.parameter_set import ParameterSet
import random
import itertools
import math

class BaseOptimizer:
    """
    Optimizer classes control what parameters to try.  This class defines the
    API which they must implement.
    """
    @classmethod
    def add_arguments(parser):
        """
        Argument parser is an instance of ArgumentParser, from the standard
        library argparse.  Optimizer classes should add their command line
        arguments to this.
        """
        pass

    @classmethod
    def use_this_optimizer(arguments):
        """
        Argument is the parsed arguments, result of add_arguments and the users
            command line.

        Returns bool, if the user has requested to use this optimizer via
            command line arguments.
        """
        return False

    def __init__(self, laboratory, arguments):
        """
        Argument laboratory is the main class of the framework.
            See class htm.optimization.ae.Laboratory

        Argument arguments is the parsed arguments, result of add_arguments and
            the users command line.
        """
        self.lab  = laboratory
        self.args = arguments

    def suggest_parameters(self):
        """
        Returns instance of ParameterSet, to be evaluated.  The parameters will
            be type cast before being passed to the main function of the users
            program/experiment.
        """
        raise NotImplementedError("BaseOptimizer.suggest_parameters")

    def collect_results(self, parameters, result):
        """
        Argument parameters was returned by suggest_parameters, and has now been
                 evaluated.

        Argument results is either a float or an exception.
            If results is a float, then it is the score to be maximized.
            If results is an Exception, then it was raised by the experiment.

        This method is optional, optimizers do not need to implement this.
        """
        pass


class EvaluateDefaultParameters(BaseOptimizer):
    def add_arguments(parser):
        parser.add_argument('--default_parameters', action='store_true',
            help='Evaluate only "experiment_module.default_parameters".')

    def use_this_optimizer(args):
        return args.default_parameters

    def suggest_parameters(self):
        return self.lab.default_parameters


class EvaluateAllExperiments(BaseOptimizer):
    def add_arguments(parser):
        parser.add_argument('--all_experiments', action='store_true',
            help='Evaluate all experiments in the lab report, don\'t start new experiments.')

    def use_this_optimizer(args):
        return args.all_experiments

    def suggest_parameters(self):
        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(self.lab.experiments, key=lambda X: X.attempts + rnd()).parameters


class EvaluateBestExperiment(BaseOptimizer):
    def add_arguments(parser):
        parser.add_argument('--best', action='store_true',
            help='Evaluate the best set of parameters on file.')

    def use_this_optimizer(args):
        return args.best

    def __init__(self, lab, args):
        super().__init__(lab, args)
        if lab.verbose:
            print("Best parameters:")
            print(str( self.suggest_parameters() ))

    def suggest_parameters(self):
        best = max(self.lab.experiments, key = lambda X: X.mean() )
        return best.parameters


class EvaluateHashes(BaseOptimizer):
    def add_arguments(parser):
        parser.add_argument('--hashes', type=str,
            help='Evaluate specific experiments, identified by their hashes.  Comma separated list.')

    def use_this_optimizer(args):
        return args.hashes

    def __init__(self, lab, args):
        hashes = [int(h, base=16) for h in args.hashes.split(',')]
        try:
            self.experiments = [lab.experiment_ids[h] for h in hashes]
        except KeyError:
            unknown = [h for h in hashes if h not in lab.experiment_ids]
            raise ValueError('Hash not recognized: %X'%unknown[0])

    def suggest_parameters(self):
        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(self.experiments, key=lambda X: X.attempts + rnd()).parameters


class GridSearch(BaseOptimizer):
    # TODO: Make these into a CLI argument.
    mod_funcs = [
        lambda v: v *  .40,
        lambda v: v *  .60,
        lambda v: v *  .80,
        lambda v: v * 1.20,
        lambda v: v * 1.40,
        lambda v: v * 1.60,
    ]

    def add_arguments(parser):
        parser.add_argument('--grid_search', type=str,
            help="Grid Search, parameter to search, use \"\" for all.")

    def use_this_optimizer(args):
        return args.grid_search is not None

    def __init__(self, laboratory, args):
        self.lab = laboratory

        # Get a list of every parameter to experiment with.
        target_parameters = []
        for start in args.grid_search.split(','):
            node = self.lab.default_parameters.get( start )
            subtree = ParameterSet.enumerate( node )
            target_parameters.extend( start + end for end in subtree )

        # Suggest modifications to each parameter.
        self.experiments = []
        for path in target_parameters:
            value = self.lab.default_parameters.get(path)
            for mod in self.mod_funcs:
                params = ParameterSet(self.lab.default_parameters)
                params.apply( path, mod(value) )
                X = self.lab.get_experiment( params )
                if not X.notes.strip():
                    X.notes += "Suggested by Grid Search.\n"
                self.experiments.append(X)

        self.lab.save() # Write all of the new grid-search experiments to the lab report.

    def suggest_parameters(self):
        # Start with a good baseline of the default parameters.
        if self.lab.experiment_ids[hash(self.lab.default_parameters)].attempts < 5:
            return self.lab.default_parameters

        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(self.experiments, key=lambda X: X.attempts + rnd()).parameters


class CombineBest(BaseOptimizer):
    def add_arguments(parser):
        parser.add_argument('--combine', type=int, default=0,
            help='Combine the NUM best experiments.')

    def use_this_optimizer(args):
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
        # Create and get the experiment object.
        params = ParameterSet(lab.default_parameters)
        for p, v in zip(paths, values):
            params.apply(p, v)
        return lab.get_experiment(params)

    def suggest_parameters(self):
        suggest = [] # Return value accumulator

        # Ignore all under-performing experiments.
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
            X = self.merge(self.lab, ideas)
            if not X.notes.strip():
                X.notes += "Suggested by Combine Best.\n"
            suggest.append( X )

        rnd = lambda: random.random() / 100 # Random Tiebreaker
        return min(suggest, key=lambda x: x.attempts + rnd()).parameters

