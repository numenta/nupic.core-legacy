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

from .nupic.optimization.parameter_set import ParameterSet
import itertools
import random


class GridSearch(object):
    """ TODO: docstring for GridSearch"""
    mod_funcs = [
        lambda v: v *  .40,
        lambda v: v *  .60,
        lambda v: v *  .80,
        lambda v: v * 1.20,
        lambda v: v * 1.40,
        lambda v: v * 1.60,
    ]

    def __init__(self, directive):
        self.directive = directive

    def __call__(self, lab):

        if lab.experiment_ids[hash(lab.default_parameters)].attempts < 10:
            return lab.default_parameters

        # Get a list of every parameter to experiment with.
        if self.directive:
            manifest = []
            for start in self.directive.split(','):
                node = eval("lab.default_parameters" + start)
                manifest.extend(start + end for end in paths(node))
        else:
            manifest = lab.default_parameters.enumerate()

        # Suggest the following modifications to each parameter.
        experiments = []
        for path in manifest:
            value = lab.default_parameters.get(path)
            for mod in self.mod_funcs:
                params = deepcopy(lab.default_parameters)
                params.apply( path, mod(value) )
                try:
                    experiments.append(
                        ExperimentSummary(lab, parameters=params))
                except ValueError:
                    # ExperimentSummary raises ValueError if it detects
                    # duplicate entry in the database.
                    experiments.append(
                        lab.experiment_ids[hash(params)])

        lab.save() # Write all of the new grid-search experiments to the lab report.

        rnd = random.random
        return min(experiments, key=lambda x: x.attempts + rnd()).parameters



class CombineBest:
    """ TODO Docs """
    def __init__(self, n=20):
        self.n = n

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

    def __call__(self, lab):

        suggest = [] # Retval accumulator
        # Ignore all underperforming experiments.
        null = lab.experiment_ids[hash(lab.default_parameters)]
        ex   = [x for x in lab.experiments if x.mean() > null.mean()]
        # For sanity: Limit to the top experiments.
        ex = sorted(ex, key = lambda x: -x.mean())[ : self.n]
        # Keep trying experiments which are not yet significant.  Experiments
        # with a single datum have a significance of NaN...
        trymore = [x for x in ex if (x.significance() > .50 or math.isnan(x.significance()))]
        ex = [x for x in ex if x not in trymore]
        suggest.extend(trymore)
        # Suggests combinations
        for ideas in itertools.combinations(ex, 2):
            suggest.append( self.merge(lab, ideas) )

        if False: # Dump the suggestions for debugging
            for x in suggest:
                for p, v in x.modifications:
                    print(p , v)
                print()
            1/0
        rnd = random.random
        return min(suggest, key=lambda x: x.attempts + rnd()).parameters

