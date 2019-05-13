#!/usr/bin/python3
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
"""
Automatic Experimenter

This is a framework for parameter optimization.
 * It methodically records the results of different sets of parameters and
analyses the results.  It then automatically suggests and evaluates
modifications to the parameters.
 * It exposes a convenient API for users to hook their program into this.
 * The framework allows for testing each set of parameters several times and
calculates the average and standard deviation of the results.  It also
calculates the confidence that a parameter change caused the score to change.
 * It is extensible: new methods for automated parameter optimization can be
added.  Currently this implements a basic grid search strategy.  In the future I
hope to add a particle swarming method.

To use this module, structure experiments as follows:
    ExperimentModule is a python3 module containing the model to be optimized as
    well as code to evaluate model performance.

    ExperimentModule.default_parameters = {}
    This global dictionary contains all of the parameters to modify.
    Parameters must be one of the following types: dict, tuple, float, int.
    Parameters can be nested in multiple levels of dictionaries and tuples.

    ExperimentModule.main(parameters=default_parameters, argv=None, verbose=True)
    Returns (float) performance of parameters, to be maximized.

Usage: $ ae.py [ae-arguments] ExperimentModule.py [experiment-arguments]

The outputs and data of this program are kept in a directory named after the
experiment which generated it.  If the experiment is "foo/bar.py" then AE's
directory is "foo/bar_ae/".  The primary output of this program is the file
"foo/bar_ae/lab_report.txt" which contains a summary of its operations.  The lab
report format is:

1) Introduction.  This text is not parsed, it is preserved so you can keep notes
here  This area is initialized with hopefully useful information, including
the experiment name, timestamps.

2) Methods.  This section contains the default parameters and the command line
invocation.

3) Summary of each experiment.  Each experiments summary contains the following
information:
    1) Modified Parameters & their New Value. This is the only required field,
    the remaining fields will be generated and written to this file as soon as
    the report is loaded.  You may choose to manually add experiments to this
    lab report in this way.
    2) MD5 Checksum of Parameters and Command Line.  This hash checksum is used
    to uniquely identify an experimental setup, it's the name of the
    experiment.  These hashes are used in filenames and searching for a hash
    finds all references to it.
    3) File Path of Experiment Journal
    4) Number of Attempted Runs
    5) Score of each Completed Run
    6) Mean & Standard Deviation of Scores
    7) Notes, these are not parsed they are preserved so you can keep notes here

This program keeps a Journal of each experimental setup.  Journals are named
after the hash of the parameters & command line they contain, with the extension
".journal".  Journals start with a self contained description of how to
reproduce the experiment, followed by a section for every run of this
experiment.  The section for each run contains the output (std-out & std-err) of
the program, as well as diagnostic information such as timestamps and memory
usage reports.  Files with the extension ".log" are temporary files for in-
progress experiment, and when the experiment finishes running they are copied to
their journal and then the ".log" file is deleted.
"""

# TODO: Default parameters need better handling...  When they change, update
# all of the modifications to be diffs of the current parameters.

# TODO: Maybe the command line invocation should be included in the experiment
# hash?  Then I could experiment with the CLI args within a single lab report.

# TODO: Every run should track elapsed time and report the average in the
# experiment journal & summary.  Some of these experiments clearly take longer
# than others but its not recorded.

# TODO: Log files should report memory usage ...

# TODO: Remove LabReport.experiment, then rename lab.experiment_ids to experiments

# TODO: Reject experiments which have failed a few times.

import argparse
import os
import sys
import shutil
import random
import pprint
import time
import datetime
import tempfile
import multiprocessing
import resource
import signal # TODO: X-Plat issue: Replace signal with threading.timer
from copy import copy, deepcopy
import re
import numpy as np
import scipy
import math

from .nupic.optimization.parameter_set import ParameterSet

class ExperimentSummary:
    """
    Attributes:
        lab           - circular reference to LabReport instance
        attempts      -
        scores        -
        notes         -
        journal       -
        parameters    -
        modifications - 
    """
    def __init__(self, lab,
        string=None,
        modifications=None,
        parameters=None,):
        """ """
        self.lab      = lab
        self.attempts = 0
        self.scores   = []
        self.notes    = ' '
        # Load or create this experiment's data.
        if string is not None:
            self.parse(string)
        elif modifications is not None:
            self.parameters = deepcopy(self.lab.default_parameters)
            for path, value in modifications:
                self.parameters.apply(path, value)
        elif parameters is not None:
            self.parameters = ParameterSet(parameters)
        else:
            raise TypeError("Not enough arguments to ExperimentSummary.__init__()")

        self.parameters    = self.parameters.typecast_parameters( self.lab.structure )
        self.modifications = self.lab.default_parameters.diff(self.parameters)

        if hash(self) not in self.lab.experiment_ids:
            self.lab.experiments.append(self)
            self.lab.experiment_ids[hash(self)] = self
        else:
            raise ValueError("Duplicate Parameters Hash %X"%hash(self))

        # Start a journal file for this experiment.
        if not hasattr(self, 'journal'):
            self.journal = os.path.join(self.lab.ae_directory, "%X.journal"%hash(self))
            with open(self.journal, 'a') as file:
                file.write('Experiment Journal For Parameters:\n')
                file.write(pprint.pformat(self.parameters) + '\n')
                file.write('Hash: %X\n'%hash(self))
                file.write('Command Line Invocation: $ ' + ' '.join(self.lab.argv) + '\n')
        else:
            # Scrape some info from the journal file.
            with open(self.journal, 'r') as file:
                journal = file.read()
            journal = journal.split(self.lab.section_divider)
            journal.pop(0) # Discard header
            elapsed_times = []
            memory_usages = []

    def parse(self, string):
        # Reconstruct the parameters.
        self.modifications = []
        if "Modification:" in string:
            for change in re.findall("Modification: (.*)", string):
                path, eq, value = change.partition('=')
                self.modifications.append((path.strip(), value.strip()))
        self.parameters = deepcopy(self.lab.default_parameters)
        for path, value in self.modifications:
            self.parameters.apply(path, value)
        #
        if "Attempts:" in string:
            self.attempts = int(re.search("Attempts: (.*)", string).groups()[0])
        if "Scores:" in string:
            self.scores   = re.search("Scores: (.*)", string).groups()[0].strip()
            self.scores   = [float(s.strip()) for s in self.scores.split(',') if s.strip()]
        if "Journal:" in string:
            self.journal  = re.search("Journal: (.*)", string).groups()[0]
        if "Notes:" in string:
            self.notes    = string.partition('Notes:')[2]
        if "Hash:" in string:
            # Override hash(self) with whats on file since this is reconstructed
            # from defaults + modifications, and the defaults might have changed.
            self._hash    = int(re.search("Hash: (.*)", string).groups()[0], base=16)

    # TODO: This should accept the baseline to compare against, and then have
    # the defaults parameters as the default baseline.
    def significance(self):
        """
        Returns the P-Value of the Null-Hypothesis test (these parameters
        against the default parameters)
        """
        try:
            null_experiment = self.lab.experiment_ids[hash(self.lab.default_parameters)]
        except KeyError:
            return float('nan')
        if not self.scores or not null_experiment.scores:
            return float('nan')
        if len(self.scores) == 1:
            pass # TODO: How to pass probabilities & statistics?
        stat, pval = scipy.stats.ttest_ind(
            null_experiment.scores, self.scores, axis=None,
            # Since both samples come from the same experimential setup  they
            # should have the same variance.
            equal_var=True,)
        return pval

    def mean(self):
        return np.mean(self.scores) if self.scores else float('-inf')

    def __str__(self):
        s = ''
        if not self.modifications:
            s += "Default Parameters\n"
        for mod, value in self.modifications:
            s += "Modification: %s = %s\n"%(mod, str(value))
        s += 'Hash: %X\n'%hash(self)
        s += 'Journal: %s\n'%self.journal
        s += 'Attempts: %d\n'%self.attempts
        s += 'Scores: %s\n'%', '.join(str(s) for s in self.scores)
        if self.scores:
            mean = np.mean(self.scores)
            std  = np.std(self.scores)
            s += 'Mean & Standard Deviation: %g & %g\n'%(mean, std)
        s += 'P-Value: %g\n'%self.significance()
        s += 'Notes:' + self.notes
        return s

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash  = hash(self.parameters)
        return self._hash


class LabReport:
    """
    Attributes:
        lab.module             - Experiment python module
        lab.name               - Name of experiment module
        lab.path               - Directory containing experiment module
        lab.structure          - Types of parameters
        lab.default_parameters - ex.module.default_parameters
        lab.argv               - Command line invocation of experiment program
        lab.tag                - Optional, identifier string for this LabReport
        lab.ae_directory       - Directory containing all files created by this program
        lab.lab_report         - File path of Lab Report
        lab.experiments        - List of ExperimentSummary
        lab.experiment_ids     - Experiments accessed by their unique hash
    """
    default_extension = '_ae'
    section_divider = '\n' + ('=' * 80) + '\n'
    def __init__(self, experiment_argv, method=None, tag='', verbose=False):
        if isinstance(experiment_argv, str):
            experiment_argv = experiment_argv.split()
        self.argv    = experiment_argv
        self.method  = method
        self.tag     = tag
        self.verbose = verbose
        self.load_experiment_module(experiment_argv[0])
        self.ae_directory = os.path.join(self.path, self.name) + self.default_extension
        if self.tag:
            self.ae_directory = self.ae_directory + '_' + self.tag
        self.lab_report   = os.path.join(self.ae_directory, 'lab_report.txt')
        self.experiments    = []
        self.experiment_ids = {}
        if os.path.isdir(self.ae_directory):
            with open(self.lab_report, 'r') as file:
                report = file.read()
            self.parse_lab_report(report)
        else:
            # Initialize the Lab Reports attributes and write the skeleton of it
            # to file.
            self.init_header()
            os.mkdir(self.ae_directory)
        # Always have an experiment for the default parameters.
        try:
            ExperimentSummary(self,  parameters = self.default_parameters)
        except ValueError:
            pass

        # Parse & Write this file immediately at start up.
        self.save()

    def init_header(self):
        self.header = str(self.name)
        if self.tag:
            self.header += ' - ' + self.tag
        self.header += ' - Automatic Experiments\n'
        self.header += time.asctime( time.localtime(time.time()) ) + '\n'

    def load_experiment_module(self, experiment_module):
        """
        Argument experiment_module is command line argument 0, specifying the
        file path to the experiment module.
        """
        self.path, experiment_module = os.path.split(experiment_module)
        self.name, dot_py = os.path.splitext(experiment_module)
        assert(dot_py == '.py')
        self.module_reload  = 'import sys; sys.path.append("%s"); '%self.path
        self.module_reload += 'import %s; '%self.name
        exec_globals = {}
        exec(self.module_reload, exec_globals)
        self.module = exec_globals[self.name]

        self.default_parameters = ParameterSet(self.module.default_parameters)
        self.structure = self.default_parameters.get_types()

    def parse_lab_report(self, report):
        if not report.strip():
            raise ValueError("Empty lab report file.")
        sections            = report.split(self.section_divider)
        self.header         = sections[0]
        default_parameters  = '\n'.join( sections[1].split('\n')[1:-1] )
        cli                 = sections[1].split('\n')[-1].strip('$ ').split()
        sorted_pval_table   = sections[2]
        experiment_sections = sections[3:]
        file_defaults = ParameterSet(default_parameters)
        # Consistency check for parameters.
        if file_defaults != self.default_parameters or cli != self.argv:
            while True:
                q = input("Default parameters or invovation have changed, options:\n" + 
                          "    old - Ignore the new/given, use what's on file.\n" +
                          "    new - Use the new/given, overwrites the old file!\n" +
                          "    abort.\n")
                q = q.strip().lower()
                if q == 'old':
                    self.default_parameters = file_defaults
                    self.argv               = cli
                    break
                elif q == 'new':
                    shutil.copy(self.lab_report, self.lab_report + '.backup')
                    break
                elif q == 'abort':
                    sys.exit()

        [ExperimentSummary(self, s) for s in experiment_sections if s.strip()]

    def significant_experiments_table(self):
        """
        Returns string
        """
        ex = sorted(self.experiments, key = lambda x: -x.mean())
        ex = ex[:20]
        s = '    Hash |   N |      Score |   P-Value | Modifications\n'
        fmt = '%8X | %3d | % 10g | % 9.3g | '
        for x in ex:
            s += fmt%(hash(x), len(x.scores), x.mean(), x.significance())
            if not x.modifications:
                s += 'Default Parameters\n'
            else:
                for idx, mod in enumerate(x.modifications):
                    param, value = mod
                    if idx > 0:
                        s += ' ' * 42
                    s += '%s = %s\n'%(param, str(value))
        return s

    def __str__(self):
        """ Returns the lab report. """
        s  = self.header
        s += self.section_divider
        s += 'Default Parameter Values = \n'
        s += pprint.pformat(self.default_parameters)
        s += '\n$ ' + ' '.join(self.argv)
        s += self.section_divider
        s += self.significant_experiments_table().rstrip()
        s += '\n\nFailed Experiments: '
        for x in self.experiments:
            if x.attempts > len(x.scores):
                s += '%X '%hash(x)
        s += self.section_divider
        s += self.section_divider.join(str(s) for s in self.experiments)
        return s

    def save(self):
        with open(self.lab_report + '.tmp', 'w') as file:
            file.write( str(self) )
        os.rename(self.lab_report + '.tmp', self.lab_report)

    def run(self, processes,
        time_limit   = None,
        memory_limit = None,):
        """
        """
        pool = multiprocessing.Pool(processes, maxtasksperchild=1)
        async_results = [] # Contains pairs of (Promise, Parameters)

        while True:
            # Check for jobs which have finished
            run_slot = 0
            while run_slot < len(async_results):
                promise, value = async_results[run_slot]
                if promise.ready():
                    # Experiment run has finished, deal with the results.
                    result = self._get_promised_results(promise, value)
                    self.save_results(value, result)
                    async_results.pop(run_slot)
                else:
                    run_slot += 1

            # Start running new experiments
            while len(async_results) < processes:
                # Pickle is picky, so clean up 'self' which is sent via pickle
                # to the process pool. pickle_self only needs to work with
                # evaluate_parameters
                pickle_self = copy(self)
                pickle_self.module = None  # Won't pickle, use self.module_reload instead.
                # Pickle balks at circular references, remove them.
                pickle_self.experiments    = None
                pickle_self.experiment_ids = None
                value = self.method(self)
                value = value.typecast_parameters( self.structure )
                if self.verbose:
                    print("%X"%hash(value))
                promise = pool.apply_async(
                    Experiment_evaluate_parameters,
                    args = (pickle_self, value,),
                    kwds = {'time_limit'   : time_limit,
                            'memory_limit' : memory_limit,},)
                async_results.append((promise, value))
            # Wait for experiments to complete
            time.sleep(1)

    def _get_promised_results(self, promise, value):
        try:
            return promise.get()
        except (ValueError, MemoryError, ZeroDivisionError, AssertionError) as err:
            print("")
            pprint.pprint(value)
            print("%s:"%(type(err).__name__), err)
            print("")
        except Exception:
            print("")
            pprint.pprint(value)
            print("Unhandled Exception.")
            print("")
            raise

    def evaluate_parameters(self, parameters,
        time_limit   = None,
        memory_limit = None,):
        """
        This function executes in a child processes.
        """
        parameters = parameters.typecast_parameters( self.structure )
        # Redirect stdour & stderr to a temporary file.
        journal = tempfile.NamedTemporaryFile(
            mode      = 'w+t',
            delete    = False,
            buffering = 1,
            dir       = self.ae_directory,
            prefix    = "%X_"%hash(parameters),
            suffix    = ".log",)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = journal
        sys.stderr = journal
        start_time = time.time()
        journal.write("Started: " + time.asctime( time.localtime(start_time) ) + '\n')
        # Setup memory limit
        if memory_limit is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
        # Setup time limit
        if time_limit is not None:
            signal.signal(signal.SIGALRM, _timeout_callback)
            time_limit = max(1, int(round(time_limit * 60 * 60)))
            signal.alarm(time_limit)

        eval_str = (self.module_reload + 
            'score = %s.main(parameters=%s, argv=[%s], verbose=%s)'%(
                self.name,
                repr(parameters),
                ', '.join(repr(arg) for arg in self.argv[1:]),
                str(self.verbose)))
        exec_globals = {}
        exec(eval_str, exec_globals)

        # Clean up time limit
        if time_limit is not None:
            signal.alarm(0)
        # Clean up memory limit
        if memory_limit is not None:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        # Restore file descriptors
        sys.stdout, sys.stderr = stdout, stderr
        run_time = datetime.timedelta(seconds = time.time() - start_time)
        journal.write("Elapsed Time: " + str(run_time))

        return exec_globals['score'], journal.name

    def save_results(self, parameters, result):
        # Update this experiment
        param_hash = hash(ParameterSet(parameters))
        if param_hash in self.experiment_ids:
            experiment = self.experiment_ids[param_hash]
        else:
            experiment = ExperimentSummary(self, parameters = parameters)
        experiment.attempts += 1
        if result is not None:
            score, run_journal = result
            experiment.scores.append(score)

        self.save()     # Write the updated Lab Report to file.

        # Append the temporary journal file to the experiments journal.
        if result is None:
            # Sadly if the experiment crashes, the temp file is abandoned and
            # the debugger must search for it manually if they want to see it...
            return
        with open(run_journal) as journal:
            content = journal.read()
        with open(experiment.journal, 'a') as experiment_journal:
            experiment_journal.write(self.section_divider)
            experiment_journal.write(content)
        os.remove(run_journal)

def Experiment_evaluate_parameters(self, *args, **kwds):
    """
    Global wrapper for LabReport.evaluate_parameters which is safe for
    multiprocessing.
    """
    return LabReport.evaluate_parameters(self, *args, **kwds)

def _timeout_callback(signum, frame):
    raise ValueError("Time limit exceded.")

################################################################################


def evaluate_default_parameters(lab):
    return lab.default_parameters


class EvaluateHashes:
    def __init__(self, hashes):
        self.hashes = [int(h, base=16) for h in hashes]

    def __call__(self, lab):
        try:
            experiments = [lab.experiment_ids[h] for h in self.hashes]
        except KeyError:
            unknown = [h for h in self.hashes if h not in lab.experiment_ids]
            raise ValueError('Hash not recognized: %X'%unknown[0])
        rnd = random.random
        return min(experiments, key=lambda x: x.attempts + rnd()).parameters
        return random.choice(experiments).parameters


def evaluate_all(lab):
    rnd = random.random
    return min(lab.experiments, key=lambda x: x.attempts + rnd()).parameters


def evaluate_best(lab):
    best = max(lab.experiments, key = lambda X: X.mean() )
    return best.parameters


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--verbose', action='store_true',)
    arg_parser.add_argument('--tag', type=str,
        help='Optional string appended to the name of the AE directory.  Use tags to '
             'keep multiple variants of an experiment alive and working at the same time')
    arg_parser.add_argument('-n', '--processes',  type=int, default=os.cpu_count(),)
    arg_parser.add_argument('--time_limit',  type=float, default=None,
        help='Hours, time limit for each run of the experiment.',)
    arg_parser.add_argument('--memory_limit',  type=float, default=None,
        help='Gigabytes, RAM memory limit for each run of the experiment.')
    arg_parser.add_argument('experiment', nargs=argparse.REMAINDER,
        help='Name of experiment module followed by its command line arguments.')

    action_parser = arg_parser.add_mutually_exclusive_group(required=True)

    action_parser.add_argument('--parse',  action='store_true',
        help='Parse the lab report and write it back to the same file, then exits.')

    action_parser.add_argument('--rmz', action='store_true',
        help='Remove all experiments which have zero attempts.')

    action_parser.add_argument('--default_parameters', action='store_true',)

    action_parser.add_argument('--all_experiments', action='store_true',
        help='Evaluate all experiments in the lab report, don\'t start new experiments')

    action_parser.add_argument('--hashes', type=str,)

    action_parser.add_argument('--best', action='store_true',
        help='Evaluate the best set of parameters on file, with verbose=True.')

    action_parser.add_argument('--grid_search', type=str)

    action_parser.add_argument('--combine', type=int, default=0, help='Combine the NUM best experiments.')

    action_parser.add_argument('--swarming', type=int, default=0, help='Particle Swarm Optimization.')

    args = arg_parser.parse_args()
    giga = 2**30
    if args.memory_limit is not None:
        memory_limit = int(args.memory_limit * giga)
    else:
        # TODO: Not X-Platform ...
        available_memory = int(os.popen("free -b").readlines()[1].split()[3])
        memory_limit = int(available_memory / args.processes)
    print("Memory Limit %.2g GB per instance."%(memory_limit / giga))

    ae = LabReport(args.experiment,
        tag     = args.tag,
        verbose = args.verbose)

    if args.parse:
        print("Lab Report written to %s"%ae.lab_report)
        print("Exit.")
        sys.exit(0) # All done.

    elif args.rmz:
        rm = [x for x in ae.experiments if x.attempts == 0]
        for x in rm:
            ae.experiments.remove(x)
            ae.experiment_ids.pop(hash(x))
        ae.save()
        sys.exit(0) # All done.

    elif args.default_parameters:
        ae.method = evaluate_default_parameters

    elif args.all_experiments:
        ae.method = evaluate_all

    elif args.hashes:
        ae.method = EvaluateHashes(args.hashes.split(','))

    elif args.best:
        ae.method = evaluate_best

    elif args.grid_search:
        from .nupic.optimization.basic_search import GridSearch
        ae.method = GridSearch(args.grid_search)

    elif args.combine:
        from .nupic.optimization.basic_search import CombineBest
        ae.method = CombineBest(args.combine)

    elif args.swarming:
        from .nupic.optimization.swarming import ParticleSwarmOptimizations
        ae.method = ParticleSwarmOptimizations( ae, args.swarming )

    else:
        print("Missing command line argument: what to do?")
        sys.exit(1)

    ae.run(
        processes    = args.processes,
        time_limit   = args.time_limit,
        memory_limit = memory_limit,)
