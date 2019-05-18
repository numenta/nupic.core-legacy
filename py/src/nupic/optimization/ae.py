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
The Automatic Experimenter

This is a framework for parameter optimization.  Key features include:
 * An API for users to hook their programs/experiments into this framework.
 * An API for adding optimization methods, and several good methods included.
 * Records everything, and helps manage the data.

Structure your program as follows:
    ExperimentModule is a python module containing the model to be optimized as
    well as code to evaluate the models performance.

    ExperimentModule.default_parameters = {}
    Global dictionary containing all of the parameters to modify.
    Parameters must be one of the following types: dict, tuple, float, int.
    Parameters can be nested in multiple levels of dictionaries and tuples.
    For more details see nupic.optimization.parameter_set

    ExperimentModule.main(parameters=default_parameters, argv=None, verbose=True)
    Returns (float) performance of parameters, to be maximized.
    For example, see file: py/src/nupic/examples/mnist.py

Run your experiment with the AE program:
$ python3 -m nupic.optimization.ae [ae-arguments] ExperimentModule.py [experiment-arguments]

The outputs and data of this program are kept in a directory named after the
experiment which generated it.  If the experiment is "foo/bar.py" then AE's
directory is "foo/bar_ae/".  The primary output of this program is the file
"foo/bar_ae/lab_report.txt" which contains a summary of its operations.  The lab
report format is:

1) Introduction.  This text is not parsed, it is preserved so you can keep notes
here.  This area is initialized with hopefully useful information, including the
experiment name, timestamps.  Do not modify the lab report file while AE is
running!

2) Methods & Analysis.  These sections contain summaries and useful information,
including:
 * The default parameters and command line invocation,
 * A leader board of the best experiments,
 * A list of all parameters which crashed the program.

3) Summary of each experiment.  Each experiments summary contains the following
information:
    1) Modified Parameters & their New Value. This is the only required field,
    the remaining fields will be generated and written to this file as soon as
    the report is loaded.  You may choose to manually add experiments to this
    lab report in this way.
    2) MD5 Checksum of Parameters and Command Line.  This hash checksum is used
    to uniquely identify an experimental setup, it's the name of the
    experiment.  These hashes are used in filenames and are searchable.
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
# all of the modifications to be diffs of the current parameters?

# TODO: Maybe the command line invocation should be included in the experiment
# hash?  Then I could experiment with the CLI args within a single lab report.

# TODO: Every run should track elapsed time and report the average in the
# experiment journal & summary.  Some of these experiments clearly take longer
# than others but its not recorded & displayed.

# TODO: Log files should report memory usage ...

# TODO: Remove Laboratory.experiment, then rename lab.experiment_ids to experiments

# TODO: Failed experiments should have its own section in the Laboratory.  Maybe
# organize them by the exception type & string?

# TODO: Consider renaming *log files to *tmp for clarity.

# TODO: Make the leader board base the P-Values off of the best experiment, and
# always keep the default parameters on the board.

# TODO: Do not lose the log file when worker process crashes!  With this fixed I
# won't need to explain what the temp files are for, the user shouldn't need to
# know about them...

# TODO: How hard would it be to allow the user to edit the lab report while the
# program is running?  Read timestamps to detect user writes.  All of the latest
# data is loaded in the program, so it should be simple to load in the new
# version and merge the human readable text into the latest data, write out to
# new file and attempt to swap it into place.

# TODO: Experiment if all of the parameters are modified, show the
# parameters instead of the modifications.  This is useful for swarming which
# touches every parameter.

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
import signal # TODO: X-Plat issue: Replace signal with threading.timer?
import re
import numpy as np
import scipy.stats

from nupic.optimization.parameter_set import ParameterSet

class Experiment:
    """
    An experiment represents a unique ParameterSet.

    Attributes:
        parameters    - ParameterSet
        lab           - Circular reference to Laboratory instance.
        attempts      - Number of times attempted to evaluate.
        scores        - List of float
        notes         - string
        journal       - File path to log file for this experiment.
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
            self._parse( string )
        elif modifications is not None:
            self.parameters = ParameterSet( self.lab.default_parameters )
            for path, value in modifications:
                self.parameters.apply( path, value )
        elif parameters is not None:
            self.parameters = ParameterSet( parameters )
        else:
            raise TypeError("Not enough arguments to Experiment.__init__()")

        self.parameters    = self.parameters.typecast( self.lab.structure )
        self.modifications = self.lab.default_parameters.diff( self.parameters )

        if hash(self) not in self.lab.experiment_ids:
            self.lab.experiments.append(self)
            self.lab.experiment_ids[hash(self)] = self
        else:
            existing = self.lab.experiment_ids[hash(self)]
            if existing.parameters == self.parameters:
                raise ValueError("Duplicate Parameters, Hash %X"%hash(self))
            else:
                raise SystemExit("Hash Collision!")

        # Start a journal file for this experiment.
        if not hasattr(self, 'journal'):
            self.journal = os.path.join(self.lab.ae_directory, "%X.journal"%hash(self))
            with open(self.journal, 'a') as file:
                file.write('Experiment Journal For Parameters:\n')
                file.write( str(self.parameters) + '\n')
                file.write('Hash: %X\n'%hash(self))
                file.write('Command Line Invocation: $ ' + ' '.join(self.lab.argv) + '\n')

    def _parse(self, string):
        # Reconstruct the parameters.
        self.modifications = []
        if "Modification:" in string:
            for change in re.findall("Modification:(.*)", string):
                path, eq, value = change.partition('=')
                self.modifications.append((path.strip(), value.strip()))
        self.parameters = ParameterSet(self.lab.default_parameters)
        for path, value in self.modifications:
            self.parameters.apply(path, value)

        if "Hash: " in string:
            # Override hash(self) with whats on file since this is reconstructed
            # from defaults + modifications, and the defaults might have changed.
            self._hash    = int(re.search("Hash: (.*)", string).groups()[0], base=16)
        if "Journal: " in string:
            self.journal  = re.search("Journal: (.*)", string).groups()[0]
        if "Attempts: " in string:
            self.attempts = int(re.search("Attempts: (.*)", string).groups()[0])
        if "Scores: " in string:
            self.scores = re.search("Scores: (.*)", string).groups()[0].strip()
            self.scores = [float(s.strip()) for s in self.scores.split(',') if s.strip()]
            assert( len(self.scores) <= self.attempts ) # Attempts may fail and not return a score.
        if "Notes:" in string:
            self.notes    = string.partition('Notes:')[2]

    # TODO: This should accept the baseline to compare against, and then have
    # the defaults argument as the default baseline.
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
            # Since both samples come from the same experimental setup  they
            # should have the same variance.
            equal_var=True,)
        return pval

    def mean(self):
        """ Returns the average score. """
        return np.mean(self.scores) if self.scores else float('-inf')

    # TODO: Consider showing min & max scores.
    # TODO: Don't show scores & P-Value if attempts == 0.
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


class Laboratory:
    """
    Main class of the AE program.

    Attributes:
        lab.module             - Users Experiment python module
        lab.name               - Name of experiment module
        lab.path               - Directory containing experiment module
        lab.structure          - Types of parameters
        lab.default_parameters - lab.module.default_parameters
        lab.argv               - Command line invocation of experiment program
        lab.tag                - Optional, identifier string for this Laboratory
        lab.ae_directory       - Directory containing all files created by this program
        lab.lab_report         - File path of Lab Report
        lab.experiments        - List of Experiment instances
        lab.experiment_ids     - Experiments accessed by their unique hash
    """
    default_extension = '_ae'
    section_divider = '\n' + ('=' * 80) + '\n'
    def __init__(self, experiment_argv, method=None, tag='', verbose=False):
        if not experiment_argv:
            raise ValueError('Missing arguments for the experiment to run!')
        if isinstance(experiment_argv, str):
            experiment_argv = experiment_argv.split()
        self.argv    = experiment_argv
        self.method  = method
        self.tag     = tag
        self.verbose = verbose
        self._load_experiment_module(experiment_argv[0])
        self.ae_directory = os.path.join(self.path, self.name) + self.default_extension
        if self.tag:
            self.ae_directory = self.ae_directory + '_' + self.tag
        self.lab_report   = os.path.join(self.ae_directory, 'lab_report.txt')
        self.experiments    = []
        self.experiment_ids = {}
        if os.path.isdir(self.ae_directory):
            with open(self.lab_report, 'r') as file:
                report = file.read()
            self._parse(report)
        else:
            # Initialize the Lab Reports attributes and write the skeleton of it
            # to file.
            self.init_header()
            os.mkdir(self.ae_directory)
        # Always have an experiment for the default parameters.
        try:
            Experiment(self,  parameters = self.default_parameters)
        except ValueError:
            pass

    def init_header(self):
        """
        Sets attribute lab.header containing the initial text in the Notes
            section at the top of the lab-report.
        """
        self.header = str(self.name)
        if self.tag:
            self.header += ' - ' + self.tag
        self.header += ' - Automatic Experiments\n'
        self.header += time.asctime( time.localtime(time.time()) ) + '\n'

    def _load_experiment_module(self, experiment_module):
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

    def _parse(self, report):
        if not report.strip():
            raise ValueError("Empty lab report file!")
        sections            = report.split(self.section_divider)
        self.header         = sections[0]
        default_parameters  = '\n'.join( sections[1].split('\n')[1:-1] )
        cli                 = sections[1].split('\n')[-1].strip('$ ').split()
        sorted_pval_table   = sections[2]
        experiment_sections = sections[3:]
        file_defaults       = ParameterSet(default_parameters)
        # Consistency check for parameters & experiment argv.
        if file_defaults != self.default_parameters or cli != self.argv:
            while True:
                q = input("Default parameters or invocation have changed, options:\n" + 
                          "  old - Ignore the new/given, use what's on file.\n" +
                          "  new - Use the new/given, overwrites the old file!\n" +
                          "  abort.\n" +
                          ">>> ")
                q = q.strip().lower()
                if q == 'old':
                    self.default_parameters = file_defaults
                    self.argv               = cli
                    break
                elif q == 'new':
                    shutil.copy(self.lab_report, self.lab_report + '.backup')
                    break
                elif q in ('abort', 'exit', 'quit') or q in 'aeq':
                    sys.exit()

        [Experiment(self, s) for s in experiment_sections if s.strip()]

    def get_experiment(self, parameters):
        """
        Returns Experiment instance for the given parameters.  If one does not
        already exist for these parameter then it is created.
        """
        p = ParameterSet( parameters ).typecast( self.structure )
        h = hash(p)
        if h in self.experiment_ids:
            return self.experiment_ids[h]
        else:
            return Experiment(self, parameters=p)

    def significant_experiments_table(self):
        """ Returns string """
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
        """ Main loop of the AE program. """
        pool = multiprocessing.Pool(processes, maxtasksperchild=1)
        async_results = [] # Contains pairs of (Promise, Experiment)
        while True:
            # Check for jobs which have finished
            run_slot = 0
            while run_slot < len(async_results):
                promise, value = async_results[run_slot]
                if promise.ready():
                    # Experiment run has finished, deal with the results.
                    self.collect_results(value, promise)
                    async_results.pop(run_slot)
                else:
                    run_slot += 1
            # Start running new experiments
            while len(async_results) < processes:
                X = self.get_experiment( self.method.suggest_parameters() )
                if self.verbose:
                    print("Evaluating %X"%hash(X))
                promise = pool.apply_async(
                    _Experiment_evaluate_parameters,
                    args = (self.argv, self.tag, self.verbose, X.parameters,),
                    kwds = {'time_limit'   : time_limit,
                            'memory_limit' : memory_limit,},)
                async_results.append((promise, X))
            # Wait for experiments to complete
            time.sleep(2)

    def evaluate_parameters(self, parameters,
        time_limit   = None,
        memory_limit = None,):
        """
        Run the users program/experiment with the given parameters.
        This function should execute in a child processes.
        """
        # Redirect stdout & stderr to a temporary file.
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
        # TODO: Deal with all of the contingencies where this fails.  Do not
        # lose the journal file!  Do append that time-stamp!
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

    def collect_results(self, experiment, async_promise):
        """ Deals with the aftermath & bookkeeping of running an experiment. """
        try:
            score, run_journal = async_promise.get()
        except (ValueError, MemoryError, ZeroDivisionError, AssertionError, RuntimeError) as err:
            print("")
            print( str( experiment.parameters ))
            print("%s:"%(type(err).__name__), err)
            print("")
            score = err
        except Exception:
            print("")
            print( str( experiment.parameters ))
            print("Unhandled Exception.")
            print("")
            raise

        # Update this experiment
        experiment.attempts += 1
        if not isinstance(score, Exception):
            experiment.scores.append(score)
            # Append the temporary journal file to the experiments journal.
            # TODO !!! Don't lose the data vvv
            # Sadly if the experiment crashes, the temp file is abandoned and
            # the debugger (you) must search for it manually if they want to see it...
            with open(run_journal) as journal:
                content = journal.read()
            with open(experiment.journal, 'a') as experiment_journal:
                experiment_journal.write(self.section_divider)
                experiment_journal.write(content)
            os.remove(run_journal)
        # Notify the parameter optimization method that the parameters which it
        # suggested have finished evaluating.
        self.method.collect_results(experiment.parameters, score)
        self.save()     # Write the updated Lab Report to file.

def _Experiment_evaluate_parameters(*args, **kwds):
    """
    Global wrapper for Laboratory.evaluate_parameters which is safe for
    multiprocessing.
    """
    experiment_argv, tag, verbose, parameters = args
    self = Laboratory(experiment_argv, tag = tag, verbose = verbose)
    return self.evaluate_parameters( parameters, **kwds)

def _timeout_callback(signum, frame):
    raise ValueError("Time limit exceeded.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true',
        help='Passed onto the experiment\'s main function.')
    parser.add_argument('--tag', type=str,
        help='Optional string appended to the name of the AE directory.  Use tags to '
             'keep multiple variants of an experiment alive and working at the same time.')
    parser.add_argument('-n', '--processes',  type=int, default=os.cpu_count(),
        help='Number of experiments to run simultaneously, defaults to the number of CPU cores available.')
    parser.add_argument('--time_limit',  type=float, default=None,
        help='Hours, time limit for each run of the experiment.',)
    parser.add_argument('--memory_limit',  type=float, default=None,
        help='Gigabytes, RAM memory limit for each run of the experiment.')
    parser.add_argument('--parse',  action='store_true',
        help='Parse the lab report and write it back to the same file, then exit.')
    parser.add_argument('--rmz', action='store_true',
        help='Remove all experiments which have zero attempts.')
    parser.add_argument('experiment', nargs=argparse.REMAINDER,
        help='Name of experiment module followed by its command line arguments.')

    import nupic.optimization.optimizers as optimizers
    from nupic.optimization.swarming import ParticleSwarmOptimization
    all_optimizers = [
        optimizers.EvaluateDefaultParameters,
        optimizers.EvaluateAllExperiments,
        optimizers.EvaluateBestExperiment,
        optimizers.EvaluateHashes,
        optimizers.GridSearch,
        optimizers.CombineBest,
        ParticleSwarmOptimization,
    ]
    assert( all( issubclass(Z, optimizers.BaseOptimizer) for Z in all_optimizers))
    for method in all_optimizers:
        method.add_arguments(parser)

    args = parser.parse_args()
    selected_method = [X for X in all_optimizers if X.use_this_optimizer(args)]

    ae = Laboratory(args.experiment,
        tag      = args.tag,
        verbose  = args.verbose)
    ae.save()
    print("Lab Report written to %s"%ae.lab_report)

    if args.parse:
        pass

    elif args.rmz:
        for x in ae.experiments:
            if x.attempts == 0:
                ae.experiments.remove(x)
                ae.experiment_ids.pop(hash(x))
        ae.save()
        print("Removed all experiments which had not yet been attempted.")

    elif not selected_method:
        print("Error: missing argument for what to to.")
    elif len(selected_method) > 1:
        print("Error: too many argument for what to to.")
    else:
        ae.method = selected_method[0]( ae, args )

        giga = 2**30
        if args.memory_limit is not None:
            memory_limit = int(args.memory_limit * giga)
        else:
            # TODO: Not X-Platform, replace with "psutil.virtual_memory.available"
            available_memory = int(os.popen("free -b").readlines()[1].split()[3])
            memory_limit = int(available_memory / args.processes)
            print("Memory Limit %.2g GB per instance."%(memory_limit / giga))

        ae.run( processes    = args.processes,
                time_limit   = args.time_limit,
                memory_limit = memory_limit,)

    print("Exit.")
