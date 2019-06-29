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
    For more details see htm.optimization.parameter_set

    ExperimentModule.main(parameters=default_parameters, argv=None, verbose=True)
    Returns (float) performance of parameters, to be maximized.
    For example, see file: py/htm/examples/mnist.py

Run your experiment with the AE program:
$ python3 -m htm.optimization.ae [ae-arguments] ExperimentModule.py [experiment-arguments]

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
usage reports.
"""

# TODO: Default parameters need better handling...  When they change, update all
# of the modifications to be diffs from the new parameters?  I could scrape the
# correct parameters from the experiment journal.

# TODO: Maybe the command line invocation should be included in the experiment
# hash?  Then I could experiment with the CLI args within a single lab report.

# TODO: Every run should track elapsed time and report the average in the
# experiment journal & summary.  Some of these experiments clearly take longer
# than others but its not displayed in an easy to find way.

# TODO: Log files should report memory usage ...

# TODO: Remove Laboratory.experiment, then rename lab.experiment_ids to experiments

# TODO: Failed experiments should have its own section in the Laboratory.  Maybe
# organize them by the exception type & string?

# TODO: How hard would it be to allow the user to edit the lab report while the
# program is running?  Read timestamps to detect user writes.  All of the latest
# data is loaded in the program, so it should be simple to load in the new
# version and merge the human readable text into the latest data, write out to
# new file and attempt to swap it into place.

# TODO: Experiment if all of the parameters are modified, show the
# parameters instead of the modifications.  This is useful for swarming which
# touches every parameter.

# TODO: Add a field to each experiment summary stating how many times it has
# crashed?

import argparse
import os
import sys
import shutil
import random
import time
import datetime
import tempfile
import threading
from multiprocessing import Process, Pipe
import resource
import re
import numpy as np
import scipy.stats

from htm.optimization.parameter_set import ParameterSet

acceptable_exceptions = [
    TypeError,
    ValueError,
    MemoryError,
    ZeroDivisionError,
    AssertionError,
    RuntimeError,]

class Experiment:
    """
    An experiment represents a unique ParameterSet.
    This class primarily deals with bookkeeping.

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
        if "Notes:" in string:
            string, _, self.notes = string.partition('Notes:')
        # Reconstruct the parameters.
        self.modifications = []
        for change in re.findall(r"^[Mm]od.*:(.*)$", string, re.MULTILINE):
            path, eq, value = change.partition('=')
            self.modifications.append((path, value))
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

    def significance(self, baseline=None):
        """
        Returns the P-Value of the Null-Hypothesis test, the probability that
        this experiment and the given experiment have the same distribution of
        scores, meaning that the change in scores is merely by chance.

        Argument baseline is an Experiment, optional defaults to default_parameters
        """
        if baseline is None:
            baseline = self.lab.default_parameters
        baseline = self.lab.get_experiment( baseline )

        if not self.scores or not baseline.scores:
            return float('nan')
        if len(self.scores) == 1:
            pass # TODO: How to pass probabilities & statistics?
        stat, pval = scipy.stats.ttest_ind(
            baseline.scores, self.scores, axis=None,
            # Since both samples come from the same experimental setup  they
            # should have the same variance.
            equal_var=True,)
        return pval

    def mean(self):
        """ Returns the average score. """
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
        if self.scores:
            s += 'Scores: %s\n'%', '.join(str(s) for s in sorted(self.scores))
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
        if isinstance( parameters, Experiment ):
            return parameters

        p = ParameterSet( parameters ).typecast( self.structure )
        h = hash(p)
        if h in self.experiment_ids:
            return self.experiment_ids[h]
        else:
            return Experiment(self, parameters=p)

    def significant_experiments_table(self):
        """ Returns string """
        ex = sorted(self.experiments, key = lambda x: (-x.mean(), -x.attempts))
        ex = ex[:20]
        # Always keep the default parameters on the leader board.
        if self.default_parameters not in (X.parameters for X in ex):
            ex.pop()
            ex.append( self.get_experiment( self.default_parameters))
        s = '    Hash |   N |      Score |   P-Value | Modifications\n'
        fmt = '%8X | %3d | % 10g | % 9.3g | '
        for x in ex:
            s += fmt%(hash(x), len(x.scores), x.mean(), x.significance(ex[0]))
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
        s += str(self.default_parameters)
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
        pool = []
        while True:
            # Start running new experiments
            while len(pool) < processes:
                X = self.get_experiment( self.method.suggest_parameters() )
                trial = Worker(self, X.parameters, time_limit, memory_limit)
                trial.start()
                pool.append(trial)

            # Wait for experiments to complete.
            time.sleep(2)

            # Check for jobs which have finished.
            for idx in range(len(pool)-1, -1, -1):
                if not pool[idx].is_alive():
                    trial = pool.pop( idx )
                    X = self.get_experiment( trial.parameters )
                    trial.collect_journal( X )
                    trial.collect_score( X )
                    # Notify the parameter optimization method that the
                    # parameters which it suggested have finished evaluating.
                    self.method.collect_results( X.parameters, trial.score )
                    self.save()     # Write the updated Lab Report to file.


class Worker(Process):
    """
    This class runs a single trial of an experiment.
    Each trial is run in a subprocess.
    """
    def __init__(self, lab, parameters, time_limit, memory_limit):
        Process.__init__(self)
        self.parameters   = parameters
        self.time_limit   = time_limit
        self.memory_limit = memory_limit
        self.journal  = tempfile.NamedTemporaryFile(
            mode      = 'w+t',
            delete    = False,
            buffering = 1,
            dir       = lab.ae_directory,
            prefix    = "%X_"%hash(parameters),
            suffix    = ".tmp",).name
        # Make pipe to return outputs/results from worker back to main AE process.
        self.output, self.input = Pipe()
        # Worker will execute this string.
        self.exec_str = (lab.module_reload + 
            'score = %s.main(parameters=%s, argv=[%s], verbose=%s)'%(
                lab.name,
                repr(parameters),
                ', '.join(repr(arg) for arg in lab.argv[1:]),
                str(lab.verbose)))

    def start(self):
        Process.start(self)
        # Setup time limit, arm the watchdog timer.
        if self.time_limit is not None:
            def watchdog():
                if self.is_alive():
                    self.terminate()
            threading.Timer( self.time_limit * 60, watchdog ).start()

    def run(self):
        # Redirect stdout & stderr to the temporary log file.
        sys.stdout = open(self.journal, 'a', buffering=1)
        sys.stderr = open(self.journal, 'a', buffering=1)
        start_time = time.time()
        print("Started: " + time.asctime( time.localtime(start_time) ) + '\n')
        # Setup memory limit
        if self.memory_limit is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, hard))

        exec_globals = {}
        try:
            exec(self.exec_str, exec_globals)
        except Exception as err:
            exec_globals['score'] = err

        run_time = datetime.timedelta(seconds = time.time() - start_time)
        print("Elapsed Time: " + str(run_time))
        self.input.send( exec_globals['score'] )

    def is_alive(self):
        """
        If a process has reported a score then it is done, even if it's
        technically still alive.  Sometimes processes just don't die.  After a
        process reported back to us, it has 60 seconds to finish before we kill
        it.
        """
        if not Process.is_alive(self):
            return False
        if self.output.poll(0):
            def watchdog():
                if self.is_alive():
                    print("Warning: experiment returned but process still alive, terminating ...")
                    self.terminate()
            threading.Timer( 60, watchdog ).start()
            return False
        return True

    def collect_journal(self, experiment):
        """ Append the text output of this run to the main journal for the experiment. """
        # Append the temporary journal file to the experiments journal.
        with open( self.journal ) as journal:
            content = journal.read()
        with open( experiment.journal, 'a') as experiment_journal:
            experiment_journal.write(Laboratory.section_divider)
            experiment_journal.write(content)
        os.remove( self.journal )
        self.journal = content

    def collect_score(self, experiment):
        """
        Get the score returned by this run, saved as attribute 'score'.
        Score may be an exception raised by the experiment.
        """
        experiment.attempts += 1
        if self.output.poll(0):
            self.score = self.output.recv()
            if not isinstance(self.score, Exception):
                experiment.scores.append(self.score)
            else:
                print("")
                print("Parameters", str( experiment.parameters ))
                print("Hash: %X" % hash( experiment.parameters ))
                print("")
                print("%s:"%(type( self.score).__name__),  self.score)
                print("")
                for err in acceptable_exceptions:
                    if isinstance( self.score, err ):
                        break
                else:
                    print("Unhandled Exception, Exit.")
                    sys.exit(1)
        else:
            # No output from python?  Something went very wrong!
            print( self.journal )
            print("Error, Exit.")
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
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

    import htm.optimization.optimizers as optimizers
    from htm.optimization.swarming import ParticleSwarmOptimization
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
