# Parameter optimization

Parameter optimization is a technique for tuning model performance by finding optimal parameters 
of a model for given task/dataset. 
In *HTM.core* it is provided in python by `python -m htm.optimization.ae `. 

There are several methods implemented:
- particle swarm optimization (PSO) by `swarming.py`
- manual search
- exhaustive search

## Requirements

- Python 3
- `ExperimentModule.py` is your python script containing the model to be optimized as
   well as code to evaluate the models performance.
- `ExperimentModule.default_parameters = {}`
   Global dictionary containing all of the parameters to modify.
   Parameters must be one of the following types: dict, tuple, float, int.
   Parameters can be nested in multiple levels of dictionaries and tuples.
   For more details see `htm.optimization.parameter_set`
- `ExperimentModule.main(parameters=default_parameters, argv=None, verbose=True)`
   Returns (float) performance of parameters, to be maximized.
   For example, see file: `py/htm/examples/mnist.py`

## Optimize your model, parameter tuning

Run your experiment with the AE program:
`$ python3 -m htm.optimization.ae [ae-arguments] ExperimentModule.py [experiment-arguments]`

### Examples

`python -m htm.optimization.ae -n 7 --memory_limit 14 -v --swarming 100  mnist.py`

Uses PSO ("swarming") with 100 particles, 7 CPU threads, upto 14GB RAM, to optimize MNIST results in mnist.py.

`python -m htm.optimization.ae --grid_search "" -n 7 --memory_limit 14 mnist.py` 

Does full exhaustive search for all parameters (the `""`) on `mnist.py` script.

`python -m htm.optimization.ae --grid_search '["potentialPct"]' -n 7 --memory_limit 14 mnist.py`

Tries several different values for the "potentialPct" parameter.

### Manual search

It is possible to manually specify parameters to evaluate.  While the AE program
is **not** running, append the following to the lab-report file:

```
================================================================================
Modification: ['potentialPct'] = 0.66
Modification: ['columnDimensions'][1] = 123
```

The 80 equal signs are significant!  The AE program uses them as section dividers.

Each "modification" line is a snippet of python code which is applied to the
default parameters.

Then run the command: `python -m htm.optimization.ae --parse mnist.py`
which will read the lab-report file, fill in any missing fields, and write it
back to the same file, without running any experiments.  After running the AE
program, the section which we appended to the lab-report will look like:

```
================================================================================
Modification: ['columnDimensions'][1] = 123
Modification: ['potentialPct'] = 0.66
Hash: FFD67F4E
Journal: mnist_ae/FFD67F4E.journal
Attempts: 0
Notes: 
```

Notice the "Hash" has been computed for this set of parameters.  The Hash is a
unique identifier for these parameters.  Use the following command to evaluate
these parameters:

`python -m htm.optimization.ae --hashes FFD67F4E mnist.py`

The option `--hashes` also accepts a comma separated list of hashes.


### Correct experiment methodology, unbiased results

Remember, in order not to bias your experiments during the optimization phase, the dataset needs to be split into 
`train/evaluation/test` parts. Where optimization trains only on `train`, and is scored (evaluated) on `eval`. Never touching the
out of sample `test` (true test set) data. Only run on these data once the optimization is finished and you don't modify the model afterwards. 

#### Notes

You may want to "lock" some of the parameters to exclude them from the search. For example parameters 
that are dependent on each other (optimize only one of the dependency pair). That helps to reduce the parameter
search-space, resulting in faster optimization times and possibly obtaining better results. 


### Results

The outputs and data of this program are kept in a directory named after the
experiment which generated it.  If the experiment is "foo/bar.py" then AE's
directory is "foo/bar_ae/".  The primary output of this program is the file
"foo/bar_ae/lab_report.txt" which contains a summary of its operations.  

The lab report format is:
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

## Creating new search methods

It is possible to implement another black-box methods for parameter optimization. 
For example:
- simulated annealing
- genetic algorithms (GA)
- random search

To do so, extend from `BaseOptimizer`. 


