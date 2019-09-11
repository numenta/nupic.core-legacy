


A simple experiment to compute the object classification accuracy of L2-L4-L6 network using objects from [YCB dataset](http://www.ycbbenchmarks.com/) and "Thing" sensor
Create a network consisting of multiple columns to run object recognition experiments. 

Each column contains one L2, one L4 and one L6a layers. In addition all the L2 columns are fully connected to each other through their lateral inputs.


                                +---lateralInput----+
                                | +---------------+ |
                                | |      +1       | |
    Phase                       v |               v |
    -----                    +-------+         +-------+
                      reset  |       |         |       | reset
    [3]               +----->|  L2   |         |  L2   |<----+
                      |      |       |         |       |     |
                      |      +-------+         +-------+     |
                      |        |   ^             |   ^       |
                      |     +1 |   |          +1 |   |       |
                      |        |   |             |   |       |
                      |        v   |             v   |       |
                      |      +-------+         +-------+     |
    [2]         +----------->|       |         |       |<----------+
                |     |      |  L4   |         |  L4   |     |     |
                |     +----->|       |         |       |<----+     |
                |     |      +-------+         +-------+     |     |
                |     |        |   ^             |   ^       |     |
                |     |        |   |             |   |       |     |
                |     |        |   |             |   |       |     |
                |     |        v   |             v   |       |     |
                |     |      +-------+         +-------+     |     |
                |     |      |       |         |       |     |     |
    [1,3]       |     +----->|  L6a  |         |  L6a  |<----+     |
                |     |      |       |         |       |     |     |
                |     |      +-------+         +-------+     |     |
           feature  reset        ^                 ^       reset  feature
                |     |          |                 |         |     |
                |     |          |                 |         |     |
    [0]     [sensorInput]  [motorInput]      [motorInput] [sensorInput]

Data
====

The data folder contains descriptions of the objects and their locations used in the experiment.
There is one file per object, each row contains one feature, location pairs. 

The format is as follows:  
        `[(-33.6705, 75.5003, 2.4207)/10] => [[list of active bits of location], [list of active bits of feature]]` 
        
The content before "=>" is the true 3D location / sensation
We ignore the encoded values after "=>" and use :class:`ScalarEncoder` to
encode the sensation in a way that is compatible with the experiment network.

Running
=======

Use the following command to run all **L2-L4-L6a** experiments:

    python l2l4l6_experiment.py


The [./results](./results) directory will contain the charts and raw data used to create the charts.
See [experiments.cfg](experiments.cfg) for details on the parameters used by the experiments.

--------------------------------------------------------------------------------
> This experiment is based on [PyExperimentSuite](https://github.com/rueckstiess/expsuite).
> Numenta keep a local copy in `htmresearch/support/expsuite.py`.
> For more details on how to configure the experiments see [PyExperimentSuite Documentation](https://github.com/rueckstiess/expsuite/blob/master/documentation.pdf)  
> For a local copy see see documentation.pdf in `union_path_integration`
