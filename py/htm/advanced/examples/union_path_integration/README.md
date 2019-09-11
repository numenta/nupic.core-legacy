# Multi-column experiment

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



Use the following command to run all **L2-L4-L6a** experiments:

    python multi_column_convergence.py


The [./results](./results) directory will contain the charts and raw data used to create the charts.
See [experiments.cfg](experiments.cfg) for details on the parameters used by the experiments.

--------------------------------------------------------------------------------
> This experiment is based on [PyExperimentSuite](https://github.com/rueckstiess/expsuite).
> Numenta keep a local copy in `htmresearch/support/expsuite.py`.
> For more details on how to configure the experiments see [PyExperimentSuite Documentation](https://github.com/rueckstiess/expsuite/blob/master/documentation.pdf)  
> For a local copy see see documentation.pdf
