# C++ example using Network API
The program `hotgym_napi` is an example of an app using the Network API tools available in the htm.core library.  In this example we generate a sin wave with some noise as the input.  This is passed to an encoder to turn that into SDR format.  This is passed to two instances of SpatialPooler (SP), one is configured for local inhibition and one for global inhibition.  The output of the SP for global inhibition is passed on to the temporalMemory (TM) algorithm.  The output of the TM can be written to a file so that it can be plotted.

```
  ///////////////////////////////////////////////////////////////
  //
  //                 .------------------.
  //                 |    encoder       |
  //         data--->|  (RDSERegion)    |
  //                 |                  |
  //                 `------------------'
  //                     |           |
  //      .------------------.    .------------------.
  //      |   SP (local)     |    |   SP (global)    |
  //      |    (SPRegion)    |    |   (SPRegion)     |
  //      |                  |    |                  |
  //      `------------------'    `------------------'
  //                                       |
  //                              .------------------.
  //                              |      TM          |
  //                              |   (TMRegion)     |
  //                              |                  |
  //                              `------------------'
  //
  //////////////////////////////////////////////////////////////////
```

Each "region" is a wrapper around an algorithm.  This wrapper provides a uniform interface that can be plugged into the Network API engine for execution. The htm.core library contains regions for each of the primary algorithms in the library. The user can create their own algorithms and corresponding regions and plug them into the Network API engine by registering them with the Network class.  The following chart shows the 'built-in' C++ regions.  
<table>
<thead>
	<tr>
		<th>Built-in Region</th>
		<th>Algorithm</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td>ScalarSensor</td>
		<td>ScalarEncoder;  Original encoder for numeric values</td>
	</tr>
	<tr>
		<td>RDSERegion</td>
		<td>RandomDistributedScalarEncoder (RDSE);  advanced encoder for numeric values.</td>
	</tr>
	<tr>
		<td>SPRegion</td>
		<td>SpatialPooler (SP)</td>
	</tr>
	<tr>
		<td>TMRegion</td>
		<td>TemporalMemory (TM)</td>
	</tr>
	<tr>
		<td>VectorFileSensor</td>
		<td>for reading from a file</td>
	</tr>
	<tr>
		<td>VectorFileEffector</td>
		<td>for writing to a file</td>
	</tr>
</tbody>
</table>

## Usage

```
   hotgym_napi  [iterations [filename]]
```
- *iterations* is the number of times to execute the regions configured into the network. The default is 5000.
- *filename* is the path for a file to be written which contains the following for each iteration.  The default is no file written.
```
        <iteration>, <sin data>, <anomaly>\n
```

## Plotting

The `hotgym_napi` program can output data if a filename is specified. Any plotting program can be used to display this CSD data.

## Experimentation
It is intended that this program be used as a launching point for experimenting with combinations of the regions and using different parameters.  Try it and see what happens...