# Breaking changes to the NuPIC API

We try to keep the API compatible with the original [numenta/nupic.core repo](https://github.com/numenta/nupic.core). 
That API is specified in their [API Docs](http://nupic.docs.numenta.org/prerelease/api/index.html).

## Motivation

We try to remain compatible where possible, to make it easy for the users and programmers to switch/use 
our implementation. And for developers to be easily able to navigate within the (known) codebase. 
Despite of this, sometimes changes need to happen - be it for optimization, removal/replacement of some 
features or implementation detail, etc.

## Summary

The NetworkAPI is mostly unchanged, since it is a compatibility layer.
Direct access to the algorithms APIs has changed:
* All classes now use the `SDR` class where applicable instead of raw data lists.
* All encoders have new API.
* `SDRClassifier` split into two classes: `Classifier` and `Predictor` with new API.
* `Anomaly` class is now built into the `TemporalMemory`
* `SpatialPooler` & `TemporalMemory` have many small changes, see below.

## API breaking changes in this repo

Compared to `Numenta/nupic.core`; the changes here are listed in order from oldest to newest (at the bottom). 

* CapnProto serialization is replaced with binary streams in PR #62.  
Calls to `read()` and `write()` are no longer available. Use `save()` and `load()`. 
`Network(path)` is no longer used to deserialize a path. Use `Network net; net.load(stream);` to deserialize.  
Helpers `SaveToFile(path)` and `LoadFromFile(path)` are used to stream to and from a file using save() 
and load().

* The function `Network::newRegionFromBundle() was replaced with `newRegion(stream, name)` where the stream 
is an input stream reading a file created by region->save(steam)  or region->saveToFile(path).  PR#62

* Removed methods `SpatialPooler::setSynPermTrimThreshold` and  `SpatialPooler::getSynPermTrimThreshold`.
Synapse trimming was an optimization which is no longer possible because of an implementation change. PR #153

* Removed method `SpatialPooler::setSynPermMax` as the maximum permanence is now defined (hardcoded) as
`nupic::algorithms::connections::maxPermancence = 1.0f;` PR #153

* Changed callback `ConnectionsEventHandler::onUpdateSynapsePermanence()`.  Instead of being called
every time a synapses permanence changes, it is now called when a synapse changes connected state,
IE: it is called when a synapses permanence crosses the connected threshold. PR #153

* SpatialPooler now always applies boosting, even when `learn=false`. PR #206

* Removed methods `SpatialPooler::setSynPermConnected` and `TemporalMemory::setConnectedPermanence`. 
  The connected synapse permanence threshold should instead be given to the constructor or the initialize method. PR #221

* When building with MS Visual Studio 2017, it will build bindings only for Python 3.4 and above.  
  (i.e. No Python 2.7 under Windows)

* Setting dimensions on a region is now optional.  If given, it overrides any region parameters that set 
  the width of the default output buffer.

* The splitter maps (and the LinkPolicy that creates them) were removed.  These were used as a way to 
re-arrange bits in the input buffer based on user defined patterns. However, thinking about how this 
works in biology, the neurons make connections to the synapse of other neurons. There is no order or 
pattern involved and in fact it needs to be fairly random as to how the neurons connect. It is the 
job of the Spatial Pooler to simulate that set of random connections. So in effect, any re-mapping 
of bits by a splitter map prior to being presented to the Spatial Pooler should have no affect on 
functionality. This is probably why this feature was not used anyplace except in the unit tests.
As a side-effect of this change, the LinkType and LinkParam parameters in the Network.Link( ) call 
are ignored.  PR #271

* Removed all matrix libraries.  Use the `Connections` class instead.  PR #169

* Removed `void SpatialPooler::stripUnlearnedColumns()` as unused and not useful (did not effectively remove any columns). PR #286 

* Rewrote ScalarEncoder API, all code using it needs to be rewritten. PR #314

* Removed old `TP` (Temporal Pooler, `Cells4.hpp`) as it was not maintained, users should default to `TemporalMemory, TM`. 
  With this we are also removing `BacktrackingTM` (and its NetworkAPI Region), which was based on TP. BackTM had slightly better
  anomaly scores results (+5% compared to TM), but did not have complete tests and verified (peer-reviewed) functionality. PR #356

* Connections class must be initialized with a connectedPermanence.  Methods
`Connections::computeActivity` and `Connections::raisePermanencesToThreshold` no
longer accept a synapse permanence threshold argument. PR #305

* SDRClassifier class is replaced by `Classifier` and `Predictor` classes.

* In NetworkAPI, access to a Region object was accessed using `net.getRegions()->getByName('name');`. 
This is obsolete. Use getRegion('name') instead. 

* Anomaly class removed as obsolete, use `TM.anomaly` which is simpler to use, and `MovingAverage` when you need to emulate 
  running averages. Internally the code still uses `computeRawAnomalyScore()` but there's no need to call it directly. `AnomalyLikelihood` 
  is still available and can be used in addition to TM.getAnomalyScore(). PR #406 

* TemporalMemory::getPredictiveCells() now returns a SDR. This ensures more convenient API and that the SDR object has correct
  dimensions matching TM. use TM.getPredictiveCells().getSparse() to obtain the sparse vector as before. PR #437, #442 

* TemporalMemory `compute()` and `activateCells()` now use only SDR variants,
  old overloads with C-style arrays removed. Bindings and tests also updated.

* Changed all use of "nupic" to "htm".   This means that C++ users must include from

  | Currently                    | Previously                     |
  | :--------------------------- | :----------------------------  |
  | <htm/algorithms/*.hpp>       | <nupic/algorithms/*.hpp>       |
  | <htm/engine/*.hpp>           | <nupic/engine/*.hpp>           |
  | <htm/math/*.hpp>             | <nupic/math/*.hpp>             |
  | <htm/encoders/*.hpp>         | <nupic/encoders/*.hpp>         |
  | <htm/types/*.hpp>            | <nupic/types/*.hpp>            |

    We also renamed the namespaces from `namespace nupic` to `namespace htm`.

* SpatialPooler: removed param `numActiveColumnsPerInhArea`, as replaced by `localAreaDensity` which has better properties
  (constant sparsity). PR #TODO


## Python API Changes

Changes made to the C++ Library also effect the Python Library, since python is
mostly just a thin wrapper around the C++ library.

- `Serialization` not supported as canproto was removed. Serialization via Pickle is not yet supported.

- Changed all use of "nupic" to "htm".  This means that Python users must import from

  | Currently                    | Previously                     |
  | :--------------------------- | :----------------------------  |
  | htm.bindings.algorithms      | nupic.bindings.algorithms      |
  | htm.bindings.engine_internal | nupic.bindings.engine_internal |
  | htm.bindings.math            | nupic.bindings.math            |
  | htm.bindings.encoders        | nupic.bindings.encoders        |

- Most algorithms now accept SDR's instead of numpy arrays.
  Recommend reading the documentation, see `python -m pydoc htm`
