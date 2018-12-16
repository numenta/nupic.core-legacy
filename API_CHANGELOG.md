#Breaking changes to the nupic API

* CapnProto serialization is replaced with binary streams in PR #62.  
Calls to read() and write() are no longer available. Use save() and load(). Network(path) is no 
longer used to deserialize a path. Use Network net; net.load(stream); to deserialize.  
Helpers SaveToFile(path) and LoadFromFile(path) are used to stream to and from a file using save() 
and load().

* The function Network::newRegionFromBundle() was replaced with newRegion(stream, name) where the stream 
is an input stream reading a file created by region->save(steam)  or region->saveToFile(path).  PR#62

* Removed methods SpatialPooler::setSynPermTrimThreshold & SpatialPooler::getSynPermTrimThreshold.
Synapse trimming was an optimization which is no longer possible because of an implementation change.

* Removed method SpatialPooler::setSynPermMax as the maximum permanence is now defined (hardcoded) as
nupic::algorithms::connections::maxPermancence = 1.0f;

* Changed callback ConnectionsEventHandler::onUpdateSynapsePermanence().  Instead of being called
every time a synapses permanence changes, it is now called when a synapse changes connected state,
IE: it is called when a synapses permanence crosses the connected threshold.
