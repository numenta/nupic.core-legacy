#Breaking changes to the nupic API

* CapnProto serialization is replaced with binary streams in PR #62.  
Calls to read() and write() are no longer available. Use save() and load(). Network(path) is no 
longer used to deserialize a path. Use Network net; net.load(stream); to deserialize.  
Helpers SaveToFile(path) and LoadFromFile(path) are used to stream to and from a file using save() 
and load().

* The function Network::newRegionFromBundle() was replaced with newRegion(stream, name) where the stream 
is an input stream reading a file created by region->save(steam)  or region->saveToFile(path).  PR#62
