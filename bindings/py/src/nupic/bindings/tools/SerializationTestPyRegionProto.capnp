@0x8ae35855f4636f7b;

using import "/nupic/proto/RandomProto.capnp".RandomProto;

struct SerializationTestPyRegionProto {
  dataWidth @0 :UInt32;

  random @1 :RandomProto;
}

