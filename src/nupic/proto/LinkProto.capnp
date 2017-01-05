@0x9d7ec2149bcaf713;

using import "/nupic/proto/ArrayProto.capnp".ArrayProto;

# Next ID: 9
struct LinkProto {
  type @0 :Text;
  params @1 :Text;

  srcRegion @2 :Text;
  srcOutput @3 :Text;

  destRegion @4 :Text;
  destInput @5 :Text;

  outputElementType @6 :Int32; # NTA_BasicType: type of output elements in output array
  outputElementCount @7 :UInt32; # number of output elements in output array
  delayedOutputs @8 :List(ArrayProto);
}
