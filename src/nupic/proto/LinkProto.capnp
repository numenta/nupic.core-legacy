@0x9d7ec2149bcaf713;

using import "/nupic/proto/ArrayProto.capnp".ArrayProto;

# Next ID: 7
struct LinkProto {
  type @0 :Text;
  params @1 :Text;

  srcRegion @2 :Text;
  srcOutput @3 :Text;

  destRegion @4 :Text;
  destInput @5 :Text;

  delayedOutputs @6 :List(ArrayProto);
}
