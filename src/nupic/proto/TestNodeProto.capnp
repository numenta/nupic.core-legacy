@0xacbfb584b84b791c;

# Next ID: 18
struct TestNodeProto {
  int32Param @0 :Int32;
  uint32Param @1 :UInt32;
  int64Param @2 :Int64;
  uint64Param @3 :UInt64;
  real32Param @4 :Float32;
  real64Param @5 :Float64;
  boolParam @16 :Bool;
  stringParam @6 :Text;

  real32ArrayParam @7 :List(Float32);
  int64ArrayParam @8 :List(Int64);
  boolArrayParam @17 :List(Bool);

  iterations @9 :UInt32;
  outputElementCount @10 :UInt32;
  delta @11 :Int64;

  shouldCloneParam @12 :Bool;
  unclonedParam @13 :List(UInt32);
  unclonedInt64ArrayParam @14 :List(List(Int64));

  nodeCount @15 :UInt32;
}
