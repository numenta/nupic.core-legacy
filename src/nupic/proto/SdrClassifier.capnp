@0x96d695b1ca7f9979;

# Next ID: 13
struct SdrClassifierProto {
  steps @0 :List(UInt16);
  alpha @1 :Float64;
  actValueAlpha @2 :Float64;
  maxSteps @3 :UInt32;
  patternNZHistory @4 :List(List(UInt32));
  recordNumHistory @5 :List(UInt32);
  weightMatrix @6 :List(StepWeightMatrix);
  maxBucketIdx @7 :UInt32;
  maxInputIdx @8 :UInt32;
  actualValues @9 :List(Float64);
  # Each index is true if the corresponding index in actualValues has been set.
  actualValuesSet @10 :List(Bool);
  version @11 :UInt16;
  verbosity @12 :UInt8;

  # Next ID: 2
  struct StepWeightMatrix {
    steps @0 :UInt32;
    # weight matrices are flattened before serialization
    weight @1 :List(Float64);
  }
}
