@0x96d695b1ca7f9979;

# Next ID: 16
struct SdrClassifierProto {
  steps @0 :List(UInt16);
  alpha @1 :Float64;
  actValueAlpha @2 :Float64;
  learnIteration @3 :UInt32;
  recordNumMinusLearnIteration @4 :UInt32;
  recordNumMinusLearnIterationSet @5 :Bool;
  maxSteps @6 :UInt32;
  patternNZHistory @7 :List(List(UInt32));
  iterationNumHistory @8 :List(UInt32);
  weightMatrix @9 :List(StepWeightMatrix);
  maxBucketIdx @10 :UInt32;
  maxInputIdx @11 :UInt32;
  actualValues @12 :List(Float64);
  # Each index is true if the corresponding index in actualValues has been set.
  actualValuesSet @13 :List(Bool);
  version @14 :UInt16;
  verbosity @15 :UInt8;

  # Next ID: 2
  struct StepWeightMatrix {
    steps @0 :UInt32;
    weight @1 :List(Float64);
  }
}
