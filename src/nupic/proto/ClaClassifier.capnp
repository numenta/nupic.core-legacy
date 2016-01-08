@0xc8e1bdd87a2feed9;

using import "/nupic/proto/BitHistory.capnp".BitHistoryProto;

# Next ID: 15
struct ClaClassifierProto {
  steps @0 :List(UInt16);
  alpha @1 :Float64;
  actValueAlpha @2 :Float64;
  learnIteration @3 :UInt32;
  recordNumMinusLearnIteration @4 :UInt32;
  recordNumMinusLearnIterationSet @5 :Bool;
  maxSteps @6 :UInt32;
  patternNZHistory @7 :List(List(UInt32));
  iterationNumHistory @8 :List(UInt32);
  activeBitHistory @9 :List(StepBitHistories);
  maxBucketIdx @10 :UInt32;
  actualValues @11 :List(Float64);
  # Each index is true if the corresponding index in actualValues has been set.
  actualValuesSet @12 :List(Bool);
  version @13 :UInt16;
  verbosity @14 :UInt8;

  # Next ID: 2
  struct StepBitHistories {
    steps @0 :UInt32;
    bitHistories @1 :List(IndexBitHistory);

    # Next ID: 2
    struct IndexBitHistory {
      index @0 :UInt32;
      history @1 :BitHistoryProto;
    }
  }
}
