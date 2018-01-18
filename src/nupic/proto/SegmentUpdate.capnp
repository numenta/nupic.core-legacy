@0xe7a17ea776969175;

# Next ID: 7
struct SegmentUpdateProto {
  sequenceSegment @0 :Bool;
  cellIdx @1 :UInt32;
  segIdx @2 :UInt32;
  timestamp @3 :UInt32;
  synapses @4 :List(UInt32);
  phase1Flag @5 :Bool;
  weaklyPredicting @6 :Bool;
}
