@0xf2eaadec04697984;

# Next ID: 2
struct InSynapseProto {
  srcCellIdx @0 :UInt32;
  permanence @1 :Float32;
}

# Next ID: 9
struct SegmentProto {
  seqSegFlag @0 :Bool;
  frequency @1 :Float32;
  nConnected @2 :UInt32;
  totalActivations @3 :UInt32;
  positiveActivations @4 :UInt32;
  lastActiveIteration @5 :UInt32;
  lastPosDutyCycle @6 :Float32;
  lastPosDutyCycleIteration @7 :UInt32;
  synapses @8 :List(InSynapseProto);
}

# Next ID: 5
struct CStateProto {
  version @0 :UInt16;
  fMemoryAllocatedByPython @1 :Bool;
  # length of list is stored as nCells in class
  pData @2 :Data;

  # CStateIndexed additional variables. If not using CStateIndexed, these
  # values should be ignored.
  countOn @3 :UInt32;
  cellsOn @4 :List(UInt32);
}
