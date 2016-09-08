@0xb1b8a459d70716ad;

# Next ID: 5
struct ConnectionsProto {

  # Next ID: 3
  struct SynapseProto {
    presynapticCell @0 :UInt32;
    permanence @1 :Float32;

    # Obsolete
    destroyed @2 :Bool;
  }

  # Next ID: 3
  struct SegmentProto {
    synapses @0 :List(SynapseProto);

    # Obsolete
    destroyed @1 :Bool;

    lastUsedIteration @2 :UInt64;
  }

  # Next ID: 1
  struct CellProto {
    segments @0 :List(SegmentProto);
  }

  # Next ID: 3
  struct SegmentOverlapProto {
    cell @0 :UInt32;
    segment @1 :UInt32;
    overlap @2 :UInt32;
  }

  cells @0 :List(CellProto);
  maxSegmentsPerCell @1 :UInt16;
  iteration @2 :UInt64;
  maxSynapsesPerSegment @3 :UInt16;
  version @4 :UInt16;

}
