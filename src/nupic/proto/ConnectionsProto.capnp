@0xb1b8a459d70716ad;

# Next ID: 2
struct ConnectionsProto {

  # Next ID: 2
  struct SynapseProto {
    presynapticCell @0 :UInt32;
    permanence @1 :Float32;
  }

  # Next ID: 1
  struct SegmentProto {
    synapses @0 :List(SynapseProto);
  }

  # Next ID: 1
  struct CellProto {
    segments @0 :List(SegmentProto);
  }

  cells @0 :List(CellProto);
  version @1 :UInt16;
}
