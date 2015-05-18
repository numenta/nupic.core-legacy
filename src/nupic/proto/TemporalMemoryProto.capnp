@0xc5bf8243b0c10764;

# TODO: Use absolute path
using import "ConnectionsProto.capnp".ConnectionsProto;
using import "RandomProto.capnp".RandomProto;

# Next ID: 16
struct TemporalMemoryProto {

  columnDimensions @0 :List(UInt32);
  cellsPerColumn @1 :UInt32;
  activationThreshold @2 :UInt32;
  initialPermanence @3 :Float32;
  connectedPermanence @4 :Float32;
  minThreshold @5 :UInt32;
  maxNewSynapseCount @6 :UInt32;
  permanenceIncrement @7 :Float32;
  permanenceDecrement @8 :Float32;

  connections @9 :ConnectionsProto;
  random @10 :RandomProto;

  # Lists of indices
  activeCells @11 :List(UInt32);
  predictiveCells @12 :List(UInt32);
  activeSegments @13 :List(UInt32);
  winnerCells @14 :List(UInt32);

}
