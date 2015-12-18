@0xc7158cb7b34ac59a;

# Next ID: 6
struct BitHistoryProto {
  # This is currently unused and should probably be removed.
  id @0 :Text;
  stats @1 :List(Stat);
  lastTotalUpdate @2 :UInt32;
  learnIteration @3 :UInt32;
  alpha @4 :Float64;
  verbosity @5 :UInt8;

  # Next ID: 2
  struct Stat {
    index @0 :UInt32; # Check if actually unsigned
    dutyCycle @1 :Float64;
  }
}
