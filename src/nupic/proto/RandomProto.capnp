@0x88eaee6db8e1f67c;

struct RandomProto {
  seed @0 :UInt64;
  impl @1 :RandomImplProto;
}

struct RandomImplProto {
  state @0 :List(Int64);
  rptr @1 :Int64;
  fptr @2 :Int64;
}
