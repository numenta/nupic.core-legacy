@0xcd9e11090abd44ca;

# Next ID: 3
struct SparseBinaryMatrixProto {
  numRows @0 :UInt32;
  numColumns @1 :UInt32;
  indices @2 :List(List(UInt32));
}
