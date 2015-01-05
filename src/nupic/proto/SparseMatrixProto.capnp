@0xeefc26a597cc6a4e;

# Next ID: 3
struct SparseMatrixProto {
  numRows @0 :UInt32;
  numColumns @1 :UInt32;
  rows @2 :List(SparseFloatList);
}

# Next ID: 1
struct SparseFloatList {
  values @0 :List(SparseFloat);
}

# Next ID: 2
struct SparseFloat {
  index @0 :UInt32;
  value @1 :Float32;
}
