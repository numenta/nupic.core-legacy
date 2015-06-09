@0xdc5b07d7180bb807;

struct Map(Key, Value) {
  entries @0 :List(Entry);

  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}
