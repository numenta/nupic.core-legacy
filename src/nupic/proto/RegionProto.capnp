@0x8e18cab40b91663a;

# Next ID: 4
struct RegionProto {
  # The type of the region, a string identifier for the RegionImpl subclass.
  nodeType @0 :Text;

  # This stores the data for the RegionImpl. The nodeType field is necessary
  # when deserializing to know what schema struct to cast this as. This will
  # be a PyRegionProto instance if it is a PyRegion.
  regionImpl @1 :AnyPointer;

  dimensions @2 :List(UInt32);

  phases @3 :List(UInt32);
}
