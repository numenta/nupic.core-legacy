@0xbec20a8f487acf07;

using import "LinkProto.capnp".LinkProto;
using import "Map.capnp".Map;
using import "RegionProto.capnp".RegionProto;

struct NetworkProto {
  regions @0 :Map(Text, RegionProto);
  links @1 :List(LinkProto);
}
