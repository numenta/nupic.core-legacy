#ifndef NTA_PTR_TYPES_HPP
#define NTA_PTR_TYPES_HPP

#include <memory>

namespace nupic
{
    class Link;
    class Region;

    typedef std::shared_ptr<Link> Link_Ptr_t;
    typedef std::shared_ptr<Region> Region_Ptr_t;

} // namespace nupic

#endif // NTA_PTR_TYPES_HPP
