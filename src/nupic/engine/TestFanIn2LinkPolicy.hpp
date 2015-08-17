/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */


#ifndef NTA_TESTFANIN2LINKPOLICY_HPP
#define NTA_TESTFANIN2LINKPOLICY_HPP

#include <string>
#include <nupic/engine/Link.hpp>
#include <nupic/ntypes/Dimensions.hpp>

namespace nupic
{

  class Link;

  class TestFanIn2LinkPolicy : public LinkPolicy
  {
  public:
    TestFanIn2LinkPolicy(const std::string params, Link* link);

    ~TestFanIn2LinkPolicy();

    void setSrcDimensions(Dimensions& dims) override;

    void setDestDimensions(Dimensions& dims) override;
  
    const Dimensions& getSrcDimensions() const override;

    const Dimensions& getDestDimensions() const override;

    void buildProtoSplitterMap(Input::SplitterMap& splitter) const override;

    void setNodeOutputElementCount(size_t elementCount) override;

    void initialize() override;

    bool isInitialized() const override;

private:
    Link* link_;
    
    Dimensions srcDimensions_;
    Dimensions destDimensions_;

    size_t elementCount_;

    bool initialized_;


  }; // TestFanIn2

} // namespace nupic


#endif // NTA_TESTFANIN2LINKPOLICY_HPP
