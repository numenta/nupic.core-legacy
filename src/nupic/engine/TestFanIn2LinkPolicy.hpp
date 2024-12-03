/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef NTA_TESTFANIN2LINKPOLICY_HPP
#define NTA_TESTFANIN2LINKPOLICY_HPP

#include <nupic/engine/Link.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <string>

namespace nupic {

class Link;

class TestFanIn2LinkPolicy : public LinkPolicy {
public:
  TestFanIn2LinkPolicy(const std::string params, Link *link);

  ~TestFanIn2LinkPolicy();

  void setSrcDimensions(Dimensions &dims) override;

  void setDestDimensions(Dimensions &dims) override;

  const Dimensions &getSrcDimensions() const override;

  const Dimensions &getDestDimensions() const override;

  void buildProtoSplitterMap(Input::SplitterMap &splitter) const override;

  void setNodeOutputElementCount(size_t elementCount) override;

  void initialize() override;

  bool isInitialized() const override;

private:
  Link *link_;

  Dimensions srcDimensions_;
  Dimensions destDimensions_;

  size_t elementCount_;

  bool initialized_;

}; // TestFanIn2

} // namespace nupic

#endif // NTA_TESTFANIN2LINKPOLICY_HPP
