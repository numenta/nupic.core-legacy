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
#ifndef NTA_YAML_HPP
#define NTA_YAML_HPP

#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Collection.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/types/Types.hpp>

namespace nupic {

namespace YAMLUtils {
/*
 * For converting default values
 */
Value toValue(const std::string &yamlstring, NTA_BasicType dataType);

/*
 * For converting param specs for Regions and LinkPolicies
 */
ValueMap toValueMap(const char *yamlstring,
                    Collection<ParameterSpec> &parameters,
                    const std::string &nodeType = "",
                    const std::string &regionName = "");

} // namespace YAMLUtils
} // namespace nupic

#endif //  NTA_YAML_HPP
