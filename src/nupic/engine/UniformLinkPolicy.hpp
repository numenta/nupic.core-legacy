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

#ifndef NTA_UNIFORMLINKPOLICY_HPP
#define NTA_UNIFORMLINKPOLICY_HPP

#include <string>
#include <vector>

#include <nupic/engine/Link.hpp>
#include <nupic/ntypes/Collection.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/types/Fraction.hpp>
#include <nupic/utils/Log.hpp>

// We use the ParameterSpec which is defined in the Spec header
#include <nupic/engine/Spec.hpp>

#include <boost/shared_ptr.hpp>

namespace nupic {
class Link;
class ValueMap;

// ---
// The UniformLinkPolicy implements a linkage structure between two Regions
// wherein the topology of the receptive fields are uniform*.
//
// * To be precise, this should say more-or-less uniform since we allow
//   strict uniformity to be disabled via parameter (in which case we build
//   a linkage "as close to uniform as possible").
//
// (Refer to GradedLinkPolicy and SparseLinkPolicy for examples of link
//  policies with non-uniform receptive field topologies.)
//
// In the simplest case, this is a direct one-to-one mapping (and
// consequently this allows for linkage of Region level inputs and outputs
// without specifying any parameters).  However, this can also take the form
// of more complex receptive field mappings as configured via parameters.
// ---

class UniformLinkPolicy : public LinkPolicy {
  // ---
  // We make our unit test class a friend so that we can test dimension
  // calculations and splitter map generation without requiring the rest of
  // the NuPIC infrastructure.
  // ---
  friend class UniformLinkPolicyInspector;

public:
  UniformLinkPolicy(const std::string params, Link *link);

  ~UniformLinkPolicy();

  // LinkPolicy Interface
  void setSrcDimensions(Dimensions &dims) override;
  void setDestDimensions(Dimensions &dims) override;
  const Dimensions &getSrcDimensions() const override;
  const Dimensions &getDestDimensions() const override;
  void setNodeOutputElementCount(size_t elementCount) override;
  void buildProtoSplitterMap(Input::SplitterMap &splitter) const override;
  void initialize() override;
  bool isInitialized() const override;

private:
  Link *link_;

  enum MappingType { inMapping, outMapping, fullMapping };

  enum GranularityType { nodesGranularity, elementsGranularity };

  enum OverhangType { nullOverhang = 0, wrapOverhang };

  MappingType mapping_;
  std::vector<Real64> rfSize_;
  std::vector<Real64> rfOverlap_;
  GranularityType rfGranularity_;
  std::vector<Real64> overhang_;
  std::vector<OverhangType> overhangType_;
  std::vector<Real64> span_;
  bool strict_;

  template <typename T> struct DefaultValuedVector : public std::vector<T> {
    typedef typename std::vector<T>::size_type size_type;
    DefaultValuedVector();
    T operator[](const size_type index) const;
    T &operator[](const size_type index);
    T at(const size_type index) const;
    T &at(const size_type index);
  };

  struct WorkingParameters {
    DefaultValuedVector<Fraction> rfSize;
    DefaultValuedVector<Fraction> rfOverlap;
    DefaultValuedVector<Fraction> overhang;
    DefaultValuedVector<OverhangType> overhangType;
    DefaultValuedVector<Fraction> span;
  };

  WorkingParameters workingParams_;

  void setValidParameters();
  void readParameters(const std::string &params);
  void validateParameterDimensionality();
  void validateParameterConsistency();
  void populateWorkingParameters();

  void copyRealVecToFractionVec(const std::vector<Real64> &sourceVec,
                                DefaultValuedVector<Fraction> &destVec);

  template <typename T>
  void populateArrayParamVector(std::vector<T> &vec, const ValueMap &paramMap,
                                const std::string &paramName);

  // ---
  // Returns a pair of fractions denoting the inclusive lower and upper
  // bounds for a destination node's receptive field in the specified
  // dimension.  This is used when calculating the splitter map (via
  // getInputForNode().  This will also be utilized when calculating the
  // getIncomingConnections() API for use by inspectors.
  // ---
  std::pair<Fraction, Fraction> getInputBoundsForNode(size_t nodeIndex,
                                                      size_t dimension) const;

  std::pair<Fraction, Fraction> getInputBoundsForNode(Coordinate nodeCoordinate,
                                                      size_t dimension) const;

  // ---
  // Calculates the entire set of bounds for a destination node's
  // receptive field, and then utilizes populateInputElements() to fill
  // in the splitter map.
  // ---
  void getInputForNode(size_t nodeIndex, std::vector<size_t> &input) const;

  void getInputForNode(Coordinate nodeCoordinate,
                       std::vector<size_t> &input) const;

  // ---
  // Recursive method which walks the entire set of bounds and populates the
  // vector "input" (the splitter map) accordingly.
  //
  // For a uniform linkage, the set of bounds defines an "orthotope" -
  // the generalization of a rectangle to n-dimensions.  That is, the
  // orthotope bounds is a collection of inclusive bounds, one for each
  // dimension, which correspond to the edges of an n-dimensional box.
  // ---
  void populateInputElements(
      std::vector<size_t> &input,
      std::vector<std::pair<Fraction, Fraction>> orthotopeBounds,
      std::vector<Fraction> &subCoordinate) const;

  // ---
  // The dimensions of the source Region, as specified by a call to
  // setSrcDimensions() or induced by a call to setDestDimensions().
  // ---
  Dimensions srcDimensions_;

  // ---
  // The dimensions of the destination Region, as specified by a call to
  // setDestDimensions() or induced by a call to setSrcDimensions().
  // ---
  Dimensions destDimensions_;

  // ---
  // The amount of elements per Node as specified by a call to
  // setNodeOutputElementCount()
  // ---
  size_t elementCount_;

  // ---
  // Parameters passed into the link policy can have varying dimensionality
  // (i.e. quantity of dimensions).  Since parameters with a dimensionality
  // of 1 can be wildcards for any number of dimensions, it is necessary to
  // calculate the true dimensionality of the parameters so as to validate
  // a requested linkage topology.  validateParameterDimensionality() checks
  // that parameter dimensionality is consistent, and sets
  // parameterDimensionality_ to the maximum dimensionality.
  // ---
  size_t parameterDimensionality_;

  // ---
  // Set after a call to initialize whereupon the working parameters are
  // valid for splitter map calculation
  // ---
  bool initialized_;

  // ---
  // A collection of parameters valid for this link policy.  Populated by
  // setValidParameters()
  // ---
  Collection<ParameterSpec> parameters_;
}; // UniformLinkPolicy

} // namespace nupic

#endif // NTA_UNIFORMLINKPOLICY_HPP
