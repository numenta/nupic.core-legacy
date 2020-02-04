/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2017, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Interface for the internal Input class.
 *
 * @note This class is internal, and is not wrapped.
 */

#ifndef NTA_INPUT_HPP
#define NTA_INPUT_HPP

#include <htm/ntypes/Array.hpp>
#include <htm/types/Types.hpp>
#include <vector>

namespace htm {
class Link;
class Region;
class Output;

/**
 * Represents a named input to a Region. (e.g. bottomUpIn)
 *
 * @note Input is not available in the public API, but is visible by
 * the RegionImpl.
 *
 * @todo identify methods that may be called by RegionImpl -- this
 * is the internal "public interface"
 */
class Input {
public:
  /**
   * Constructor.
   *
   * @param region
   *        The region that the input belongs to.
   * @param type
   *        The type of the input, i.e. TODO
   * @param isRegionLevel
   *        Whether the input is region level, i.e. TODO
   */
  Input(Region* region, const std::string& inputName, 
        NTA_BasicType type);

  /**
   *
   * Destructor.
   *
   */
  ~Input();

  /**
   *
   * Set the name for the input.
   *
   * Inputs need to know their own name for error messages.
   *
   * @param name
   *        The name of the input
   *
   */
  void setName(const std::string &name);

  /**
   * Get the name of the input.
   *
   * @return
   *        The name of the input
   */
  const std::string &getName() const;

  /**
   * Add the given link to this input and to the list of links on the output
   *
   * @param link
   *        The link to add.
   * @param srcOutput
   *        The output of previous Region, which is also the source of the input
   */
  void addLink(const std::shared_ptr<Link> link, std::shared_ptr<Output> srcOutput);

  /**
   * Locate an existing Link to the input.
   *
   * It's called by Network.removeLink() and internally when adding a link
   *
   * @param srcRegionName
   *            The name of the source Region
   * @param srcOutputName
   *            The name of the source Output
   *
   * @returns
   *     The link if found or @c NULL if no such link exists
   */
  std::shared_ptr<Link> findLink(const std::string &srcRegionName,
                 const std::string &srcOutputName);

  /**
   * Removing an existing link from the input.
   *
   * It's called in four cases:
   *
   * 1. Network.removeLink()
   * 2. Network.removeRegion() when given @a srcRegion
   * 3. Network.removeRegion() when given @a destRegion
   * 4. Network.~Network()
   *
   * It is an error to call this if our containing region
   * is uninitialized.
   *
   * @note This method will set the Link pointer to NULL on return (to avoid
   * a dangling reference)
   *
   * @param link
   *        The Link to remove, possibly retrieved by findLink(), note that
   *        it is a reference to the pointer, not the pointer itself.
   */
  void removeLink(const std::shared_ptr<Link>& link);

  /**
   * Make input data available.
   *
   * Called by Region.prepareInputs()
   */
  void prepare();

  /**
   *
   * Get the data of the input.
   *
   * @returns
   *         A mutable reference to the data of the input as an @c Array
   */
  Array &getData() { NTA_CHECK(initialized_); return data_; }
  const Array &getData() const { NTA_CHECK(initialized_); return data_; }
  /**
   *  Get the data type of the output
   */
  NTA_BasicType getDataType() const { return data_.getType(); }

  /**
   *
   * Get the Region that the input belongs to.
   *
   * @returns
   *         The mutable reference to the Region that the input belongs to
   */
  const Region *getRegion() const { return region_; }

  /**
   *
   * Get all the Link objects added to the input.
   *
   * @returns
   *         All the Link objects added to the input
   */
  std::vector<std::shared_ptr<Link>> &getLinks() { return links_; }

  /**
   * Called by Region.evaluateLinks() as part
   * of network initialization.
   *
   * 1. Tries to make sure that dimensions at both ends
   *    of a link are specified.
   * 2. Ensures that region input dimensions are consistent
   *    either by setting source and destination dimensions 
   *    or by raising an exception if they are inconsistent.
   *
   */
  void initialize();

  /**
   * Tells whether the Input is initialized.
   *
   * @returns
   *         Whether the Input is initialized
   */
  bool isInitialized();

    /**
   * Get dimensions for this input
   */
  Dimensions &getDimensions() { return dim_; }

  /**
   * Set dimensions for this input
   */
  void setDimensions(Dimensions dim) { dim_ = std::move(dim); }

  /**
   * true if we have links connected to this input.
   */
  bool hasIncomingLinks() { return !links_.empty(); }

  /**
   * Resize the input buffer.  This is called if a connected output is resized.
   */
  void resize();

  /**
   *  Print raw data...for debugging
   */
  friend std::ostream &operator<<(std::ostream &f, const Input &d);

private:
  // Cannot use the shared_ptr here.
  Region* region_;


  // Use a vector of links because order is important.
  std::vector<std::shared_ptr<Link>> links_;

  bool initialized_;
  Dimensions dim_;
  Array data_;

  // Useful for us to know our own name
  std::string name_;

  // Internal methods

  /*
   * uninitialize is called by removeLink
   * and in our destructor. It is an error
   * to call it if our region is initialized.
   * It frees the input buffer and the splitter map
   * but does not affect the links.
   */
  void uninitialize();
};

} // namespace htm

#endif // NTA_INPUT_HPP
