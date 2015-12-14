/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

#if !CAPNP_LITE

#include <nupic/py_support/PyCapnp.hpp>
#include <nupic/py_support/PyHelpers.hpp>

namespace nupic
{
  bool pyCapnpInitialized = false;

  PyObject* getPyReader(capnp::DynamicStruct::Reader reader)
  {
    if (!pyCapnpInitialized) {
      initCapnpToPycapnp();
      pyCapnpInitialized = true;
    }
    py::Ptr parent(Py_None);
    return createReader(reader, parent);
  }

  PyObject* getPyBuilder(capnp::DynamicStruct::Builder builder)
  {
    if (!pyCapnpInitialized) {
      initCapnpToPycapnp();
      pyCapnpInitialized = true;
    }
    py::Ptr parent(Py_None);
    return createBuilder(builder, parent);
  }

} // namespace nupic

#endif // !CAPNP_LITE
