# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


# The build prepends this module verbatim to each nupic.bindings python
# extension proxy module to load pycapnp's extension shared library in global
# scope before loading our own extension DLL (that doesn't contain capnproto
# code) so that our capnproto references will resolve against capnproto included
# in pycapnp. This ensures that the methods of the same capnproto build that
# creates the capnproto objects in nupic will be used on those objects from both
# nupic and nupic.bindings shared objects.


def _nupic_bindings_load_capnp_shared_object():
  import platform
  # Windows nupic.bindings extensions include CAPNP_LITE capnproto subset and
  # must not depend on pycapnp
  if platform.system() != "Windows":
    import ctypes, imp, os
    capnpPackageDir = imp.find_module('capnp')[1]
    capnpDLLPath=os.path.join(capnpPackageDir, 'lib', 'capnp.so')
    ctypes.CDLL(capnpDLLPath, ctypes.RTLD_GLOBAL)

_nupic_bindings_load_capnp_shared_object()

del _nupic_bindings_load_capnp_shared_object
