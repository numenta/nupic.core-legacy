# API Changelog to Python-only code (from numenta/nupic repo)

Note: that these changes only affect python-only code under `py/` path 
(= `from nupic.xxx import yyy`), and do not affect our C++ python bindings 
(= `from nupic.bindings.xxx import yyy`)

- `Serialization` not supported as canproto was removed. 

- Changed all use of nupic to htm.   This means that Python users must import from
    - htm.bindings.algorithms
    - htm.bindings.engine_internal
    - htm.bindings.math
    - htm.bindings.encoders
    - htm.bindings.sdr
  rather than
    - nupic.bindings.algorithms
    - nupic.bindings.engine_internal
    - nupic.bindings.math
    - nupic.bindings.encoders
    - nupic.bindings.sdr

