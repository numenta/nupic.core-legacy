Nupic.Base is a very slim cpp library containing mostly the htm algorithms. It's cpp only with very few dependencies:

* boost filesytem

No python is needed!


Changes to Numenta's nupic.core

* removed bindings/
* removed external/common/
* removed external/windows32-gcc
* removed external/windows64
* removed external/windows64-gcc

* revamped cmake script

* all dependencies need to be supplied, see cmake script