htm_advanced
==============

This package contains a port to Python3 and htm.core of the location framework code from the [htmresearch repository](https://github.com/numenta/htmresearch) of Numenta. 
htmresearch contains experimental algorithm work done internally at Numenta.

The htmresearch code was written in Python2 which is reaching end of life at the end of 2019. Numenta has stated that it has no plans to port htmresearch to Python3 and is no longer going continue development in htmresearch.

Since Numenta is no longer developing htmresearch, the API in htm advanced can me considered stable. The only caveat is that the modules directly use the htm Connections class so any changes to its API may effect htm advanced.

Location Framework
==============
The location framework is the code that underlies Jeff Hawkins Thousand Brains model of the Neocortex and hence is probably the most important framework in htmresearch.  
There are some example applications showing how to use the location framework.

Additional goodies are the RawSensor and RawValues regions, the GridCellLocationRegion and the ColumnPoolerRegion.

Future Additions
================
Other frameworks from htmresearch may be added in the future.

Namespace
=========
The location framework is located in the htm_advanced namespace. In the future it may be promoted to the htm namespace.