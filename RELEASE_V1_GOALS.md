

We, the active developers of the community fork, have agreed on these tentative goals for the next release of this project:

*    An HTM which is accessible in C++ and Python
       - Will be implemented in C++, with bindings to python language.
       - Loose API compatibility with previous versions of Nupic. We will change the API to improve the implementation and the usability of the algorithms, while trying not to diverge too much from the original API.

*    The NetworkAPI
       - Strong API compatibility with previous versions of Nupic
       - To support: Python, stand-alone C++, and C# interfaces.
       - Access to more built-in algorithms and encoders (C++ and do not require Python)
       - Multi-threading; parallel execution of regions when possible.

*    Ongoing Research:
       - HTM theory is still being researched, and as new models are made and tested, we can polish them and bring them into this repository.

### Code Organization

* We will endeavour to deliver a single code repository which contains the necessary tools in both C++ & Python.

   - However we won't copy the entire python repository into the C++ repository. Instead we will copy files when they're needed and ready (meaning they pass a code review).
  -  Not on this agenda: supporting duplicate algorithms in both C++ & Python.
   - Merging the unit-tests is our priority here, since they can be used to verify API compatibility
