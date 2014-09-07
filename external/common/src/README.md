Sources of External libraries 

========================================

 

Currently, the policy of using external libraries in `nupic.core` is to include their headers and built binaries for supported platforms, so no source is included.                                                        

 

But for some libraries, it may consists only one source file and the policy above become cumbersome.



The directory is a possibly temporary exception to the policy, for the one-source-file external libraries: 



* [Google Test](https://code.google.com/p/googletest/): see [here](https://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide#Fusing_Google_Test_Source_Files) for how the one file source is generated.
