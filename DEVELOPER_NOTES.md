# Developer Notes

This file contains informataion that might be useful for the developers of this module.

# Trace Macros

This is a library wide trace facility.  This may be useful for the mantainers 
of the library, particularly the NetworkAPI framework modules.  This facility is found
in LogItem.hpp which is included in Log.hpp.  Log.hpp should be included with all 
modules because it includes the exception handling macros as well as the DEBUG macros.

Developers should add NTA_DEBUG, NTA_INFO, NTA_WARN macros within the code at
appropreate places. Nothing will be displayed from these unless the log level
is set to values indicated below. By default the LogLevel is None so nothing is output
except exceptions.

Functions:
 * LogItem::setOutputFile(filename);
		This sets the file to which the trace will be sent.
		By default it goes to std::cerr.
		
 * LogItem::setLogLevel(LogLevel level);
		This function sets the log level.  The levels are:
		enum {LogLevel_None = 0, LogLevel_Minimal, LogLevel_Normal, LogLevel_Verbose}
		At the Verbose level,all of the NTA_DEBUG macros will be active.
		By default it is set to LogLevel_None and no trace is output.
		This function can be called from any code and it affects the trace globally.
		
 * LogItem::getLogLevel( ); 
		This function returns the log level;
		
 * bool LogItem::isDebug( );
		This function returns true if log level is LogLevel_Verbose.
		
Macros:
 * NTA_DEBUG << "compute " << *out << std::endl;
		If the log level is verbose, this macro will output the stream to std::cerr 
		or the file as specified by setOutputFile(filename).
		
 * NTA_INFO << "some text " << *out << std::endl;
		If the log level is normal or higher outputs an Info message.
		
 * NTA_WARN << "some text " << *out << std::endl;
		If the log level is normal or higher outputs an WARN message.
		
 * NTA_LDEBUG(level)
		If the log level set by setLogLevel( ) is greater than or equal to level
		this macro will output the stream to std::cerr or the file as specified
		by setOutputFile(filename).