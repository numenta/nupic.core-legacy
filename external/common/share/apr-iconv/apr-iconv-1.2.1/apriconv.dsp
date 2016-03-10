# Microsoft Developer Studio Project File - Name="apriconv" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=apriconv - Win32 Release
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "apriconv.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "apriconv.mak" CFG="apriconv - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "apriconv - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "apriconv - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "apriconv - x64 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "apriconv - x64 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "apriconv - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "LibR"
# PROP BASE Intermediate_Dir "LibR"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "LibR"
# PROP Intermediate_Dir "LibR"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MD /W3 /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /FD /c
# ADD CPP /nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Yu"iconv.h" /Fo"$(INTDIR)\" /Fd"$(OUTDIR)\apriconv-1" /FD /c
# ADD BASE RSC /l 0x409
# ADD RSC /l 0x409
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"LibR\apriconv-1.lib"

!ELSEIF  "$(CFG)" == "apriconv - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "LibD"
# PROP BASE Intermediate_Dir "LibD"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "LibD"
# PROP Intermediate_Dir "LibD"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MDd /W3 /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /FD /EHsc /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Yu"iconv.h" /Fo"$(INTDIR)\" /Fd"$(OUTDIR)\apriconv-1" /FD /EHsc /c
# ADD BASE RSC /l 0x409
# ADD RSC /l 0x409
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"LibD\apriconv-1.lib"

!ELSEIF  "$(CFG)" == "apriconv - x64 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "x64\LibR"
# PROP BASE Intermediate_Dir "x64\LibR"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "x64\LibR"
# PROP Intermediate_Dir "x64\LibR"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MD /W3 /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /FD /c
# ADD CPP /nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Yu"iconv.h" /Fo"$(INTDIR)\" /Fd"$(OUTDIR)\apriconv-1" /FD /c
# ADD BASE RSC /l 0x409
# ADD RSC /l 0x409
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"x64\LibR\apriconv-1.lib"

!ELSEIF  "$(CFG)" == "apriconv - x64 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "x64\LibD"
# PROP BASE Intermediate_Dir "x64\LibD"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "x64\LibD"
# PROP Intermediate_Dir "x64\LibD"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MDd /W3 /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /FD /EHsc /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Yu"iconv.h" /Fo"$(INTDIR)\" /Fd"$(OUTDIR)\apriconv-1" /FD /EHsc /c
# ADD BASE RSC /l 0x409
# ADD RSC /l 0x409
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"x64\LibD\apriconv-1.lib"

!ENDIF 

# Begin Target

# Name "apriconv - Win32 Release"
# Name "apriconv - Win32 Debug"
# Name "apriconv - x64 Release"
# Name "apriconv - x64 Debug"
# Begin Group "Source Files"

# PROP Default_Filter ".c"
# Begin Group "lib"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\lib\iconv.c
# ADD CPP /Yc"iconv.h"
# End Source File
# Begin Source File

SOURCE=.\lib\iconv_ces.c
# End Source File
# Begin Source File

SOURCE=.\lib\iconv_ces_euc.c
# End Source File
# Begin Source File

SOURCE=.\lib\iconv_ces_iso2022.c
# End Source File
# Begin Source File

SOURCE=.\lib\iconv_int.c
# End Source File
# Begin Source File

SOURCE=.\lib\iconv_module.c
# End Source File
# Begin Source File

SOURCE=.\lib\iconv_uc.c
# End Source File
# End Group
# End Group
# Begin Group "Header Files"

# PROP Default_Filter ".h"
# Begin Source File

SOURCE=.\include\api_version.h
# End Source File
# Begin Source File

SOURCE=.\include\apr_iconv.h
# End Source File
# Begin Source File

SOURCE=.\lib\charset_alias.h
# End Source File
# Begin Source File

SOURCE=.\lib\iconv.h
# End Source File
# End Group
# End Target
# End Project
