# Microsoft Developer Studio Generated NMAKE File, Based on apriconv.dsp
!IF "$(CFG)" == ""
CFG=apriconv - Win32 Release
!MESSAGE No configuration specified. Defaulting to apriconv - Win32 Release.
!ENDIF 

!IF "$(CFG)" != "apriconv - Win32 Release" && "$(CFG)" != "apriconv - Win32 Debug" && "$(CFG)" != "apriconv - x64 Release" && "$(CFG)" != "apriconv - x64 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
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
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "apriconv - Win32 Release"

OUTDIR=.\LibR
INTDIR=.\LibR
# Begin Custom Macros
OutDir=.\LibR
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\apriconv-1.lib"

!ELSE 

ALL : "apr - Win32 Release" "$(OUTDIR)\apriconv-1.lib"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"apr - Win32 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\apriconv-1.idb"
	-@erase "$(INTDIR)\apriconv-1.pdb"
	-@erase "$(INTDIR)\apriconv.pch"
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(OUTDIR)\apriconv-1.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\apriconv.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\apriconv-1.lib" 
LIB32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"..\apr\LibR\apr-1.lib"

"$(OUTDIR)\apriconv-1.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "apriconv - Win32 Debug"

OUTDIR=.\LibD
INTDIR=.\LibD
# Begin Custom Macros
OutDir=.\LibD
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\apriconv-1.lib"

!ELSE 

ALL : "apr - Win32 Debug" "$(OUTDIR)\apriconv-1.lib"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"apr - Win32 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\apriconv-1.idb"
	-@erase "$(INTDIR)\apriconv-1.pdb"
	-@erase "$(INTDIR)\apriconv.pch"
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(OUTDIR)\apriconv-1.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /EHsc /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\apriconv.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\apriconv-1.lib" 
LIB32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"..\apr\LibD\apr-1.lib"

"$(OUTDIR)\apriconv-1.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "apriconv - x64 Release"

OUTDIR=.\x64\LibR
INTDIR=.\x64\LibR
# Begin Custom Macros
OutDir=.\x64\LibR
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\apriconv-1.lib"

!ELSE 

ALL : "apr - x64 Release" "$(OUTDIR)\apriconv-1.lib"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"apr - x64 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\apriconv-1.idb"
	-@erase "$(INTDIR)\apriconv-1.pdb"
	-@erase "$(INTDIR)\apriconv.pch"
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(OUTDIR)\apriconv-1.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\apriconv.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\apriconv-1.lib" 
LIB32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"..\apr\x64\LibR\apr-1.lib"

"$(OUTDIR)\apriconv-1.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "apriconv - x64 Debug"

OUTDIR=.\x64\LibD
INTDIR=.\x64\LibD
# Begin Custom Macros
OutDir=.\x64\LibD
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\apriconv-1.lib"

!ELSE 

ALL : "apr - x64 Debug" "$(OUTDIR)\apriconv-1.lib"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"apr - x64 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\apriconv-1.idb"
	-@erase "$(INTDIR)\apriconv-1.pdb"
	-@erase "$(INTDIR)\apriconv.pch"
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(OUTDIR)\apriconv-1.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /EHsc /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\apriconv.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\apriconv-1.lib" 
LIB32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"..\apr\x64\LibD\apr-1.lib"

"$(OUTDIR)\apriconv-1.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("apriconv.dep")
!INCLUDE "apriconv.dep"
!ELSE 
!MESSAGE Warning: cannot find "apriconv.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "apriconv - Win32 Release" || "$(CFG)" == "apriconv - Win32 Debug" || "$(CFG)" == "apriconv - x64 Release" || "$(CFG)" == "apriconv - x64 Debug"
SOURCE=.\lib\iconv.c

!IF  "$(CFG)" == "apriconv - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\apriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "apriconv - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /EHsc /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\apriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "apriconv - x64 Release"

CPP_SWITCHES=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\apriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "apriconv - x64 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "APR_DECLARE_STATIC" /D "API_DECLARE_STATIC" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\apriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(OUTDIR)\apriconv-1" /FD /EHsc /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\apriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\lib\iconv_ces.c

"$(INTDIR)\iconv_ces.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\apriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_ces_euc.c

"$(INTDIR)\iconv_ces_euc.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\apriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_ces_iso2022.c

"$(INTDIR)\iconv_ces_iso2022.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\apriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_int.c

"$(INTDIR)\iconv_int.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\apriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_module.c

"$(INTDIR)\iconv_module.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\apriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_uc.c

"$(INTDIR)\iconv_uc.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\apriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!IF  "$(CFG)" == "apriconv - Win32 Release"

"apr - Win32 Release" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - Win32 Release" 
   cd "..\apr-iconv"

"apr - Win32 ReleaseCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - Win32 Release" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ELSEIF  "$(CFG)" == "apriconv - Win32 Debug"

"apr - Win32 Debug" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - Win32 Debug" 
   cd "..\apr-iconv"

"apr - Win32 DebugCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - Win32 Debug" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ELSEIF  "$(CFG)" == "apriconv - x64 Release"

"apr - x64 Release" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - x64 Release" 
   cd "..\apr-iconv"

"apr - x64 ReleaseCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - x64 Release" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ELSEIF  "$(CFG)" == "apriconv - x64 Debug"

"apr - x64 Debug" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - x64 Debug" 
   cd "..\apr-iconv"

"apr - x64 DebugCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\apr.mak" CFG="apr - x64 Debug" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ENDIF 


!ENDIF 

