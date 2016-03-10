# Microsoft Developer Studio Generated NMAKE File, Based on libapriconv.dsp
!IF "$(CFG)" == ""
CFG=libapriconv - Win32 Release
!MESSAGE No configuration specified. Defaulting to libapriconv - Win32 Release.
!ENDIF 

!IF "$(CFG)" != "libapriconv - Win32 Release" && "$(CFG)" != "libapriconv - Win32 Debug" && "$(CFG)" != "libapriconv - x64 Release" && "$(CFG)" != "libapriconv - x64 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "libapriconv.mak" CFG="libapriconv - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "libapriconv - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libapriconv - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libapriconv - x64 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libapriconv - x64 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "libapriconv - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ELSE 

ALL : "libapr - Win32 Release" "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"libapr - Win32 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(INTDIR)\libapriconv.pch"
	-@erase "$(INTDIR)\libapriconv.res"
	-@erase "$(INTDIR)\libapriconv_src.idb"
	-@erase "$(INTDIR)\libapriconv_src.pdb"
	-@erase "$(OUTDIR)\libapriconv-1.dll"
	-@erase "$(OUTDIR)\libapriconv-1.exp"
	-@erase "$(OUTDIR)\libapriconv-1.lib"
	-@erase "$(OUTDIR)\libapriconv-1.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /c 

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

MTL=midl.exe
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /o /win32 "NUL" 
RSC=rc.exe
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\libapriconv.res" /i "./include" /i "../apr/include" /d "NDEBUG" /d "API_VERSION_ONLY" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libapriconv.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib advapi32.lib /nologo /base:"0x6EE50000" /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)\libapriconv-1.pdb" /debug /out:"$(OUTDIR)\libapriconv-1.dll" /implib:"$(OUTDIR)\libapriconv-1.lib" /MACHINE:X86 /opt:ref 
LINK32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"$(INTDIR)\libapriconv.res" \
	"..\apr\Release\libapr-1.lib"

"$(OUTDIR)\libapriconv-1.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

TargetPath=.\Release\libapriconv-1.dll
SOURCE="$(InputPath)"
PostBuild_Desc=Embed .manifest
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

"$(DS_POSTBUILD_DEP)" : "$(OUTDIR)\libapriconv-1.dll"
   if exist .\Release\libapriconv-1.dll.manifest mt.exe -manifest .\Release\libapriconv-1.dll.manifest -outputresource:.\Release\libapriconv-1.dll;2
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "libapriconv - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ELSE 

ALL : "libapr - Win32 Debug" "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"libapr - Win32 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(INTDIR)\libapriconv.pch"
	-@erase "$(INTDIR)\libapriconv.res"
	-@erase "$(INTDIR)\libapriconv_src.idb"
	-@erase "$(INTDIR)\libapriconv_src.pdb"
	-@erase "$(OUTDIR)\libapriconv-1.dll"
	-@erase "$(OUTDIR)\libapriconv-1.exp"
	-@erase "$(OUTDIR)\libapriconv-1.lib"
	-@erase "$(OUTDIR)\libapriconv-1.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /EHsc /c 

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

MTL=midl.exe
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /o /win32 "NUL" 
RSC=rc.exe
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\libapriconv.res" /i "./include" /i "../apr/include" /d "_DEBUG" /d "API_VERSION_ONLY" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libapriconv.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib advapi32.lib /nologo /base:"0x6EE50000" /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)\libapriconv-1.pdb" /debug /out:"$(OUTDIR)\libapriconv-1.dll" /implib:"$(OUTDIR)\libapriconv-1.lib" /MACHINE:X86 
LINK32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"$(INTDIR)\libapriconv.res" \
	"..\apr\Debug\libapr-1.lib"

"$(OUTDIR)\libapriconv-1.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

TargetPath=.\Debug\libapriconv-1.dll
SOURCE="$(InputPath)"
PostBuild_Desc=Embed .manifest
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

"$(DS_POSTBUILD_DEP)" : "$(OUTDIR)\libapriconv-1.dll"
   if exist .\Debug\libapriconv-1.dll.manifest mt.exe -manifest .\Debug\libapriconv-1.dll.manifest -outputresource:.\Debug\libapriconv-1.dll;2
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "libapriconv - x64 Release"

OUTDIR=.\x64\Release
INTDIR=.\x64\Release
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep
# Begin Custom Macros
OutDir=.\x64\Release
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ELSE 

ALL : "libapr - x64 Release" "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"libapr - x64 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(INTDIR)\libapriconv.pch"
	-@erase "$(INTDIR)\libapriconv.res"
	-@erase "$(INTDIR)\libapriconv_src.idb"
	-@erase "$(INTDIR)\libapriconv_src.pdb"
	-@erase "$(OUTDIR)\libapriconv-1.dll"
	-@erase "$(OUTDIR)\libapriconv-1.exp"
	-@erase "$(OUTDIR)\libapriconv-1.lib"
	-@erase "$(OUTDIR)\libapriconv-1.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /c 

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

MTL=midl.exe
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /o /win32 "NUL" 
RSC=rc.exe
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\libapriconv.res" /i "./include" /i "../apr/include" /d "NDEBUG" /d "API_VERSION_ONLY" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libapriconv.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib advapi32.lib /nologo /base:"0x6EE50000" /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)\libapriconv-1.pdb" /debug /out:"$(OUTDIR)\libapriconv-1.dll" /implib:"$(OUTDIR)\libapriconv-1.lib" /MACHINE:X64 /opt:ref 
LINK32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"$(INTDIR)\libapriconv.res" \
	"..\apr\x64\Release\libapr-1.lib"

"$(OUTDIR)\libapriconv-1.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

TargetPath=.\x64\Release\libapriconv-1.dll
SOURCE="$(InputPath)"
PostBuild_Desc=Embed .manifest
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

# Begin Custom Macros
OutDir=.\x64\Release
# End Custom Macros

"$(DS_POSTBUILD_DEP)" : "$(OUTDIR)\libapriconv-1.dll"
   if exist .\x64\Release\libapriconv-1.dll.manifest mt.exe -manifest .\x64\Release\libapriconv-1.dll.manifest -outputresource:.\x64\Release\libapriconv-1.dll;2
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "libapriconv - x64 Debug"

OUTDIR=.\x64\Debug
INTDIR=.\x64\Debug
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep
# Begin Custom Macros
OutDir=.\x64\Debug
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ELSE 

ALL : "libapr - x64 Debug" "$(OUTDIR)\libapriconv-1.dll" "$(DS_POSTBUILD_DEP)"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"libapr - x64 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\iconv.obj"
	-@erase "$(INTDIR)\iconv_ces.obj"
	-@erase "$(INTDIR)\iconv_ces_euc.obj"
	-@erase "$(INTDIR)\iconv_ces_iso2022.obj"
	-@erase "$(INTDIR)\iconv_int.obj"
	-@erase "$(INTDIR)\iconv_module.obj"
	-@erase "$(INTDIR)\iconv_uc.obj"
	-@erase "$(INTDIR)\libapriconv.pch"
	-@erase "$(INTDIR)\libapriconv.res"
	-@erase "$(INTDIR)\libapriconv_src.idb"
	-@erase "$(INTDIR)\libapriconv_src.pdb"
	-@erase "$(OUTDIR)\libapriconv-1.dll"
	-@erase "$(OUTDIR)\libapriconv-1.exp"
	-@erase "$(OUTDIR)\libapriconv-1.lib"
	-@erase "$(OUTDIR)\libapriconv-1.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yu"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /EHsc /c 

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

MTL=midl.exe
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /o /win32 "NUL" 
RSC=rc.exe
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\libapriconv.res" /i "./include" /i "../apr/include" /d "_DEBUG" /d "API_VERSION_ONLY" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libapriconv.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib advapi32.lib /nologo /base:"0x6EE50000" /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)\libapriconv-1.pdb" /debug /out:"$(OUTDIR)\libapriconv-1.dll" /implib:"$(OUTDIR)\libapriconv-1.lib" /MACHINE:X64 
LINK32_OBJS= \
	"$(INTDIR)\iconv.obj" \
	"$(INTDIR)\iconv_ces.obj" \
	"$(INTDIR)\iconv_ces_euc.obj" \
	"$(INTDIR)\iconv_ces_iso2022.obj" \
	"$(INTDIR)\iconv_int.obj" \
	"$(INTDIR)\iconv_module.obj" \
	"$(INTDIR)\iconv_uc.obj" \
	"$(INTDIR)\libapriconv.res" \
	"..\apr\x64\Debug\libapr-1.lib"

"$(OUTDIR)\libapriconv-1.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

TargetPath=.\x64\Debug\libapriconv-1.dll
SOURCE="$(InputPath)"
PostBuild_Desc=Embed .manifest
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

# Begin Custom Macros
OutDir=.\x64\Debug
# End Custom Macros

"$(DS_POSTBUILD_DEP)" : "$(OUTDIR)\libapriconv-1.dll"
   if exist .\x64\Debug\libapriconv-1.dll.manifest mt.exe -manifest .\x64\Debug\libapriconv-1.dll.manifest -outputresource:.\x64\Debug\libapriconv-1.dll;2
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("libapriconv.dep")
!INCLUDE "libapriconv.dep"
!ELSE 
!MESSAGE Warning: cannot find "libapriconv.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "libapriconv - Win32 Release" || "$(CFG)" == "libapriconv - Win32 Debug" || "$(CFG)" == "libapriconv - x64 Release" || "$(CFG)" == "libapriconv - x64 Debug"
SOURCE=.\lib\iconv.c

!IF  "$(CFG)" == "libapriconv - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\libapriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libapriconv - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /EHsc /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\libapriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libapriconv - x64 Release"

CPP_SWITCHES=/nologo /MD /W3 /Zi /O2 /Oy- /I "./include" /I "../apr/include" /D "NDEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\libapriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libapriconv - x64 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Zi /Od /I "./include" /I "../apr/include" /D "_DEBUG" /D "API_DECLARE_EXPORT" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\libapriconv.pch" /Yc"iconv.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\libapriconv_src" /FD /EHsc /c 

"$(INTDIR)\iconv.obj"	"$(INTDIR)\libapriconv.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\lib\iconv_ces.c

"$(INTDIR)\iconv_ces.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libapriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_ces_euc.c

"$(INTDIR)\iconv_ces_euc.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libapriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_ces_iso2022.c

"$(INTDIR)\iconv_ces_iso2022.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libapriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_int.c

"$(INTDIR)\iconv_int.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libapriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_module.c

"$(INTDIR)\iconv_module.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libapriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\lib\iconv_uc.c

"$(INTDIR)\iconv_uc.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libapriconv.pch"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!IF  "$(CFG)" == "libapriconv - Win32 Release"

"libapr - Win32 Release" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - Win32 Release" 
   cd "..\apr-iconv"

"libapr - Win32 ReleaseCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - Win32 Release" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ELSEIF  "$(CFG)" == "libapriconv - Win32 Debug"

"libapr - Win32 Debug" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - Win32 Debug" 
   cd "..\apr-iconv"

"libapr - Win32 DebugCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - Win32 Debug" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ELSEIF  "$(CFG)" == "libapriconv - x64 Release"

"libapr - x64 Release" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - x64 Release" 
   cd "..\apr-iconv"

"libapr - x64 ReleaseCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - x64 Release" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ELSEIF  "$(CFG)" == "libapriconv - x64 Debug"

"libapr - x64 Debug" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - x64 Debug" 
   cd "..\apr-iconv"

"libapr - x64 DebugCLEAN" : 
   cd ".\..\apr"
   $(MAKE) /$(MAKEFLAGS) /F ".\libapr.mak" CFG="libapr - x64 Debug" RECURSE=1 CLEAN 
   cd "..\apr-iconv"

!ENDIF 

SOURCE=.\libapriconv.rc

"$(INTDIR)\libapriconv.res" : $(SOURCE) "$(INTDIR)"
	$(RSC) $(RSC_PROJ) $(SOURCE)



!ENDIF 

