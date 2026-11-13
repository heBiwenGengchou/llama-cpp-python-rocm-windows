@echo off
REM Compile llama.cpp + ggml-hip + mtmd for gfx1201 (AMD RX 9070 XT)
setlocal

set "SRC=J:\llama_cpp_build\src"
set "VENV=J:\llama_cpp_build\buildenv"
set "ROCM=J:\rocm72"
set "BUILD=J:\llama_cpp_build\build_hip"
set "LOG=J:\llama_cpp_build\hip_build.log"

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 exit /b 1

set "PATH=%VENV%\Scripts;%ROCM%\bin;%PATH%"
set "ROCM_PATH=%ROCM%"
set "AMDGPU_TARGETS=gfx1201"
set "CC=%ROCM%\bin\clang.exe"
set "CXX=%ROCM%\bin\hipcc.exe"

echo === compile start %date% %time% ===
cmake --build "%BUILD%" --config Release --parallel 12 > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [compile] exit=%RC%  %date% %time%

if not "%RC%"=="0" (
  echo ---- last 40 lines ----
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 40"
) else (
  echo ---- built DLLs ----
  dir /b "%BUILD%\bin\*.dll" 2>nul
  dir /b "%BUILD%\bin\*.dll" 2>nul
)
exit /b %RC%
