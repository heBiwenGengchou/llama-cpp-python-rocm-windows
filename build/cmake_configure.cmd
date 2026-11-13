@echo off
REM Direct CMake configure for the fork source tree (bypasses pip / scikit-build-core)
REM
REM   Windows has no "MSVC host + HIP device" mode: CMake explicitly refuses to
REM   mix Clang and MSVC (Platform/Windows-Clang.cmake), so everything goes through
REM   ROCm's clang / hipcc.  The MSVC-STL incompatibilities that clang trips over
REM   are patched in the vendored llama.cpp (see git branch
REM   patch/windows-ninja-define-quoting).
setlocal

set "SRC=J:\llama_cpp_build\src"
set "VENV=J:\llama_cpp_build\buildenv"
set "ROCM=J:\rocm72"
set "BUILD=J:\llama_cpp_build\build_hip"

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 exit /b 1

set "PATH=%VENV%\Scripts;%ROCM%\bin;%PATH%"
set "ROCM_PATH=%ROCM%"
set "CC=%ROCM%\bin\clang.exe"
set "CXX=%ROCM%\bin\hipcc.exe"

echo === configure ===
cmake -S "%SRC%" -B "%BUILD%" -G Ninja ^
  -DGGML_HIP=ON ^
  -DGGML_HIP_NO_VVM=ON ^
  -DGPU_TARGETS=gfx1201 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_CURL=OFF ^
  -DGGML_OPENMP=OFF ^
  -DCMAKE_CXX_FLAGS=--offload-arch=gfx1201 ^
  -DPython_EXECUTABLE=%VENV%\Scripts\python.exe ^
  -DSKBUILD_PLATLIB_DIR=J:/llama_cpp_build/platlib
echo [cmake] exit=%ERRORLEVEL%
exit /b %ERRORLEVEL%
