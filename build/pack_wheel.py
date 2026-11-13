"""Package the locally built llama.cpp shared libraries into a llama-cpp-python wheel.

Reproduces the wheel published in dist/.  It is a straight file copy of the
upstream Python package plus the DLLs that `cmake --install` produced, with a
hand-written dist-info (the upstream scikit-build-core backend hangs on this
machine, and the wheel metadata is simple enough to write by hand).

Usage:
    python pack_wheel.py <src-dir> <platlib-dir> <out-dir>

    <src-dir>      the llama-cpp-python checkout (contains llama_cpp/*.py)
    <platlib-dir>  where `cmake --install` put the DLLs (…/llama_cpp/lib)
    <out-dir>      where the .whl is written
"""

import shutil
import sys
from pathlib import Path

VERSION = "0.4.7"
DIST_INFO = f"llama_cpp_python-{VERSION}.dist-info"

# AMD's ROCm runtime DLLs are not redistributable and are not part of the
# wheel.  They are resolved at runtime through the ROCm SDK that PyTorch's
# ROCm build already registers via os.add_dll_directory().
ROC_DLLS = ("amdhip64_7.dll", "hipblas.dll", "rocblas.dll")

METADATA = f"""Metadata-Version: 2.1
Name: llama-cpp-python
Version: {VERSION}
Summary: Efficient Python bindings for the llama.cpp library (community ROCm/HIP build for AMD Windows)
Author: Andrei Betlen
Author-email: andrei.betlen@gmail.com
Maintainer: JamePeng
Maintainer-email: jame_peng@sina.com
License: MIT
Project-URL: Homepage, https://github.com/JamePeng/llama-cpp-python
Project-URL: Source, https://github.com/JamePeng/llama-cpp-python/tree/c7c7055661f87c21602d81b386747c2321f6cca5
Requires-Python: >=3.9
Requires-Dist: typing-extensions>=4.8.0
Requires-Dist: numpy>=1.21.6
Requires-Dist: diskcache>=5.6.2
Requires-Dist: jinja2>=3.1.0
Requires-Dist: Pillow>=9.5.0
Description-Content-Type: text/markdown

COMMUNITY BUILD - NOT AN OFFICIAL RELEASE.

This wheel is a locally built AMD ROCm (HIPBLAS) variant of the JamePeng fork of
llama-cpp-python.  It is NOT published by, endorsed by, or an official release of
Andrei Betlen or JamePeng.

Provenance
----------
  llama-cpp-python : JamePeng/llama-cpp-python @ c7c7055661f87c21602d81b386747c2321f6cca5 (v0.4.7), MIT
                      (contains JamePeng's Qwen35ChatHandler / GLM46VChatHandler /
                       Gemma4ChatHandler / Qwen3VLChatHandler additions)
  llama.cpp        : ggml-org/llama.cpp @ 2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c, MIT
  Local patches   : ggml-org/llama.cpp fixes, upstream-worthy:
                      - quote GGML_CUDA_FA_QUANTS so Windows+Ninja+clang keeps the quotes
                      - jinja: use the non-const std::function::target() overload
                        (MSVC's STL casts away qualifiers, which clang rejects)
                      - allow the modern CMake HIP language on Windows
  Toolchain       : ROCm 7.2 (clang 21, HIP SDK) + MSVC v142 (VS2019) + cmake 4.4.4 + ninja 1.13
  Target          : gfx1201 (AMD RDNA4, e.g. Radeon RX 9070 XT / 9070 GRE)

Runtime requirements
--------------------
  * The AMD ROCm runtime DLLs (amdhip64_7.dll / hipblas.dll / rocblas.dll ...) are
    NOT bundled: they are not redistributable.  Install the ROCm Python SDK
    (pip install rocm rocm-sdk-core rocm-sdk-libraries) or the official ROCm
    Windows installer, and make sure the DLL directories are on the process'
    DLL search path (importing torch with the ROCm build does this automatically).
  * Only the ggml-hip backend is compiled for gfx1201; other AMD architectures
    need a rebuild with a different GPU_TARGETS value.
"""

WHEEL = """Wheel-Version: 1.0
Generator: pack_wheel.py (community ROCm build)
Root-Is-Purelib: false
Tag: cp312-cp312-win_amd64
"""


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src, platlib, out = (Path(p) for p in sys.argv[1:])

    root = out / "wheelroot"
    if root.exists():
        shutil.rmtree(root)
    pkg = root / "llama_cpp"
    (pkg / "lib").mkdir(parents=True)
    (root / DIST_INFO).mkdir(parents=True)

    for py in sorted((src / "llama_cpp").glob("*.py")):
        shutil.copy2(py, pkg / py.name)
    typed = src / "llama_cpp" / "py.typed"
    if typed.exists():
        shutil.copy2(typed, pkg / "py.typed")

    for dll in sorted((platlib / "llama_cpp" / "lib").glob("*.dll")):
        if dll.name.lower() in ROC_DLLS:
            print("skip (not redistributable):", dll.name)
            continue
        shutil.copy2(dll, pkg / "lib" / dll.name)

    (root / DIST_INFO / "METADATA").write_text(METADATA, encoding="utf-8")
    (root / DIST_INFO / "WHEEL").write_text(WHEEL, encoding="utf-8")
    (root / DIST_INFO / "top_level.txt").write_text("llama_cpp\n", encoding="utf-8")

    # MIT requires the copyright notice and the permission text to travel with
    # the software, so both licences go into the wheel's dist-info.
    for src_lic, dst_name in (
        (src / "LICENSE.md", "LICENSE"),
        (src / "vendor" / "llama.cpp" / "LICENSE", "LICENSE.llama.cpp"),
    ):
        if src_lic.exists():
            shutil.copy2(src_lic, root / DIST_INFO / dst_name)
            print("license:", dst_name, "<-", src_lic)
        else:
            print("license not found, skipped:", src_lic)

    out.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "wheel", "pack", str(root), "-d", str(out)],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
