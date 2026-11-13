# llama-cpp-python — AMD ROCm (HIPBLAS) Windows wheels

Community-built wheels of the **JamePeng fork of
[llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)** compiled with
the **AMD ROCm HIP backend** for **Windows**, targeting **gfx1201** (AMD RDNA4 —
Radeon RX 9070 XT / 9070 GRE / R9700).

> **This is not an official release.** Not published by, endorsed by, or an
> official release of Andrei Betlen (abetlen) or JamePeng. It is a community
> build produced because no Windows/AMD wheel of this fork was published.

## Why this exists

As of this build there is **no downloadable `llama-cpp-python` wheel for AMD
GPUs on Windows**:

| Source | Windows AMD wheels |
|---|---|
| PyPI (`abetlen/llama-cpp-python`) | CPU only |
| `JamePeng/llama-cpp-python` releases | CUDA (cu124–cu132) + Metal — no AMD |
| `jllllll/llama-cpp-python-cuBLAS-wheels` | ROCm tag exists, but Windows ROCm stopped at 0.2.21 |
| `abetlen` `build-wheels-rocm.yaml` | a `workflow_dispatch`-only job that can build them, but nothing is published |

This matters for ComfyUI, because several node packs call `llama_cpp`
**in-process** and need the fork's newer chat handlers:

* `Goohaitools-comfyui` → `from llama_cpp.llama_chat_format import Qwen35ChatHandler`
* `ComfyUI-llama_Reversal` → `from llama_cpp import Llama, llama_split_mode`

`Qwen35ChatHandler` only exists in the JamePeng fork — upstream 0.3.36 has
`Qwen25VLChatHandler` / `Gemma4ChatHandler` but no `Qwen35`. So a stock HIP
wheel of upstream would not satisfy those nodes even if one were published.

## Provenance and attribution

| Component | Version | License |
|---|---|---|
| `llama-cpp-python` | [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @ `c7c7055` (v0.4.7) | MIT — [`LICENSE`](LICENSE) |
| `llama.cpp` (vendored) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) @ `2a4daf0e` | MIT — [`LICENSE.llama.cpp`](LICENSE.llama.cpp) |
| `libomp140.x86_64.dll` | LLVM OpenMP runtime (shipped with the wheel) | Apache-2.0 WITH LLVM-exception |

The fork's `Qwen35ChatHandler`, `GLM46VChatHandler`, `Gemma4ChatHandler` and
`Qwen3VLChatHandler` are JamePeng's additions and are **the reason this wheel
exists**. Please credit them.

The three patches in [`patches/`](patches/) fix bugs in **upstream ggml-org/llama.cpp**
and are written to be upstreamable:

1. `0001` — `GGML_CUDA_FA_QUANTS` loses its quotes on Windows + Ninja + clang
   (`use of undeclared identifier 'q4_0'`)
2. `0002` — allow the modern CMake HIP language on Windows
3. `0003` — jinja: use the non-const `std::function::target()` overload
   (MSVC's STL casts away qualifiers, which clang rejects)

## Install

```powershell
# 1) ROCm runtime (AMD's DLLs are NOT redistributable, so they are not in the wheel)
python -m pip install rocm rocm-sdk-core rocm-sdk-libraries rocm-sdk-device-gfx1201

# 2) the wheel
python -m pip install --no-deps llama_cpp_python-0.4.7-cp312-cp312-win_amd64.whl
```

`--no-deps` matters: the fork pins `numpy<=2.3.2`, while current PyTorch/ComfyUI
stacks run numpy 2.5.x. Let pip resolve the dependencies and it will try to
downgrade numpy.

The ROCm DLL directories have to be on the process' DLL search path. Importing
the ROCm build of PyTorch does that automatically (`os.add_dll_directory`), so
in ComfyUI — which always imports torch first — nothing else is needed. For
standalone scripts, import torch before `llama_cpp`, or add the ROCm `bin`
directory yourself.

> Do **not** put an older ROCm install's `bin` on `PATH` next to a ROCm
> PyTorch build. Windows resolves DLLs by base name, so the first
> `amdhip64_7.dll` loaded wins and the two ROCm runtimes will fight.

## Verified

| Check | Result |
|---|---|
| ROCm device discovery | `found 1 ROCm devices` → `AMD Radeon RX 9070 XT, gfx1201, VRAM 16304 MiB` |
| `llama_cpp.__version__` | `0.4.7` |
| `llama_supports_gpu_offload()` | `True` |
| `Qwen35ChatHandler` / `Gemma4ChatHandler` / `Qwen3VLChatHandler` / `GLM46VChatHandler` | all importable |
| Real inference | Qwen-Image-2.1 PE I2I (9B Q5_K, 6.0 GB) + mmproj (879 MB): loaded in **4.3 s**, prompt eval **622.9 tok/s** |
| End-to-end in ComfyUI | the "Qwen image 2.1 提示词优化" node produced an optimized prompt in **9.91 s** |

Reproduce with `build/test_llama_gpu.py`.

## Build from source

Requirements: Windows x64, Python 3.12, Visual Studio 2019 (MSVC v142) or newer,
the ROCm HIP SDK (7.2 was used here), CMake ≥ 3.21, Ninja.

```powershell
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
git submodule update --init --recursive
cd vendor/llama.cpp
git apply ../../patches/0001-*.patch ../../patches/0002-*.patch ../../patches/0003-*.patch
cd ../..
build\cmake_configure.cmd
build\build_hip.cmd
python build\pack_wheel.py <src> <platlib> <out>
```

The scripts in `build/` are the exact ones used. Key CMake arguments:

```
-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
-DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
CC=<rocm>/bin/clang.exe   CXX=<rocm>/bin/hipcc.exe   ROCM_PATH=<rocm>
```

### Things that silently do the wrong thing

These are the seven traps this build walked into. Each one either fails with a
misleading message or, worse, succeeds while building the wrong thing.

1. **The option is `GGML_HIP`, not `GGML_HIPBLAS`.** Passing the old name does
   not error — the HIP backend is simply left out of the build graph
   (`build.ninja` contains zero `ggml-hip` references). The tell is the
   configure line `HIP and hipBLAS found`.
2. **Do not put compiler paths in `CMAKE_ARGS`.** `scikit-build-core` mangles
   `J:/rocm72/bin/hipcc.exe` into `J:rocm72/bin/hipcc.exe`. Use the `CC`/`CXX`
   environment variables instead.
3. **`hipcc.bat` cannot be the C++ compiler.** ROCm's `hip-config-amd.cmake`
   runs `${CMAKE_CXX_COMPILER} -print-libgcc-file-name`, and the `.bat` echoes
   the whole command line (including the `>` prompt) into stdout, which CMake
   then rejects as trailing whitespace. Use `hipcc.exe`.
4. **Quotes in `-D` survive three parsers.** Ninja → cmd → `hipcc.exe` → clang
   can each strip them, so `add_compile_definitions(GGML_CUDA_FA_QUANTS="...")`
   arrives at clang as a bare token list. `cl.exe` keeps the quotes; clang does
   not. See patch `0001`.
5. **A failed `git submodule update` can wipe the worktree.** Once git enters
   its "Retry scheduled" state, the next git operation deletes and re-clones
   `vendor/llama.cpp` — and the re-clone fails, leaving an empty directory.
   Build from a copy without `.git`.
6. **CMake refuses to mix Clang and MSVC on Windows.** `Windows-Clang.cmake`
   errors out with *"mixes Clang and MSVC ... not supported"*, even though
   CMake ≥ 3.21 ships `Platform/Windows-Clang-HIP.cmake`. Everything therefore
   has to be compiled by ROCm's clang, not just the device code.
7. **MSVC 2019's STL and clang do not get along.** `std::function::target()`'s
   const overload is implemented as a `reinterpret_cast` that casts away
   qualifiers, which MSVC's compiler accepts and clang rejects. See patch
   `0003`.

## Known limitations

* Only **gfx1201** is compiled in. Other AMD architectures need a rebuild with
  a different `GPU_TARGETS` (e.g. `gfx1100;gfx1200;gfx1201` for a fat binary).
* AMD's ROCm runtime is **not** bundled — install it separately.
* `ggml-hip.dll` is ~86 MB for a single architecture.
* The wheel is CPython 3.12 / Windows x64 only.

## Credits

* [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT
* [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — MIT
* [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) — MIT
  (the `Qwen35` / `GLM46V` / `Gemma4` / `Qwen3VL` chat handlers)
* [AMD ROCm](https://rocm.docs.amd.com/) — the HIP SDK and runtime

## License

MIT. See [`LICENSE`](LICENSE) (llama-cpp-python) and
[`LICENSE.llama.cpp`](LICENSE.llama.cpp) (llama.cpp).
