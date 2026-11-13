"""End-to-end check for the locally built llama-cpp-python (HIPBLAS / gfx1201).

Run with ComfyUI's embedded python:
    J:\\ComfyUI_windows_portable_amd\\ComfyUI_windows_portable\\python_embeded\\python.exe \\
        J:\\llama_cpp_build\\test_llama_gpu.py

`import torch` must come first: torch registers the ROCm runtime directories
(amdhip64_7.dll / hipblas.dll / rocblas.dll live in the pip ROCm SDK, not in
torch\\lib), so ggml-hip.dll can resolve them.
"""

import os
import sys
import time

import torch  # noqa: F401  (registers ROCm DLL directories)

import llama_cpp
from llama_cpp.llama_chat_format import Qwen35ChatHandler

print("=" * 60)
print("llama_cpp      :", llama_cpp.__version__)
print("module         :", llama_cpp.__file__)
print("gpu offload    :", llama_cpp.llama_supports_gpu_offload())
print("max devices    :", llama_cpp.llama_max_devices())
print("torch cuda     :", torch.cuda.is_available())
print("=" * 60)

CANDIDATES = [
    # (main model, mmproj, label)
    (
        r"G:\models\LLM\Qwen-Image-2.1-PE-I2I.Q5_K_S.gguf",
        r"G:\models\LLM\Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf",
        "Qwen-Image-2.1 prompt enhancer (text + mmproj)",
    ),
    (
        r"J:\models\Qwen3.5-4B-Heretic-MTP\Qwen3.5-4B-EmperoAI-Qwen3.8-Distill-Heretic-Abliterated-MTP-GGUF-Q4_K_M.gguf",
        None,
        "Qwen3.5-4B (text only; Qwen35ChatHandler needs an mmproj, so this may fail)",
    ),
]


def probe(model_path, mmproj_path, label):
    if not os.path.exists(model_path):
        print(f"[skip] {label}: model not found -> {model_path}")
        return False
    print(f"\n--- {label} ---")
    print("model:", model_path)
    handler_kwargs = {"enable_thinking": False, "verbose": True}
    if mmproj_path:
        handler_kwargs["mmproj_path"] = mmproj_path
        handler_kwargs["use_gpu"] = True
    handler = Qwen35ChatHandler(**handler_kwargs)
    t0 = time.time()
    try:
        llm = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=handler,
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=2048,
            n_ubatch=512,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] load failed: {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] loaded in {time.time() - t0:.1f}s")
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=32,
        )
        text = out["choices"][0]["message"]["content"]
        print("[OK] reply:", repr(text[:200]))
        print("usage:", out.get("usage"))
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] inference failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        llm.close()
    return True


if __name__ == "__main__":
    ok = False
    for model_path, mmproj_path, label in CANDIDATES:
        if probe(model_path, mmproj_path, label):
            ok = True
            break
    sys.exit(0 if ok else 1)
