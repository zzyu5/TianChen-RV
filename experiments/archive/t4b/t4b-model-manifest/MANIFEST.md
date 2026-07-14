# T4b model manifest — DeepSeek-R1-Distill-Llama-8B-Q4_K_M (ssh rvv board)

Operational index only. The model body is **board-only, NOT in git** — this MANIFEST
is the sole tracked artifact. M3-prerequisite (real Q4_K gguf for full-construct e2e
integration); does not block M1. Same model as the conduction ledger (对账口径一致).

## Model
- Name: `DeepSeek R1 Distill Llama 8B` (`general.name`)
- File: `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`
- Quantization: **Q4_K_M** (`general.file_type = 15` = `MOSTLY_Q4_K_M`)
- Architecture: `llama`

## Board location (rvv)
- Path: `/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`
- Board: `ssh rvv` (openEuler / VLEN128 / 64c; `/home` had 739G free before, ~4.6G used)

## Integrity
- Size: `4920736608` bytes (4.58 GiB) — matches source `x-linked-size` / `content-length` exactly
- sha256: `87bcba20b4846d8dadf753d3ff48f9285d131fc95e3e0e7e934d4f20bc896f5d`

## Provenance / source
- Canonical HF repo: `bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF`, file
  `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` (public, no token).
- **Downloaded via mirror** `https://hf-mirror.com/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`
  — direct `huggingface.co:443` is unreachable from the board (IPv4 connect timeout /
  IPv6 network unreachable); hf-mirror.com serves byte-identical files at the same path
  (general internet OK: tuna / hf-mirror / modelscope all 200). No token, no auth.
- Downloaded 2026-07-09 with `wget -c` (~35 MB/s, "saved [4920736608/4920736608]").

## GGUF metadata (verified on board via a self-contained header parser, no llama.cpp / gguf pip)
- GGUF version: 3
- tensor_count: 292   kv_count: 32
- `general.quantization_version = 2`
- `llama.block_count = 32` (n_layers)
- `llama.attention.head_count = 32`
- `llama.embedding_length = 4096`
- `llama.context_length = 131072`
- tensor ggml_type histogram (confirms q4_K blocks present):
  - `Q4_K` (12): 193 tensors
  - `Q6_K` (14):  33 tensors
  - `F32`  (0):  66 tensors

  → standard Q4_K_M mix (bulk weights Q4_K, a few Q6_K, norms/biases F32).

## Reproduce (board)
    ssh rvv
    cd ~/models && wget -c -O DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
      "https://hf-mirror.com/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    sha256sum DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf   # -> 87bcba20...896f5d
    python3 ~/gguf_meta.py DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf  # -> file_type MOSTLY_Q4_K_M

Parser (`~/gguf_meta.py` on the board) is a scratch helper, not part of the repo.
