**English** · [繁體中文](TWO_PC.zh-TW.md)

# Two PCs: one model split across a home network

This fork adds one feature to Strata: **run one model across two PCs on the same LAN**. The first PC runs the first
K layers, the second PC runs the rest and the head. Each PC keeps its own layers' experts on its own GPU and in its own
RAM, so two ordinary machines together can hold a model that neither can hold alone.

It was built for one concrete case: Qwen3.8-Flash-Next **IQ3_XXS (47 GB of RAM + VRAM)** on a desktop with 32 GB RAM
and a gaming laptop with 16 GB RAM. On the desktop alone it runs at about **10 tokens/s** (the experts that do not fit
are read from the SSD in small random reads). Split across the two PCs it writes at **50-62 tokens/s**.

- [Results](#results)
- [How it works](#how-it-works)
- [What you need](#what-you-need)
- [Build](#build)
- [Run](#run)
- [Tuning: what mattered](#tuning-what-mattered)
- [Limits and caveats](#limits-and-caveats)
- [Why not an existing multi-PC engine](#why-not-an-existing-multi-pc-engine)

## Results

Test machines:

| | First PC (desktop) | Second PC (laptop) |
| --- | --- | --- |
| GPU | RTX 3080 Ti, 12 GB | RTX 4060 Laptop, 8 GB |
| RAM | 32 GB | 16 GB DDR5-5600, **one stick (single channel)** |
| CPU | desktop CPU with AVX-512 | i7-14650HX (no AVX-512) |
| Network | 1 Gbps wired, ~2 ms round trip | same |
| OS | Windows 10 | Windows 11 |

Model: Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS**, split at **K = 32** (the laptop runs layers 32-47). Server path, 32K
context, temperature 0.6, 800 tokens, English prompts. All runs with no expert reads from the SSD (see
[Tuning](#tuning-what-mattered) for why that matters).

| Task | Warm-up run | Measured run |
| --- | ---: | ---: |
| Chat | 50.0 t/s | 44.3 t/s |
| Code 1 (INI parser) | 64.9 t/s | 61.5 t/s |
| Code 2 (LRU cache) | 66.9 t/s | 60.5 t/s |
| Code 3 (TS debounce) | 62.4 t/s | 50.4 t/s |
| Long prompt, 4,132 tokens: reading | 228 t/s (18.2 s) | |
| Long prompt: writing | 48.1 t/s | |
| Second turn of a conversation (4,268 tokens, 4,237 reused) | 2.1 s total | |

How the speed got there (decode t/s, three short prompts: code / story / explain):

| Step | code | story | explain |
| --- | ---: | ---: | ---: |
| Desktop alone, no speed-ups | 10.5 | 11.1 | 10.4 |
| Two PCs, K = 36, no speed-ups | 17.1 | 16.0 | 28.6 |
| + MTP draft layer, RAM headroom tuned, K = 32 | 27.7-29.6 | 19.0-19.6 | 32.3-35.1 |
| + laptop power plan "High performance" | 56.5 | 38.1 | 58.1 |
| + adaptive expert cache on both PCs | 68.8-70.7 | 46.8-48.8 | 51.2-52.8 |

For comparison, a dense 27B model split across the same two PCs with llama.cpp RPC (+ MTP) wrote code at about
52 t/s and chat at 28.8 t/s, and read prompts at about 400 t/s.

**Real-world use is slower.** With a browser, an editor and a chat app open on the desktop, only about 15 GiB of RAM
is free, about 7 GiB of experts are read from the SSD, and code runs at 40-43 t/s after warm-up.

## How it works

```
 First PC (desktop)                               Second PC (laptop)
 ┌──────────────────────────────┐   TCP, LAN    ┌──────────────────────────────┐
 │ server, tokenizer            │               │ strata --net-listen PORT     │
 │ layers 0 .. K-1              │ ── rows ────▶ │ layers K .. 47 + head        │
 │ its experts: GPU + RAM (+SSD)│               │ its experts: GPU + RAM only  │
 │ MTP draft layer              │ ◀── picks ─── │ sampling                     │
 └──────────────────────────────┘  (+ last R)   └──────────────────────────────┘
```

- Strata already splits a model by layers across two GPUs in one PC (`--layer-split`). The hand-off between stages
  is one place in the verifier: each token's row of `hc × n_embd + n_embd + hc` floats. This fork sends that row over
  TCP instead of mapped pinned memory. The engine's math is unchanged.
- Flash-Next's hidden size is 2560, so a token's hand-off is about **10 KB**. One verify window (4-5 tokens) is one
  round trip. Measured network cost: about **3.5 ms per window**; it is not the bottleneck.
- The second PC samples the tokens itself and sends back only the picks. When the first PC runs the MTP draft layer,
  the reply also carries the last layer's residual rows (T × hc × n_embd floats, about 160 KB for T = 4).
- These speed-ups were reconnected across the two PCs one by one, each checked against the one-PC output:
  - **MTP draft layer**: stays on the first PC, fed by the residual rows sent back.
  - **Conversation checkpoints**: each PC saves and restores its own layers' state, so the next turn only reads the
    new tokens.
  - **Batched prompt reading**: the first PC reads chunk c + 1 while the second reads chunk c.
  - **Adaptive expert cache**: each PC adapts its own layers' cache; the second PC runs its rounds between windows.
  - **Control vectors** (`--control-vector-scaled`): the per-request switch is forwarded to the second PC.
- Protocol (`include/strata/core/net_stage.hpp`, version 4): messages `Hello`, `Run`, `Commit`, `Zero`, `Logits`,
  `Bye`, `Save`, `Restore`, `Prefill`, `Cvec`, each `[u32 magic][u32 op][u64 bytes][payload]`.

Code changes: `src/core/net_stage.cpp` and `include/strata/core/net_stage.hpp` (new), `src/program/generate.cpp`
(options, per-PC loading, the second PC's message loop), `src/core/verify.cpp` / `.hpp` and `src/prefill/prefill.cpp`
/ `.hpp` (the hand-off goes to the network when the next stage is remote), `CMakeLists.txt`.

## What you need

- Two PCs with NVIDIA GPUs on the same LAN, **wired** if possible. Wi-Fi works but its latency jitter costs every
  window.
- Enough RAM + VRAM across both PCs for the model. For IQ3_XXS: about 47 GB, not counting the 27 GB lookup table
  that stays on the SSD.
- The same `strata` build on both PCs (the protocol version is checked at `Hello`).
- The model files on both PCs. The second PC only reads the GGUF headers, the non-expert tensors and the expert
  blocks of layers K and above, so it can hold a sparse copy with only those bytes.

## Build

Build from source with CUDA. Two flags matter:

- **`-DSTRATA_PORTABLE=ON`** if either CPU lacks AVX-512 (most laptops). A native build compiles ggml-cpu with AVX-512
  and crashes on the other PC. Use the portable build on **both** PCs.
- **`-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`** with CUDA 12.x on Windows, or linking fails (LNK1169).
- `-DCMAKE_CUDA_ARCHITECTURES="86;89"` (or your two GPUs' architectures) lets one binary run on both PCs.

```bat
:: from a "x64 Native Tools" prompt (Visual Studio 2022 Build Tools), in the repository folder
cmake -G Ninja -S . -B build-port -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
  -DSTRATA_PORTABLE=ON -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CUDA_ARCHITECTURES=86;89" ^
  "-DSTRATA_GGML_DIR=<path to llama.cpp, as set up by setup.py>"
cmake --build build-port --target strata -j 8
```

Copy `build-port\strata.exe` (and the CUDA runtime DLLs it needs) to the second PC.

## Run

Both PCs get the same engine options as a normal one-PC run (pack, model, `--expert-profile`, `--resident-experts`,
`--spec`, ...), plus:

| Option | Where | Meaning |
| --- | --- | --- |
| `--net-listen PORT` | second PC | wait for the first PC on PORT (all interfaces) |
| `--net-split K` | both | the second PC's first layer (the same K on both) |
| `--net-connect HOST:PORT` | first PC | connect to the second PC |

Requirements checked at start: `--resident-experts` and `--expert-profile` are needed; `--layer-split` (two GPUs in
one PC) cannot be combined with the network split.

1. **Second PC first.** Start `strata.exe <options> --net-listen 47700 --net-split 32`. It loads its layers and prints
   `NETREADY` when it is ready to accept the first PC.
2. **First PC.** Start the server as usual with `--net-connect <second PC IP>:47700 --net-split 32` added to the
   engine `args` of the server config (`python -m serve.server --engine strata --config <config.json>`).
3. When the first PC disconnects, the second PC waits for the next connection, so it can stay running between
   sessions. Stop it when you need its GPU back.

Useful environment variables:

| Variable | Use |
| --- | --- |
| `STRATA_RESIDENT_HEADROOM_GIB` | RAM left free on that PC. We use 3 on the desktop and 1 on the laptop. |
| `STRATA_WATCHDOG_S` | stall watchdog in seconds; raise it if the second PC's SSD can stall (we use 120-600) |
| `STRATA_NET_TRACE=1` | log every network message |
| `STRATA_NET_EXACT=1` | test mode: with a fixed `--expert-cache N`, each PC holds exactly the experts a one-PC cache of N slots would hold for its layers, so greedy output can be compared token for token |

Our best settings (K = 32):

- First PC: `--spec 4 --spec-min-p 0.5 --mtp <mtp dir> --prefill 2048 --pcie-frac 0 --suffix-draft 0 --expert-cache auto --vram-reserve-mib 700`, `STRATA_RESIDENT_HEADROOM_GIB=3`.
- Second PC: `--spec 4 --vram-reserve-mib 300`, `STRATA_RESIDENT_HEADROOM_GIB=1`.

## Tuning: what mattered

In order of how much speed each one cost or gave:

1. **The laptop's Windows power plan.** "Balanced" throttles the CPU that computes the RAM-resident experts: the
   laptop's part of a window took 78 ms. "High performance" brought it to 24 ms and **doubled the total speed**
   (27.7 → 56.5 t/s). Check with `powercfg /getactivescheme`.
2. **Never let experts spill to the SSD.** When 1-4 GiB of the desktop's experts did not fit in RAM, speed dropped
   20-50% (56-59 → 29-44 t/s). Other programs on the PC are the usual cause. Compare speeds only between runs with
   the same free RAM.
3. **Where to split (K).** Give the second PC exactly as many layers as fit in its VRAM + RAM with no SSD reads, and
   the rest to the PC with more RAM. K = 36 → 32 moved the desktop from about 36 GB of SSD reads per prompt to none.
4. **MTP helps only once nothing spills.** Checking 4-5 drafted tokens needs 4-5 times the experts. With experts on the
   SSD, MTP gave almost nothing; with everything in RAM it is a large gain.
5. **Prompt chunk size.** `--prefill 2048` read a 4.5K-token prompt at 448 t/s, versus 344 at 8192 (one chunk, so no
   overlap between the two PCs) and 364 at 1024.
6. **Do not force more CPU threads.** Strata uses one worker per physical core. Forcing more (`--pool-workers`) was
   slower on both PCs: the expert math is bound by memory bandwidth.
7. **Laptop housekeeping.** Remove heavy background software, use a fixed page file. On our laptop this raised the
   commit limit from 18.7 GB to 31.7 GB, which Strata needs to pin its experts in RAM.

Tried and not worth it: `--spec 3` or `5` (4 is best), PCIe split for missed experts (no change), suffix drafting
(helps only on edits that repeat earlier text).

## Limits and caveats

- **No authentication.** `--net-listen` accepts any connection on the port. Use it only on a home LAN and limit the
  firewall rule to the local subnet.
- **Two PCs, one GPU each.** No more than two stages; not combined with `--layer-split` or helper GPUs.
- **Off in two-PC mode:** conversation parking (`conversation_cache_mib`), batching several requests (`parallel`).
  `window_logprobs` is not available.
- **Different GPU generations do not round identically.** On the same GPU model the split output is token-for-token
  identical to one PC (checked on prompts up to 4,461 tokens). Between an RTX 3080 Ti (Ampere) and an RTX 4060 (Ada),
  greedy output can pick a different word after some dozens of tokens; the text stays coherent. This is the hardware,
  not the network code.
- **First prompt reading is slower than one PC.** About 230-255 t/s at 32K context on our machines. Later turns reuse
  the checkpoints and only read the new tokens.
- Tested on Windows only. The socket code has a BSD-sockets path for Linux, untested.
- Based on Strata commit `d5ea713` (0.1.40.3). Upstream has moved on since; this branch has not been rebased.

## Why not an existing multi-PC engine

Checked on 2026-10-08, from each project's own documentation:

| Project | Why it did not fit |
| --- | --- |
| prima.cpp | no Windows; no MoE models of this architecture |
| exo | Windows + CUDA listed as long-term |
| distributed-llama | CPU / Vulkan only; up to Qwen3 30B-A3B |
| Parallax | weights must fit in VRAM (20 GB across our two PCs); WSL2 |
| llama.cpp RPC | runs the model, but on one PC with everything in RAM llama.cpp writes 9-16 t/s where Strata writes 42-58; across two PCs it can only be slower (we measured 6 t/s on the Coder model) |
| Strata upstream | the author does not plan to split across machines (issue #1256) |

Tensor parallelism across PCs (both PCs computing every layer) was also measured with llama.cpp RPC `-sm tensor` on
wired gigabit: 0.6-0.7 t/s. Layer splitting is the only practical option without RDMA networking.
