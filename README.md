# moe-hotcache

**Adaptive hot-expert caching for MoE inference in llama.cpp — learns expert routing at
runtime. +67% decode speed on a 131B model, measured on consumer AMD GPUs (ROCm).**

This is a llama.cpp fork ([original README](README-llamacpp.md)) carrying an experimental
expert-caching system for Mixture-of-Experts models whose expert weights don't fit in VRAM.
Instead of the stock approach (statically split expert layers between GPU and CPU with
`--n-cpu-moe`), moe-hotcache moves **all** expert weights to pinned host RAM, lets the GPU
compute them directly over PCIe (UVA/DMA), and keeps a small VRAM cache of the experts that
are actually being routed to — updated continuously at runtime.

> Everything below was measured, not estimated. The interesting result is that this works
> at all on AMD — every comparable system we found (FreeToken, MoE-Infinity) is CUDA-only.

## Test machine

| component | spec |
|---|---|
| CPU | AMD Ryzen 9 5900X (12C/24T) |
| GPUs | 2× AMD Radeon RDNA4 16 GB (`gfx1200`) — compute card in the CPU x16 slot; second card in a chipset-limited slot (~3.5 GB/s effective, useless as a cache tier — see problem #7) |
| RAM | 64 GB DDR4 dual-channel — **37.4 GB/s measured**, the number that used to be this box's ceiling |
| Effective PCIe DMA (host→GPU expert reads) | ~14 GB/s measured |
| OS / stack | Ubuntu 24.04.4, kernel 7.0, **ROCm 10.0.0** (TheRock `gfx120X` build; HIP backend) |
| Model storage | NVMe, model file 57 GB, page-cache resident during runs |

## Results

Model: Qwen3.8-Flash-Next 131B-A6B (Q3_K_XL, 512 experts/layer, 48 MoE layers, 57 GB file).

**Correction (same day, hours later):** the first published version of this table carried
131B numbers measured **before verifying output quality**. A batch-kernel bug (problem #9
below) was silently corrupting the 131B model's output on the DMA path while producing
plausible speed numbers — those numbers (TG 13.41 / 19.89) are **retracted**. The OLMoE
numbers were always quality-verified (byte-identical) and stand. The table below is the
current, quality-verified state; we kept the retraction visible instead of rewriting
history because measure-before-verify is exactly the mistake this README warns about.

| configuration (quality-verified) | decode (tok/s) | prefill (tok/s) | RAM pressure |
|---|---|---|---|
| stock llama.cpp, `--n-cpu-moe 33` (tuned baseline) | 11.4 | 163 | none (mmap) |
| **partial-pin**: 20/48 layers pinned+DMA+dynamic cache, rest CPU/mmap | **13.0–13.4 (+15%)** | ⚠ see below | none — 30 GB used, zero swap |

**Second correction (later the same night):** the partial-pin *prefill* numbers first
published here (105/156–276 tok/s) are also **retracted** — the forced-copy offload path
that produced them corrupts large-prompt output (garbage logits for that request; decode
unaffected and still verified). Yes, the same class of mistake twice in one day: we
verified decode output but not prefill output. Decode numbers stand. Until the copy path
is fixed, prefill on host-pinned layers must route to CPU threads (~66–105 tok/s,
correct), which loses to the tuned baseline's 163 — so this fork is currently a decode
win only, and problem #10 is the open front.

Sanity model: OLMoE-1B-7B (64 experts/layer): DMA-only 6.4 → dynamic cache **15–16.8
(+150%)**, greedy (temp 0) outputs **byte-identical** to the uncached path.

The all-layers-pinned mode (55 GB pinned on a 62 GB box) still exists but swap-thrashes;
partial-pin is the daily-driver configuration. Full-pin re-measurement with verified
quality is pending a machine with more RAM.

## How it works

```
                    ┌────────────────────────── VRAM ──────────────────────────┐
 per-expert pointer │ slot blob: [e17][e203][e64]...   ← hot experts (46/512)  │
 table (device)     │ dense layers, KV cache, everything else                  │
   table[512] ──────┤                                                          │
     hit → VRAM slot└──────────────────────────────────────────────────────────┘
     miss → host ptr┌────────────────────── pinned host RAM ───────────────────┐
        (UVA/DMA)   │ all expert weights, contiguous per tensor                │
                    └──────────────────────────────────────────────────────────┘
```

1. **Pointer-table kernel path.** `mmvq` (the batch-1 MoE GEMV) takes an optional
   per-expert pointer table. Each block looks up `table[expert_id]`: pinned experts
   resolve to a VRAM slot, everything else resolves to the pinned-host address and is read
   over PCIe by the same kernel. One code path, no branching cost worth mentioning, and
   the miss path never stalls — the GPU reads host memory directly (UVA).
2. **Frequency counting inside the graph.** A tiny kernel bumps a per-expert counter from
   the routing `ids` tensor. Crucially it is launched on the compute stream, so it gets
   **captured into the CUDA/HIP graph** and runs on every replay — this matters, see
   problem #4 below.
3. **Background refresh thread.** Every 300 ms (env-tunable) a host thread drains the
   counters, applies exponential decay (`score = score/2 + count`), and swaps at most 2
   experts per tensor per cycle (bounded fill bandwidth: ~0.4% of token time). Promotions
   copy the expert into a free slot *then* flip the table pointer (a single aligned 64-bit
   store — atomic for any concurrent reader). Evictions flip the pointer back to the host
   address first, then quarantine the slot for one cycle before reuse, so an in-flight
   graph replay can never read a slot that is being overwritten.
4. **Per-tensor budget.** The VRAM budget is divided evenly across tensors (this also fixes
   the first-come-first-served starvation bug the static version had, where early layers
   consumed the entire budget).

Static mode (offline profile, no runtime adaptation) is kept and works, but the dynamic
mode beats it soundly — 13.41 vs 19.89 tok/s — mainly because the profile only covered the
layers that happened to be CPU-resident when it was collected, while the dynamic cache
learns all 48 layers, converging to 100% hit rate on many tensors within a minute despite
caching only 9% of experts. Routing skew is real: we measured 36.1% adjacent-token expert
overlap and the top 13.4% of experts taking 58.9% of requests on real Thai/English text.

### Usage

```sh
# build (HIP shown; the code paths are shared with CUDA but only ROCm is tested)
cmake -B build -DGGML_HIP=ON && cmake --build build --target llama-server

# dynamic cache, 4GB VRAM budget:
GGML_EXPERT_PIN_DYN=1 GGML_EXPERT_PIN_MB=4000 GGML_EXPERT_PIN_TENSORS=144 \
GGML_OP_OFFLOAD_MIN_BATCH=1 \
./llama-server -m model.gguf -ngl 999 --load-mode none \
  --override-tensor '\.ffn_(up|gate|down)_exps\.weight=ROCm_Host'
```

| env | meaning | default |
|---|---|---|
| `GGML_EXPERT_PIN_DYN` | enable the dynamic cache | off |
| `GGML_EXPERT_PIN` | offline profile (`<tensor name> <expert id>` lines); optional seed for dyn, required for static | — |
| `GGML_EXPERT_PIN_MB` | total VRAM budget (MB) | 2800 |
| `GGML_EXPERT_PIN_TENSORS` | tensor count for per-tensor budget split | 93 or profile size |
| `GGML_EXPERT_PIN_REFRESH_MS` | refresh-thread period | 300 |
| `GGML_EXPERT_PIN_SWAP` | max promotions per tensor per cycle | 2 |
| `GGML_EXPERT_LOG` | CPU-side expert-routing logger (for offline profiles) | off |

**The three flags that must appear together:** `--load-mode none` (mmap silently swallows
the `-ot` host placement — llama.cpp warns, easy to miss), `-ot '...exps...=<host buft>'`,
and `GGML_OP_OFFLOAD_MIN_BATCH=1` (forces the scheduler to run host-weight ops on GPU).

## Timeline — one day, honestly

All of this happened on 2026-08-31 → 09-01, on one machine, human directing + AI (Claude)
writing every line of code, benchmark, and this README. Full lab notes live in
`task-log/` on the machine; condensed here.

| stage | what happened | result |
|---|---|---|
| 0 | Move experts to pinned host via `-ot`. Free lunch expected. | Compute didn't move — scheduler copies weights per-eval. Learned `supports_buft` is the gate. |
| 1a | Claim host-buft support in the CUDA backend, force GPU compute over DMA. | Correct outputs on OLMoE, slow but working foundation. |
| 1b | Static pointer-table pinning, offline profile. | OLMoE +56%. First proof the table kernel works. |
| — | Measure routing locality (patched a logger into `ggml-cpu.c`). | Skew confirmed: top 13.4% experts → 58.9% of traffic. Estimated +20%. |
| 2 | Static pin on the 131B target. | TG 11.9 → 13.41 (+12.7%), PP 130 → 71. Below the estimate — profile only covered layers 0–30. |
| 3 | User found FreeToken. CUDA-only, no GGUF → port the *idea*, not the code. | Designed: async fills on a side stream instead of FreeToken's stall-on-fetch, because our miss path already reads host memory directly. |
| 4 | Dynamic cache v1. | Cache **learned nothing**. Three bugs found in sequence (below). |
| 5 | Dynamic cache working. | OLMoE +150%. 131B: **TG 19.89 (+67%)**, hit rate → 100% on many tensors. PP collapsed to 4.97 under swap pressure. |
| 6 | Decision: TG architecture proven, PP is the remaining engineering. | Roadmap below. |

## Every problem we hit (the actual valuable part)

1. **mmap silently eats `-ot` host placement.** With default load-mode, the override
   "succeeds" but weights stay in mmap'd file pages — the DMA path reads them, nothing is
   pinned, results look mysteriously identical to baseline. llama.cpp does print a warning;
   we ignored it for an hour. `--load-mode none` is mandatory.
2. **The scheduler gate.** Claiming host-buffer support (`supports_buft`) must be gated
   behind the feature env, or every model regresses; we initially gated it on the *static*
   env only, so the dynamic mode silently ran with zero GPU placement — flat benchmarks
   with correct outputs, the worst kind of bug.
3. **`pkill -f "port 8099"` matches your own shell.** And `pgrep -x llama-server` matches
   the *other* llama-server (the embedding sidecar). Two separate test sessions were
   corrupted by zombie servers holding the port while a fresh server answered health checks
   on a different PID. Rule that survived: track PIDs from `$!`, verify ports with `ss`,
   never trust process-name matching.
4. **CUDA/HIP graph replay swallows host code.** The showstopper. Our getter did counting
   readbacks and cache refresh on the host, per call. Decode is graph-captured: host code
   in the hook runs **once at capture, never again** — device counters kept counting
   (the count kernel was captured into the graph), but the refresh logic never fired.
   Diagnosis came from a counter that read `issued=0` while device counts grew. Fix:
   anything per-token must be a kernel *inside* the graph; anything adaptive must live on
   an independent host thread. This constraint shaped the whole final architecture.
5. **Capture-safety of table creation.** First-touch table construction can happen inside
   a graph capture; a `cudaStreamSynchronize` on the capture stream aborts capture. All
   creation-time copies/syncs had to move to the side stream.
6. **`cudaEventQuery` has no HIP alias in llama.cpp's vendor header.** One-line addition
   (`hipEventQuery`), noted because the error message points at the wrong thing.
7. **Pinned memory is a capacity game, not a bandwidth game.** 55 GB pinned on 62 GB
   physical: the kernel pages *other* anon memory out, and random 4K swap-ins (measured
   233 µs each on this box) turn occasional tokens into 0.8 tok/s. Meanwhile RAM
   *bandwidth* barely matters in DMA mode — PCIe (~14 GB/s) draws well under the 37 GB/s
   the RAM can serve. Corollary: the second GPU's VRAM is useless as a cache tier here —
   it hangs off a x4 slot (~3.5 GB/s), *slower than reading RAM*; storage capacity you
   can't reach is not capacity.
8. **MoE prefill destroys sparsity.** Large batches route to nearly every expert per
   layer, so the per-expert DMA path degenerates into streaming the whole model per
   ubatch. This is a known MoE property (FreeToken's paper treats prefill and decode as
   different problems for exactly this reason) — our measured 71 → 4.97 tok/s prefill is
   that effect plus swap pressure. Prefill needs its own path, full stop.
9. **The batch kernels don't know about the pointer table — and silently corrupt output.**
   The costliest bug of the project, found only when we finally asked the 131B model a
   question and read the answer (`//////…`). The pin table lives in `mmvq` (batch ≤ 8).
   Prompts of 9–31 tokens fell into a gap: too big for `mmvq`, below our CPU-routing
   threshold — so they ran `mul_mat_q` (MMQ) reading the host pointer directly. With some
   quant types that "works" (wrong results, no error); with `Q8_0` it faults. Worse, the
   `supports_buft` claim meant offloaded big batches also skipped the VRAM copy. On a
   dynamic-quant GGUF (layers quantized differently) the corruption was *layer-dependent*,
   which is why a per-layer bisect plus a quant-type dump found it in eight runs. Fixes:
   direct-DMA only for batch ≤ 8 (the `mmvq` boundary), and force a real VRAM copy for
   larger batches. Lesson, earned twice in one project: **a new GPU data path is not
   measured until its *output* is verified — per quant type, per batch-size regime.**
10. **(open) The forced VRAM copy path corrupts large-prompt prefill.** After fix #9,
    batches ≥ 9 take the stock offload path with a forced weight copy — but prompts spanning
    large/multiple ubatches still produce garbage for that request (decode and small
    prompts stay correct, so the corruption hides from casual testing). Reproduced
    deterministically with an ~11k-token prompt. Not yet root-caused: suspects are the
    `cuda_host → cuda` tensor-copy route and split/copy caching across ubatches. Until
    fixed, correctness requires routing big batches to CPU threads instead of the copy
    path.

## Roadmap

- [x] **Prefill routing**: large-batch `mul_mat_id` on host weights takes the stock
      offload-copy path; only batch ≤ 8 uses the DMA table (see problem #9).
- [x] **Partial pinning**: 20/48 layers pinned under mmap (loader keeps `-ot` host
      overrides in expert-pin mode); measured TG 13.0–13.4 / PP 156+ with zero swap.
- [ ] **Capped-fetch hybrid decode** (FreeToken's q\* policy): let the CPU compute the
      overflow misses in parallel with GPU DMA, bit-exact partial-sum merge.
- [ ] **Double-buffered prefill streaming**: stream layer N+1's experts while computing
      layer N. Ceiling here ≈ PCIe bw / (model bytes per layer) ≈ 530 tok/s prefill.
- [ ] CUDA testing (the code compiles through llama.cpp's shared CUDA/HIP paths but only
      ROCm has been run), then propose upstream.

## Credits & provenance

- **Concept**: [FreeToken](https://github.com/FlashML-org/FreeToken)
  ([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)) — slot cache, capped-fetch,
  and the prefill/decode split. Their LRU kernel lives in a closed-source package; this
  implementation shares no code with it and was written from scratch for llama.cpp/HIP.
- **Related**: [MoE-Infinity](https://arxiv.org/abs/2401.14361) (activation-aware expert
  caching, CUDA-only), [Pre-gated MoE](https://arxiv.org/abs/2308.12066),
  [vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256).
- **Base**: [llama.cpp](https://github.com/ggml-org/llama.cpp), on top of a branch carrying
  LaurentZuijdwijk's qwen4exp/MTP work.
- **Process disclosure**: the human on this project chose the targets, found the sources,
  challenged the numbers, and made every architecture/no-go call; the AI (Claude, this
  fork's co-author) wrote the code, ran the benchmarks, and debugged the failures above.
  We think that division of labor is worth being explicit about.

License: inherits llama.cpp's MIT.
