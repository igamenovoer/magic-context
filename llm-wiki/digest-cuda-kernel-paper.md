# Digest a CUDA Kernel Paper

You are tasked with digesting a CUDA-kernel-oriented paper into a structured wiki entry. Unlike a generic paper note, a CUDA kernel digest must capture the low-level design decisions, the hardware rationale behind them, quantitative kernel-level measurements, and (when a public implementation exists) how the code realizes the described design.

This instruction targets papers whose primary contribution is a GPU kernel or a small family of kernels — for example FlashAttention, DeepGEMM, ScatterMoE, FlashMLA, SageAttention, BLASST, MegaBlocks, or any paper that names specific kernel instructions (MMA, TMA, cp.async, tcgen05, WGMMA, WMMA), memory tiers (registers, SMEM, TMEM, HBM, L2), or scheduling primitives (warp specialization, CTA clusters, persistent kernels, CUDA Graphs, PDL).

If the source is a systems or compiler paper where the kernel is incidental, fall back to a generic paper note instead.

---

## Required Inputs

Before producing the digest, make sure you have:

1. **The paper itself** — PDF, arXiv HTML, or extracted text. Read it end-to-end; do not skim.
2. **Target hardware** — the GPU generation the kernel is written for (SM80/A100, SM90/H100, SM100/B200, SM120/RTX5090, multi-generation, etc.). This gates which instructions and memory tiers are relevant.
3. **Source code status (optional)** — whether a public repo, gist, or supplementary source exists. Record:
   - Repo URL and commit hash (or release tag) used for the analysis
   - License
   - Whether the paper's claims match the released code (papers sometimes describe an internal variant)
4. **KB context (optional)** — if you are writing into an existing `llm-wiki`-style knowledge base (with `raw/`, `wiki/sources/`, `wiki/entities/`, `wiki/concepts/`, `wiki/synthesis/`), read the target `wiki/index.md` first so you can link to existing entity/concept pages instead of duplicating them.

If any of the above is missing, state it explicitly in the digest rather than guessing.

---

## Required Sections

A CUDA kernel digest MUST include the following sections, in this order. Omit a section only when the paper has no material on that topic; in that case write one sentence stating so (e.g. "The paper does not report ablations.") rather than dropping the heading.

### 0. Metadata

- **Full title**, **authors**, **affiliation**
- **Venue / year** (conference, workshop, arXiv-only)
- **arXiv ID / DOI / project page / repo URL**
- **Target architecture(s)**: SM80, SM90, SM100, SM120, cross-gen
- **Kernel family**: attention, GEMM, grouped-GEMM, MoE dispatch/combine, quantization, KV-cache, collective, etc.
- **Precision(s)**: FP32, TF32, BF16, FP16, FP8 (E4M3/E5M2), FP6, FP4 (NVFP4/MXFP4), INT8, INT4, mixed
- **Reference code available?** yes / no / partial
- **Date ingested** (YYYY-MM-DD)

### 1. TL;DR (3–6 bullets)

Bold key terms. Cover what the kernel does, the single most important hardware insight, and the headline number.

- **Problem**: …
- **Core insight**: …
- **Kernel technique**: …
- **Hardware target**: …
- **Result**: … (e.g. "1613 TFLOPS/s on B200, 1.3× vs cuDNN 9.13")

### 2. Kernel Design

Describe the kernel as the paper presents it. Research papers vary widely in how much low-level detail they expose — some go down to tile shapes and PTX, others stay at a block-diagram level. Do not invent detail the paper does not provide.

**Baseline (paper-only):** capture the key design points the paper actually discusses. Typically this includes:

- **Problem formulation**: mathematical operation, tensor shapes, any sparsity or structure the kernel exploits
- **Core idea**: the one or two design decisions the paper calls out as its main contribution (e.g. "software exp emulation on FMA units", "single persistent kernel fusing dispatch + compute + combine")
- **High-level dataflow**: the stages the kernel goes through per tile / per token / per expert
- A Mermaid flowchart is encouraged when the paper has a system diagram.

**Expanded (when source code is available):** once the code is in hand (see §6), extend the design section to cover whichever of the following the code reveals — the paper itself may be silent on most of them:

- **Tiling strategy**: block/warp/thread-fragment tile sizes; tile reshapes across kernel phases
- **Memory hierarchy plan**: what lives in registers, SMEM, TMEM (SM100+), L2, HBM; persistence/residency choices
- **Data movement**: cp.async / TMA / `tma_gather4` / `cp.reduce.async.bulk` / NVSHMEM calls; multi-stage pipeline depth
- **Compute units**: Tensor Core instruction family (`wmma`, `mma.sync`, `wgmma`, `tcgen05.mma`, `mma.block_scale`); CUDA core vs MUFU usage
- **Warp / CTA organization**: warp specialization (producer/consumer), CTA clusters / 2-CTA MMA, persistent-grid patterns, cluster launch control (CLC)
- **Synchronization**: mbarriers, named barriers, `bar.sync`, `fence.mbarrier_init.release.cluster`, signal/wait primitives
- **Epilogue / fusion**: what is fused into the kernel (bias, activation, softmax, scatter, quantize, reduce-scatter, all-gather, top-k)
- **Scheduling**: grid shape, launch bounds, occupancy target, CUDA Graphs / PDL interactions
- **Numerical details**: accumulator precision, scale formats (E8M0, UE8M0, FP8 scales), online-softmax rescaling, block-scale layouts

Clearly label which facts come from the paper vs. from the code; when the two disagree, surface the discrepancy and prefer the code for the "as-built" description.

Ground every non-obvious paper claim with a blockquote citation:

```markdown
> "We adopt a 128×128 MMA tile enabled by tcgen05.mma, doubling the per-instruction work over Hopper's wgmma." (Section 3.2; paper-source/flashattention-4.pdf#page=5)
```

### 3. Rationale (Why This Design)

Designs are interesting only in the context of the bottleneck they address. For every major design choice in §2, answer:

- **What bottleneck motivated it?** (e.g. "SMEM bandwidth is unchanged from Hopper while MMA throughput doubled, so softmax exp on MUFU becomes the new bottleneck")
- **What alternative was rejected, and why?** (tile-size sweeps, alternative pipeline stages, Hopper-style kernel on Blackwell)
- **Which hardware counter / microbenchmark supports the reasoning?** (tensor-core utilization, SMEM bytes/clock, MUFU ops/clock, register pressure, occupancy, stall reasons)

Rationale is usually scattered across the paper's introduction, motivation, and discussion sections. Gather it into one place. When the paper is terse, cite microbenchmarking papers or NVIDIA tuning guides from the same KB and mark the inference explicitly ("inferred — not stated in the paper").

### 4. Empirical Experiment Results

Capture the numbers precisely. Vague claims ("significant speedup") are not acceptable.

- **Hardware setup**: exact GPU SKU (A100 80GB SXM, H100 SXM5, B200 SXM, HGX B200, DGX B200), clock settings, CUDA/cuDNN/driver versions
- **Baselines compared**: name each baseline and its version (e.g. "cuBLAS 12.5", "cuDNN 9.13", "Triton 3.6", "CUTLASS 4.2", "FlashAttention-3 HEAD@…")
- **Workload geometry**: batch, sequence length, head dim, hidden dim, expert count, top-k, sparsity pattern, block size
- **Metrics reported**: TFLOPS/s, TOPS, GB/s, kernel latency (µs/ms), end-to-end TPS, tokens/s, perplexity loss, MSE vs reference, occupancy, SM utilization, stall reasons
- **Headline numbers**: reproduce the main result table as a compact Markdown table when feasible
- **Scaling trends**: how performance varies with sequence length, batch size, expert count, precision, sparsity ratio
- **Ablations**: which component contributes how much of the speedup (e.g. "TMEM accumulator alone: +18%; software exp emulation: +22%")
- **Error / quality budget**: for lossy kernels (FP4, sparse attention, dropless MoE) record the accuracy tradeoff

Cite each number with section/table/figure reference so a future reader can verify.

### 5. Limitations

Summarize the limitations the paper itself states — typically found in the discussion, conclusion, or a dedicated limitations section. Keep it to what the authors acknowledge; do not speculate. If the paper states no limitations, write one sentence saying so.

### 6. Source Code Analysis (if source code is available)

Search online (paper footnotes, project page, arXiv abstract, GitHub/GitLab, paperswithcode) for a public implementation. If none exists, write one sentence — "No public implementation at the time of digest." — and stop the section.

If found, download it locally:

- **Destination**: prefer a project-scoped tmp directory (e.g. `tmp/`, `.tmp/`, or an existing scratch dir at the repo root). If none exists, fall back to the system tmp (`/tmp/` on Linux/macOS, `%TEMP%` on Windows).
- **GitHub/GitLab repos**: shallow clone with `git clone --depth=1 <url>` (add `--branch <tag>` if the paper pins a release). Record the resolved commit hash.
- **Non-git sources** (tarballs, zips, gists): download, extract, note the source URL and fetch date.

Then analyze the code with tight focus on what the paper actually describes:

- Identify the kernel entry point(s) and the few files that implement the paper's core contribution. Ignore unrelated code.
- Note any material deviation between the paper and the code.
- **Present the key algorithm as CUDA-flavored pseudocode** in the digest — tiling, memory moves, MMA calls, synchronization — with shape/precision annotations in comments. The pseudocode should let a reader understand the kernel without opening the repo.
- Cite concrete locations with `path/to/file.cuh:line_range` when making specific claims.

---

## Style Rules

- **Ground every non-trivial claim** with a blockquote quoting the paper, the repo, or an authoritative secondary source. Put the citation immediately after the claim, not at the end of the section.
- **Preserve numbers exactly** as reported. Do not round TFLOPS figures, speedups, or memory footprints. If the paper gives a range, keep the range.
- **Distinguish stated from inferred** with an explicit marker: `(inferred)` or `(not in paper — derived from NVIDIA B200 tuning guide)`.
- **Use Mermaid** for kernel pipelines and dataflow. Use tables for result comparisons, tile-size sweeps, and ablations.
- **Shape annotations** on any pseudocode comment: `# Q: (B, H, S, D); K,V: (B, H, S_kv, D)`.
- **Never invent** instruction mnemonics, register counts, or latencies. If you don't know, say you don't know.
- **Cross-generation claims** must name the generation. "Tensor Core MMA" is ambiguous; "Hopper `wgmma` 64×256×16 BF16" is not.
- **Kernel-vs-end-to-end**: always label whether a number is kernel-level or model/system-level. Do not compare across categories.

---

## Quality Checklist

Before saving the digest, verify:

- [ ] Metadata lists target architecture(s) and precision(s).
- [ ] Kernel Design covers tiling, memory hierarchy, data movement, compute units, warp/CTA organization, and epilogue/fusion (or explicitly states the paper omits a topic).
- [ ] Every design choice in §2 has a matching bottleneck/rationale in §3.
- [ ] §4 specifies exact GPU SKU, baseline versions, and workload geometry.
- [ ] §5 separates author-stated limitations from reader-inferred ones.
- [ ] §6 is either one sentence ("No public implementation…") or has repo URL + commit + file map + deviations.
- [ ] Every paper claim has a blockquote citation with section/table/figure reference.
- [ ] Every code claim has a `file:line_range` citation.
