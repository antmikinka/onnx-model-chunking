# Work record

Start with [EXPLAINER.md](EXPLAINER.md) if you do not already know this repository. The challenge narrative is [CHALLENGES.md](CHALLENGES.md). This file is the number ledger for the Qwen3-8B split work in this repository through the commit that adds it. Measurements below were taken on one machine. A passing unit test is not an NPU placement. An ELF line is not a placement unless `session.run` returned a finite tensor. A Vitis-versus-CPU gap of 0.0 is lowering fidelity against that CPU session, not agreement with the packed INT4 monolith.

Weights file `model.onnx.data` (7,241,596,928 bytes) was not rewritten. `original-info-signature.txt` and `original-model-signature.txt` stay untracked. `C:\Users\antmi\qwen3-work` is not this git repository. Caches, logs, extracted ONNX graphs, and the 7 GB weight file stay there.

## Machine

- CPU: Ryzen AI 9 HX 370 (Strix, XDNA2). Driver 32.0.203.329.
- Ryzen AI installs on disk: 1.5.1, 1.6.0, 1.7.0, 1.8.0 under `C:\Program Files\RyzenAI`. There is no 1.7.1 install.
- Sessions that talk to Vitis use conda env `ryzen-ai-1.8.0` through `conda activate`. Calling that env's `python.exe` by path leaves `RYZEN_AI` unset.
- That env is ONNX Runtime 1.27.0 and onnxruntime-vitisai 1.27.0. Providers include Vitis, DirectML, and CPU.
- `import quark` in that env raises `ModuleNotFoundError`. Quark was not pip-installed there. A toy MatMul was quantized in the already-installed 1.7 Quark. That toy is not the 8B model.
- `import xrt` is absent from 1.8.0. Device names `vitisai` and `ryzenai` raise unsupported device type. `npu` plus vendor `0x1002` cannot place an OrtValue. Host OrtValues stay on CPU.
- PowerShell 5.1. A batch file whose last line is `echo PY_EXIT:%ERRORLEVEL%` returns 0, because `echo` clears `ERRORLEVEL`. The toy runners save `RC` first and then `exit /b %RC%`.

## Repository state

Origin is `https://github.com/antmikinka/onnx-model-chunking.git`. The repo is private.

`origin/main` at the start of this commit is `e2ac77b`, the merge of pull request #8. This branch is `cursor/quark-int8-pr10`. It had no upstream. Pull request #10 is a different branch, `cursor/amd-quark-optimize-18aa`, at `694ccfa`, based on old main `05518f0`. It stays open. This push does not force-update that branch.

`quark_optimize.py` and `test_quark.py` in this tree hash-match the files on pull request #10 (`f305ee085168bda18c9e1c06f82e56a8cd0a972e` and `8063ff2c06c8cda29a9be255c480f5dfc6ffce3d`). The splitter and package export on this branch wire that same stage into current main, which pull request #10 does not contain.

Local branch `experiment/xdna-layout-probe` is commit `c11039f`. It was not merged and was not pushed by this record.

## What the unit tests just did

From `MASTER_POC`, conda env `ryzen-ai-1.8.0`:

- `test_quark.py` exited 0. Bad bit widths, empty paths, and an unknown calibration name are rejected. Stage order is rewrite, then Quark, then split. A missing Quark install raises `QuarkUnavailable` and writes no ONNX file. `split_onnx(..., quark=...)` defaults to off.
- `test_static_shapes.py` exited 0. The new checks are `gqa_position`, `component_bake`, and `external_slice`. Printed gaps from the older checks, re-run in this session: down-projection K-split max abs `2.122e-05` against a reference max abs of `11.21`; bf16 bake gaps `2.080e-03` (reference max abs `2.312`) and `7.820e-03` (reference max abs `4.984`).

## Merged pull requests

### #1 through #4

- #1, `cursor/amd-ortvalue-io-binding-fdef`, merged. Title: allocate AMD OrtValues on NPU and iGPU memory. Later sessions on onnxruntime-vitisai 1.27 keep OrtValues on CPU. The publish path that ran is a host pointer swap after `session.run`.
- #2, shared quantization and KV contracts, plus pip packaging.
- #3, route split inference by phase.
- #4, keep GenAI scale and zero-point fields on the shared `QuantizationInfo`.

### #5 KV concat and pointer swap

Merged as `b0dfcf5` from `cursor/npu-kv-concat-swap-18aa`. Pull request #11 (`21ed86f` and follow-ups on `fix/npu-kv-cpu-ortvalue`) published the swap after a real ONNX session and was merged. The landed attention toys:

- `fixed_window` is a one-axis Slice plus Concat. Scatter, ScatterND, and ScatterElements are rejected. A CPU session turned past `[0..7]` plus token 9 into `[1,2,3,4,5,6,7,9]`.
- Windows `VirtualLock` page-locks the CPU OrtValue. That is not HipPinned memory and not XRT DMA registration.
- The first Vitis graph was two Identity nodes, shape `[1,1,4,4]`, float32, IR 10. It printed a 131416-byte AIE2P ELF. OrtValues stayed on CPU. The swapped host buffer matched the prefix written before the run. That is not attention, not MatMul, and not a tokens-per-second result.
- Commit `b07e2f5` is a 9-node golden graph that Vitis ran as one `vitis_ai_ep_1` kernel. CPU matched the float32 reference (key diff 0, value under `1e-6`). `query_len` other than 1, and a head size that is not a multiple of 16, are rejected. The graph uses a constant Q column and no projection weights.
- Commit `37bf7fc` adds `build_projected_attention_model`: static Slice/Concat cache `[1,1,8,64]`, opset 17, IR 10, with Wq/Wk/Wv. The CPU unit test passed. It is not a Qwen layer.

ONNX models this runtime loads use IR version at most 13. IR 10 is what ran. IR 14 failed.

### #6 Additive mask

Merged as `d9ced35` from `cursor/additive-attention-mask-18aa`. Attention Where/MaskedFill becomes an additive bias of `-10000`. The masked projected block at `94666bd` is a 15-node one-head decode graph. No Where. No Transpose.

### #7 Prefill and decode as two graphs

Merged as `cefd3ca` from `cursor/phase-split-prefill-decode-18aa`. Static prefill and decode are separate graphs. Heads 8 and head size 128 were faster on Vitis at prompt lengths 256 and 512 for that toy layout. `SessionManager` does not read that constant. These graphs are not the Qwen decode path. The Qwen graphs that were compiled are query length 1. A prefill graph, if one is built later, must feed `position_ids` as `[[0, 1, ..., query_len-1]]`, not a scalar.

### #8 LM-head tile width

Merged as `e2ac77b` from `cursor/vocab-sram-split-18aa`, feature commit `dd103e8`. LM-head remainder tiles are zero-padded to one width and sliced back to vocab 151936. Embeddings keep the short tail. Tile count is 149 rows of 1024, padded vocab 152576. `split_lm_head` defaults off. The 1.0 garbage pad was not committed.

An earlier unequal-tail probe, `vocab_chunk_rows(151936, 4096, 2)`, failed `VAIML-BE-TPGEN 9032` on the 149-chunk Concat. That graph is not the merged one. A later 149-tile chain of byte-identical one-hot tiles with a 640-column zero tail also reproduced 9032 (CPU 153, no ELF). The same shapes with a different row-0 constant per tile returned a finite `[1,1,151936]` tensor, NPU 1, ELF 5328344 bytes, max abs gap 1.0 versus those constants. A real Qwen LM-head weight was not the input of that probe.

### #9 and the revert

Pull request #9 merged as `4c1f480` and was reverted on main as `4cd5288`. GitHub still shows #9 as merged, so it cannot be merged again. The static-window work came back through pull request #12.

### #12 Static window, ReduceSum norms, K-split, gated bf16 bake

Merged as `5c80fd4` from `cursor/static-shapes-npu-18aa`. `npu_bf16_fallback` defaults off. An omitted MatMulNBits zero point is 8. An NPU target does not turn the bake on.

On a Qwen3-8B layer the contract is batch 1, query 1, max sequence 16. Past and present are `[1, 8, 16, 128]`. `seqlens` stays the past window. `total_sequence_length` is 17. `attention_bias` is `[1, 1, 1, 17]`. A constant Slice on axis 2 returns the graph boundary to length 16. The fused `GroupQueryAttention` node stays.

`rewrite_fused_norms` replaces `SimplifiedLayerNormalization` and `SkipSimplifiedLayerNormalization` with a float32 square, `ReduceSum` (keepdims 1) divided by the weight length, epsilon from the node, then the reciprocal multiply. A `ReduceMean` form was re-fused by Vitis into `SinglePassRMSNorm` and stayed on CPU. The ReduceSum form is the one in the bake. CPU gaps versus the fused layer at past 16: present key `6.104e-5`, present value `7.451e-8`, residual `1.192e-7`, down projection `7.749e-7`.

`down_proj` is K=12288, N=4096, bits 4, block 32, packed UINT8 `[4096, 384, 16]`, split into three K=4096 `MatMulNBits`. CPU max abs versus the unsplit op is `2.122e-05` (reference max abs 11.21). The Vitis compile of that layer (file base `e671f01e271af234c16d124d6a33387e`, 77 nodes) still aborted in `superkernel_GemmBf16Int4` because the weight buffer is bfloat16 and the kernel expects int8. No operator stat. No NPU line. No ELF.

A 3-node qkv bisect with the external UINT8 weight failed at `superkernels.cpp:69`. Float16 was CPU 3 after `VAIML-COMPILE 1048` and did not name that kernel. Block size 128 and an explicit zero point still aborted in the same kernel. `fuse_MatMulNBits` is already enabled in `vaip_config.json`.

The bake that did run on Vitis is a different graph: MatMulNBits replaced by bfloat16 MatMul, norms left in float32. Fresh cache key `pr9_bf16dense_fresh`. File base `98f392257665e11144c1346566e89924`, 78 nodes. ELFs 1840696 and 1253096 bytes. Operator stat CPU 3, VAIML 75, NPU 2. `session.run` returned four finite float32 tensors. Kernel `GemmBfp16`. `GroupQueryAttention` and two present Slices stayed on CPU. Feeds were zeros. An earlier access violation (`0xC0000005`) was reuse of a DequantizeLinear compile under the same Algorithm-A id, not this fresh key.

The 36-layer component bake that later ran is described under the decode section. Commit `84c3c3b` gates that bake. The commit itself was not a Vitis run.

## Code this commit adds on top of e2ac77b

### Host position ids

`rewrite_gqa_to_static_window` now connects `GroupQueryAttention` input 9 when `do_rotary` is 1. The input is int64 `position_ids` of shape `[batch, query_len]`. `do_rotary` 0 does not add it. `seqlens_k` stays the window length. An empty position input rotates with the frozen seqlens row.

On CPU, fused `GroupQueryAttention` with `do_rotary` 1, `rotary_interleaved` 0, head size 16, and seqlens fixed at 16 indexes cos/sin row 16 when the position input is empty (max abs `1.19e-7` versus that row, `0.792` versus row 0). Connecting input 9 to int64 `position_ids` `[1,1]` selects rows 0, 1, 8, 16, and 20 within `1e-6`. The half-split formula is `out[:half] = first * cos - second * sin` and `out[half:] = second * cos + first * sin`. Decode feeds `[[absolute_position]]`, and that index is not capped at `max_seq_len`. A second Gather plus rotate-half in front of a node that still has `do_rotary` 1 would double-rotate. ONNX MatMul has no alpha. The Qwen scale is already `0.0883883461356163`.

`test_gqa_position_ids_selects_the_rope_row` covers the rewriter.

### Per-component bake

`apply_static_contract` gains `defer_matmul_bake`. It performs the structural rewrites, leaves packed `MatMulNBits` in place, and keeps the norm body in float32. Setting it together with `npu_bf16_fallback` raises `StaticShapeError`.

`split_onnx` no longer bakes the loaded master. When the caller sets `npu_bf16_fallback`, each saved component is contracted with `defer_matmul_bake=True` and then baked. `only_components` limits which files are written. External initializers that the component still needs are copied into that component's directory. A parent-escaping location (`../model.onnx.data`) is rejected. `load_external_data` defaults true.

The two new tests check that the master graph is not expanded and that a copied external slice reloads after the master directory is removed.

### Empty generation is not a match

`benchmark_onnx_split_vs_whole.py` used to treat a missing monolith string as a quality match, and the report defaulted `quality_match` to true. Both now require a non-empty equal pair. This script was not run against the baked graphs.

### Quark stage, fail-closed

`chunking_utils/onnx/quark_optimize.py` is the pull request #10 module, unchanged. Preset for 8 bits is `INT8_TRANSFORMER_DEFAULT`. Preset for 4 bits is `MATMUL_NBITS`. `optimize_for_npu` is not called unless `quark=True`. A missing install is `QuarkUnavailable`, not a successful split. The package `__init__` exports `QuarkOptimizeConfig`, `QuarkOptimizeResult`, `QuarkUnavailable`, `optimize_for_npu`, `pipeline_order`, and `prepare_model_for_split`.

Quark is still not installed in `ryzen-ai-1.8.0`. There is no calibration set for a distribution-gate rerun. The stage does not compile an XDNA binary.

### Decode harness

`benchmark_npu_decode.py` loads the baked split, runs the packed INT4 monolith on CPU for the comparison rows, releases that session, then times only the split loop. Defaults point at `C:\Users\antmi\qwen3-work\split-qwen-npu-full`. It refuses to start when `C:\Users\antmi\qwen3-work\vitis-pos-batch\RUNNING` exists unless `--allow-overlap` is passed. `--tolerance` is accepted and is not the pass check.

The live gate, all of which must hold:

- top-1 agreement on at least 95 percent of steps
- top-5 overlap at least 0.8 on every step
- KL(softmax of the monolith, softmax of the split) under 0.05 on every step

Max absolute error is recorded and is not the gate. A failing conjunction exits 2.

`benchmark_cpu_fp32_isolation.py` is CPU only. It does not open Vitis. It upcasts the baked bf16 MatMul weights to float32 and compares them with the INT4 monolith. Its default path calls `stamp_position_ids` on the 36 baked layer protos and rewrites that split's `model_info.json`. Weights stay external. A default run changes those protos.

`bf16_npu_gate.md` states the gate result in short form. This file is the long form.

## Baked 36-layer compile

`split_onnx` wrote 40 ONNX files under `C:\Users\antmi\qwen3-work\split-qwen-npu-full\model.onnx` in 240.5 seconds. Flags: static batch 1, query 1, max sequence 16, `npu_bf16_fallback`, `split_lm_head`, `load_external_data` false. The master still has 181 `MatMulNBits`. Each layer file baked 7 of them and is 406,847,488 bytes. The LM head is 149 tiles of 1024, padded vocab 152576, sliced to 151936. Embeddings baked 0 MatMulNBits (2,489,319,424 bytes) and keep the short tail. `other.onnx` has 12 nodes and 0 initializers. A scan found 0 dangling external locations. Master `model.onnx.data` stayed 7,241,596,928 bytes.

After the position-ids stamp, the 36 layer protos wire `GroupQueryAttention` input 9 to int64 `position_ids` `[1, 1]`. Weight-file bytes stayed 406,847,488 per layer. The stamp changes the model hash, so the old `fullbake_layer_XX_bf16` caches must not be reused. The position caches live under `vitis-cache\layer_XX_pos` with key `fullbake_layer_XX_pos`.

The 3-wide compile of layers 0-35 hit a shared-cwd signature race on layers 2, 20, 22, 25, 34, and 35 (exit 3221226505). Solo retries then finished 6 of 6. All 36 `fullbake_layer_XX_pos.rai` files exist. Each retry logged `Actually running on NPU 2`. Feeds of the earlier zero-feed batch of layers 1-35 were zeros: execution of those graphs, not an INT4 numeric match. That batch's parent printed `ALL_DONE ok=34 fail=1` because layer 26's timestamp contained the digits 9032. That layer exited 0.

The layer contract that ran is hidden float32 `[1, 1, 4096]`, KV `[1, 8, 16, 128]`, `position_ids` int64 `[1, 1]`, and `attention_bias` `[1, 1, 1, 17]`. Q is 32 heads of 128. K and V are 8 heads of 128.

## Decode measurements

CPU isolation, report `C:\Users\antmi\qwen3-work\cpu-fp32-isolation-report.json`. Teacher-forced 17 token ids, first id 785. INT4 monolith versus the stamped bake after bf16 MatMul weights were upcast to float32. Top-1 matched all 17 steps. Step 0 max abs `0.27833956480026245`. KL peak `0.0155`. This is the CPU floor. It still includes stored bf16 rounding. It is not an NPU gap.

NPU position-ids harness, report `C:\Users\antmi\qwen3-work\npu-decode-position-report.json`. Exit 2. `quality_pass` is false.

- Top-1 agreed on 15 of 17 steps. Misses: step 0 monolith 271 versus split 1124 (KL `0.02409`), step 9 monolith 264 versus split 220.
- Top-5 overlap was at least 0.8 on every step.
- KL exceeded 0.05 on steps 7, 8, 9, 12, 13, and 16. Peak `0.182684` is step 16, which still agrees on top-1 (token 39088). Step 9's KL is `0.1782`. Step 1's KL is `0.0169` and does not fail the KL check.
- Step 0 max abs `1.205`. Step 16 max abs `2.998`, both sides token 39088.
- `decode_tps` `1.5858`. Session load 21.047 seconds. Split peak RSS 17948.83 MB.

The gate was not widened. `1.72` tokens per second is an earlier split-loop figure, not this gate run. The 17-step gate was not rerun after the INT8 work.

An older aligned run, before position ids, is `npu-decode-aligned-report.json`. Step 0 max abs `1.7266`. Step 16 max abs `17.6949` with the monolith past length 16 and the split window full. That step-16 gap is a different first token from the isolation prompt. It is not the position-ids result.

## INT8 QDQ, one layer, then parked

This path is per-tensor QDQ built in `C:\Users\antmi\qwen3-work\quark-int8-toy`. It is not Quark, and it is not a Vitis compile of the full model. The short evidence file there is `REPORT.md`.

Full graph: `E:\qwen3-int8\qwen3_qdq.onnx` (624280 bytes) plus sibling data 10079621120 bytes. Counts: MatMul 181, QuantizeLinear 362, DequantizeLinear 543, GroupQueryAttention 36, MatMulNBits 0, absolute external locations 0. ORT 1.27 rejects absolute external-data locations. Symmetric int8 weights, scale max-abs/127, from the offset-8 low-nibble-first unpack of the packed INT4 weights. Activation scales are one-token affine uint8.

One CPU compare, calibration token id 151643, empty cache. INT4 logits argmax 33975, max abs `15.012137413024902`. QDQ logits finite, max abs `15.05298137664795`, same argmax. Gap max abs `3.3970890045166016`, mean abs `0.7304399609565735`. That token is the token the activation scales were fit on. It is not a held-out prompt, not the 17-step gate, and not a Vitis run.

Decoder layer 0 was extracted (five QDQ MatMuls plus one fused GroupQueryAttention). Four distinct Vitis frontend failures followed. Each graph edit removed one abort and exposed the next. No full-layer INT8 compile has finished with an NPU subgraph.

1. Fused `GroupQueryAttention` in the mixed layer. XIR cancelled. The last warning named GroupQueryAttention output 0, type Selector, as undefined in XIR. `PY_EXIT` `-1073741819` (`0xC0000005`), twice. Removing that fused op left five projections. Cache key `layer0_qdq_mm_only`, `PY_EXIT` 0, no `superkernel_GemmBf16Int4`. Four subgraphs ran on the NPU (qkv, o_proj, gate, up). `down_proj` at K=12288 stayed on CPU: `qlinear-conv2d` had no legal tiling. Output `DequantizeLinear` stayed on CPU. Vitis versus CPU QDQ gaps were 0.0 on the five outputs. Peak working set 2448.74 MB.

2. GroupQueryAttention replaced by rotate-half primitives. Position is `seqlens_k` (a numpy sweep: position 1 matches the fused op; positions 0 and 2 do not). With fused norms, CPU gaps versus fused GroupQueryAttention were attention `3.725e-9`, present key `9.537e-7`, present value 0, residual 0, down 0. Vitis aborted `PY_EXIT` `-1073740791` (`0xC0000409`) with `tensor<*xf32>` against `tensor<1x1x4096xf32>` on `@main_graph` return 0. That return is `SkipSimplifiedLayerNormalization` output 3. Stamping value infos made `infer_shapes` report 0 unranked tensors, and VAIML's `tensor_shape.json` already listed output 3 as `1 1 4096`. The same abort remained. The 31 empty shape entries were scalar QDQ scales and zero-points. The five-MatMul graph contained those scalars and compiled.

3. `rewrite_fused_norms` replaced the four fused norms with the ReduceSum chain. 4 `ReduceSum` nodes, 0 fused norms, 124 value infos, 0 unranked, no manual stamps. CPU gaps versus the fused extract: attention `1.024e-8`, present key `2.289e-5`, present value 0, residual 0, down 0. The key gap is under the bake's `6.104e-5` for this ReduceSum form. Default X2 then aborted before partitioning. The measured line is signedness on one rank-4 type:

    `tosa.reshape` inferred `tensor<6144x1x1x4096xi8>` against return `tensor<6144x1x1x4096xui8>`

    on the fused qkv group (QuantizeLinear, MatMul, both DequantizeLinear), then `VAIML-FRONTEND 2008 Failed selecting AIE kernels`. `PY_EXIT` `-1073740791`. No `ep_report.json`. The qkv weight initializer is int8 `[4096, 6144]` with an int8 zero-point of 0. The five-MatMul graph compiled that same weight later, through the XIR MLADF reshape, because its activations were graph boundaries. This session dies in VAIML kernel selection, before that pass.

    `opt_level` 0 and `opt_level` 1 repeat the qkv line. The documented default is already 0. A qkv-plus-norms cut with no attention primitives repeats it. An attention-plus-MLP cut with no qkv node fails the same class on o_proj: inferred `tensor<4096x1x1x4096xi8>` against `tensor<4096x1x1x4096xui8>`. The qkv-to-attention edge is not required.

4. `target: X1` on the same norms file, cache key `layer0_qdq_prim_norms_x1`, reached `PY_EXIT` 0 and wrote `ep_report.json`. Backend `AMD_AIE2P_4x4_Overlay`. Device stat: CPU 91, `VITIS_EP_CPU` 26, no NPU entry, DPU subgraphs 0. The 26 `VITIS_EP_CPU` nodes are only `QuantizeLinear` and `DequantizeLinear`. Score MatMul, Softmax, context MatMul, RoPE, ReduceSum, and the five projection MatMuls are on CPU. Output gaps were 0.0 because the run stayed on CPU. The docs allow X1 on Strix when it places better. On this file it did not place a DPU subgraph. `xclbin` was not set.

The 4-rank conv pre-layout of the qkv weight was not run. Both sides of the abort are already rank 4. The mismatch is `i8` versus `ui8`. INT8 layer probes stopped after that report.

`down_proj` CPU fallback and output `DequantizeLinear` on CPU were measured only on the five-MatMul extract. They are unmeasured on a full layer.

## Packed INT4 and issue 407

https://github.com/amd/RyzenAI-SW/issues/407 is open against Ryzen AI 1.8.0 and ONNX Runtime 1.27.0 on XDNA2. `com.microsoft.MatMulNBits` with bits 4 reaches `superkernel_GemmBf16Int4`. Argument 2 is an ADF bfloat16 buffer. The candidate weight parameter is `adf::input_buffer_conf<int8>`. Peano stops at `superkernels.cpp:69` on the 3-node probe and at line 292 on the 77-node layer. The reproducer gist is https://gist.github.com/antmikinka/d07df971f941c8325a33f6e5b2fb371b. There is no AMD-authored workaround in the issue thread.

A public check on 2026-10-15, written up outside this repo, found no Ryzen AI release after 1.8.0 (tag dated 2026-07-23) that mentions the abort. The five commits on `amd/RyzenAI-SW` main after that tag change example and README files. No other issue and no matching pull request turned up. Quark XINT8, A8W8, and A16W8 are a documented INT8 compiler path for CNN-style MatMul and Gemm. They are not a packed-INT4 bypass, and the 1.8 compiler-format notes do not list INT8 as a transformer input. The bf16 MatMul bake is the path that runs around the abort. It is not the intended packed-INT4 architecture. `npu_bf16_fallback` stays default off.

The string `unvectorized` is absent from the 1.8 tree as a session switch. AMD's BF16 config example lists `preferred_data_storage` values `auto`, `vectorized`, and `unvectorized`. That switch was not used on these INT8 graphs. `A16W8` appears as PSD xclbin names, not as a Quark switch that was applied to this graph.

## Layout probe, not merged

Commit `c11039f` on `experiment/xdna-layout-probe` records a seq-1 float32 GEMM in `probe_xdna_layout.py`. For MatMul `[1,1,4096] x [4096,4096]`: wall 38.4 seconds, VAIML 1, NPU 1, ELF 353160 bytes, finite output, max abs `2.114` versus the fp32 reference, kernel `GemmBfp16`. Conv1d stayed CPU-only. BC1S Conv2d used `Conv2DBf16` with a larger ELF and max abs `3.198`. Replacing that GEMM with Conv1d or BC1S Conv2d stops there. QK^T and the KV cache inside GroupQueryAttention were not compiled on that branch.

## Left outside git on purpose

- `original-info-signature.txt` and `original-model-signature.txt` at the repo root. Vitis copies them during compile. They are not source.
- `quantized_info.csv` at the repo root. It is a 569-byte log from a toy `o_proj` quantize on 2026-10-15. It points at `C:\Users\antmi\qwen3-work\quark-int8-toy\o_proj_f32.onnx` and is not a calibration set.
- Every file under `C:\Users\antmi\qwen3-work`, including the INT8 `REPORT.md`, the layer extracts, the Vitis caches, the `.rai` packages, and both weight files.
- The 36 baked layer protos. They are generated graphs, not the rewriter.

## Still unmeasured

- No full-layer INT8 compile with an NPU subgraph.
- `down_proj` K=12288 tiling and output `DequantizeLinear` placement, on a full INT8 layer.
- Whether the 15/17 NPU miss is only bf16 accumulation. The CPU float32 upcast and the NPU `GemmBfp16` kernel are different executors, and the missed token ids were not decoded to text.
- Tokens per second of an INT8 graph. None is claimed.
- A distribution-gate rerun. The 17-step result stands.
- Direct NPU-resident OrtValues on this wheel.
- Quark on the 8B model. The install is absent and there is no calibration set.

