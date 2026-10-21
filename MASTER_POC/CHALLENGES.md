# Challenges

A reader who does not know the project should start with [EXPLAINER.md](EXPLAINER.md). This is the account of what blocked the Qwen3-8B split on this Ryzen AI machine, what was tried, and what the measurement actually showed. Figures, commit ids, and log names live in [WORK_RECORD.md](WORK_RECORD.md). A challenge is closed only when a run produced the tensor or the abort it claims.

The goal was one thing: run the existing 40-component Qwen3-8B ONNX split on the Ryzen AI 9 HX 370 NPU (Strix, XDNA2, Ryzen AI 1.8.0, ONNX Runtime 1.27.0) and have decode pass a distribution gate against the packed INT4 monolith. The split stays 40 components. It does not go through AMD's fused Qwen graph, `prefill.pb.bin`, or `com.ryzenai:MatMulNBitsBf`. The weight file `model.onnx.data` (7,241,596,928 bytes) was never rewritten.

The gate, all of which must hold: top-1 agreement on at least 95 percent of steps, top-5 overlap at least 0.8 on every step, and KL under 0.05 on every step. Max absolute error is recorded and is not the gate.

## 1. The NPU will not hold our tensors

Pull request #1 was titled as NPU and iGPU OrtValue allocation. On onnxruntime-vitisai 1.27 the device names `vitisai` and `ryzenai` are rejected, and `npu` plus AMD vendor `0x1002` cannot place an OrtValue. `import xrt` is absent. Host buffers stay on CPU. The execution provider copies them during `session.run`.

What ran instead is a host pointer swap. The past buffer is bound to the active CPU OrtValue, the present buffer is bound to the shadow, the session runs, and the pointers swap only after the run returns. Windows `VirtualLock` page-locks that CPU buffer. That is not HipPinned memory and not an XRT registration. The first graph that exercised the swap was two Identity nodes. It printed a 131416-byte ELF, the swapped host prefix matched, and the OrtValues were still on CPU. That closed "can we publish a buffer after a real session." It did not close "the KV cache lives on the NPU."

## 2. The exported graph is the wrong shape for this compiler

The monolith is an onnxruntime-genai export: IR 10, opset 22 plus `com.microsoft` 1, 181 `MatMulNBits` nodes, bits 4, block 32, symbolic batch and sequence axes, and a fused `GroupQueryAttention`. Splitting that file does not remove Scatter, Where, or Gather, and it does not make the sequence length static.

Each of those showed up as its own wall:

- Scatter updates to the KV cache were rejected. The cache that compiled in the toys is a one-axis Slice plus Concat. A rank-4 begin/end with zeros on the non-sequence axes produced an empty kept past. The one-axis form turned past `[0..7]` plus a new token into the expected window on CPU.
- `Where` / masked-fill does not belong on the NPU path we kept. Pull request #6 lowers it to an additive bias of `-10000`.
- IR 14 failed to load. IR 10 is what ran. The runtime accepts IR at most 13.
- A static window whose total length was 16 made `GroupQueryAttention` crash on CPU: `seqlens_k[0] = 16` is out of range `[0, 16)`. The legal contract is seqlens equal to the past window and total equal to past plus the query. For query length 1 and a past of 16, total is 17, `attention_bias` is `[1,1,1,17]`, and a Slice on axis 2 returns the graph boundary to length 16. Vitis had been warning that axis 2 was 17 against a frozen 16. That warning was the compiler telling the truth.
- Fused rotary with an empty position input indexes the cos/sin row at the frozen seqlens value. On a small CPU check that was row 16, max abs `1.19e-7` against that row and `0.792` against row 0. Wiring input 9 to int64 `position_ids` `[1,1]` selects the host row. Decode feeds `[[absolute_position]]`, uncapped. Prefill, if it is ever built, must feed `[[0, 1, ..., query_len-1]]`, not a scalar. A second Gather in front of a node that still has `do_rotary` 1 double-rotates. ONNX MatMul has no alpha. The Qwen scale is already `0.0883883461356163`.
- Calling `python.exe` by path skips conda activate scripts, so `RYZEN_AI` stays unset and the Vitis provider does not come up as it does in an activated shell. Ryzen AI 1.7 option names are snake_case (`cache_dir`). The 1.8 probes that worked used camelCase `cacheDir` and `cacheKey` as well.

Pull request #7 showed that a toy prefill can be faster on Vitis at prompt 256 and 512 for one layout (8 heads, head 128). `SessionManager` does not read that constant. The Qwen graphs that were compiled are query length 1. Static prompt padding was parked.

## 3. Packed INT4 does not lower

This is the wall under every "just run MatMulNBits on the NPU" attempt.

`com.microsoft.MatMulNBits` with bits 4 reaches `superkernel_GemmBf16Int4`. The generated call passes argument 2 as a bfloat16 buffer. The kernel parameter is `adf::input_buffer_conf<int8>`. Peano stops at `superkernels.cpp:69` on a 3-node probe and at line 292 on the 77-node layer. `fuse_MatMulNBits` is already enabled. Float16 never names that kernel: it dies earlier at `VAIML-COMPILE 1048` and stays on CPU. Block size 128 and an explicit zero-point still abort in the same kernel. The zero-point tensor is not shaped like the scales. For bits 4 and kpack 2 it is `[cols, (k_blocks+1)//2]`.

`superkernels.cpp` is not shipped source. The compiler emits it under the VAIP cache on each run. Editing that cache does not fix the lowering.

The bug is filed as [amd/RyzenAI-SW#407](https://github.com/amd/RyzenAI-SW/issues/407), with the reproducer at [this gist](https://gist.github.com/antmikinka/d07df971f941c8325a33f6e5b2fb371b). A public check on 2026-10-15 found no Ryzen AI release after 1.8.0 (tag 2026-07-23) that mentions the abort, no other issue, and no matching pull request. The issue thread has no AMD-authored workaround.

Quark's documented INT8 schemes (XINT8, A8W8, A16W8) are a different compiler path, aimed at CNN-style MatMul and Gemm. They are not a packed-INT4 bypass. The 1.8 compiler notes do not list INT8 as a transformer input. `import quark` in the 1.8.0 env raises `ModuleNotFoundError`. Nothing was pip-installed there. Pull request #10 is a fail-closed stage that refuses to pretend a missing Quark install succeeded. It has not quantized the 8B model. There is no calibration set for a gate rerun.

The path that does run is `npu_bf16_fallback`: bake each `MatMulNBits` into a bfloat16 `MatMul` after the component is saved, leave the norm body in float32, and let Vitis pick `GemmBfp16`. The flag defaults off. Targeting the NPU does not turn it on. Baking the unsplit master was rejected because that graph is about 7 GB of external data and the bake was estimated near 16 GB. The bake runs per saved component, then that component is released. A fresh cache key matters. Reusing Algorithm-A `f67197bfc54d8b3cb60f81692dae5e62` from a DequantizeLinear compile access-violated (`0xC0000005`) on the baked graph. Cache key `pr9_bf16dense_fresh` compiled and `session.run` returned four finite tensors. Feeds were zeros. `GroupQueryAttention` and two present slices stayed on CPU.

That bake is a route around issue 407. It is not the intended packed-INT4 architecture.

## 4. The bake runs, and the gate still fails

All 36 layers were compiled with position ids. Six layers died in a 3-wide batch with exit `3221226505` because the workers shared a cwd and raced `original-model-signature.txt`. Solo retries finished 6 of 6. Every `fullbake_layer_XX_pos.rai` exists. Each retry logged two subgraphs actually running on the NPU. An earlier batch looked failed because a log timestamp contained the digits 9032. That layer had exited 0. Compiler error `VAIML-BE-TPGEN 9032` is a different string.

CPU isolation upcast the baked MatMul weights to float32 and compared them with the INT4 monolith on 17 teacher-forced tokens, first id 785. Top-1 matched all 17. KL peak was 0.0155. Step 0 max abs was 0.278. That is the CPU floor, including stored bf16 rounding. It is not an NPU gap.

The NPU run of the same 17 ids failed the gate. Top-1 agreed on 15 of 17. Misses were step 0 (monolith 271, split 1124, KL 0.024) and step 9 (monolith 264, split 220, KL 0.178). KL was over 0.05 on steps 7, 8, 9, 12, 13, and 16. The peak, 0.182684, is step 16, and that step still agrees on top-1 (token 39088). Top-5 overlap stayed at or above 0.8. `decode_tps` was 1.5858. Split peak RSS was 17948.83 MB. `quality_pass` is false. The gate was not widened and was not rerun.

An earlier run, before position ids, had step 16 max abs 17.69 with a different first token. Filling the attention window did not shrink that gap. The position-ids result replaced the "frozen seqlens row explains the whole miss" theory for the RoPE index. It did not make the distribution gate pass. The missed token ids were not decoded to text. The CPU upcast and the NPU `GemmBfp16` kernel are different executors, so the 15/17 miss is not proven to be bf16 accumulation alone.

`1.72` tokens per second is an earlier split-loop timing. It is not this gate run.

## 5. Each fused op had its own compiler wall

These were easy to collapse into "GQA is the problem" or "the norms are the problem." They are separate.

**Norms.** `SimplifiedLayerNormalization` has no ONNX shape function, so the first unranked producer in a decomposed layer was the input layernorm even when its inputs were already `[1,1,4096]`. `SkipSimplifiedLayerNormalization` output 3 is the residual before normalization. Vitis aborted with `tensor<*xf32>` against `tensor<1x1x4096xf32>` on that return. Stamping value infos made `infer_shapes` report zero unranked tensors, and VAIML's own `tensor_shape.json` already listed the shape. The abort remained. The shape stamp was not the defect.

A `ReduceMean` RMS chain was re-fused into `SinglePassRMSNorm` and failed L1 tiling, staying on CPU. The `ReduceSum` divided by the weight length is the chain in the bake. CPU gaps versus the fused layer at past 16 were present key `6.104e-5`, present value `7.451e-8`, residual `1.192e-7`, down `7.749e-7`. An independent split of the 4096-wide norm into 2048-wide pieces blew the present-key gap out to `3.039e-1` and was removed.

**Down projection.** Live `down_proj` is K=12288, N=4096, packed UINT8 `[4096, 384, 16]`. A memtile rejection showed up on the wide weight. Splitting K into three 4096-wide `MatMulNBits` matches float32 ORT within `2.122e-05` (reference max abs 11.21). The Vitis compile of that layer still died in `superkernel_GemmBf16Int4`. The K-split removed the memtile line. It did not place the op on the NPU. On the INT8 five-MatMul extract the same K=12288 projection stayed on CPU because `qlinear-conv2d` found no legal tiling.

**LM head.** An unequal tail, `vocab_chunk_rows(151936, 4096, 2)`, failed `VAIML-BE-TPGEN 9032` on a 149-chunk Concat. The merged pad (pull request #8) makes every tile one width and slices back to vocab 151936. A later probe reproduced 9032 with 149 byte-identical one-hot tiles plus a zero tail, and avoided 9032 when each tile had a distinct row-0 constant. Those probes were one nonzero row, not a dense Qwen table. The max abs gap versus those constants was 1.0. A real vocabulary weight was not the Vitis input. The baked LM head that ran in the decode harness is the padded 149-tile graph, and it did return logits. That does not make the 9032 probe a numeric match to the INT4 LM head.

**GroupQueryAttention on the INT8 graph.** With five QDQ MatMuls and one fused GQA, XIR reported the GQA output type Selector as undefined and the process access-violated, twice (`PY_EXIT` `-1073741819`). Removing GQA let four of the five projections run on the NPU (qkv, o_proj, gate, up) with Vitis-versus-CPU QDQ gaps of 0.0. That 0.0 is lowering fidelity against CPU QDQ. It is not agreement with the INT4 monolith. The full-model CPU compare on the calibration token (id 151643, empty cache) matched argmax 33975 and left a max abs gap of 3.397 and a mean abs gap of 0.730.

## 6. INT8 on one layer was four frontend failures, then a stop

Decomposing GQA on layer 0 matched the fused CPU op closely (attention `3.725e-9`, present key `9.537e-7`). Vitis then aborted on the unranked norm return described above (`PY_EXIT` `-1073740791`). Replacing the four fused norms with the ReduceSum chain removed that abort. CPU gaps stayed at or under `2.289e-5` on the present key. The next abort was a different stage: `tosa.reshape` inferred `tensor<6144x1x1x4096xi8>` against return `tensor<6144x1x1x4096xui8>` on the fused qkv group, then `VAIML-FRONTEND 2008`. Both types are rank 4. The mismatch is signedness. The weight initializer is int8 with zero-point 0. The five-MatMul graph had compiled that same weight later, through an XIR MLADF reshape, because its activations were graph boundaries. A graph that keeps a producer or a consumer of the QDQ MatMul dies in VAIML kernel selection before partitioning, which is why those runs have no `ep_report.json`.

Three probes followed, cheapest first:

- `target: X1` compiled. The backend was `AMD_AIE2P_4x4_Overlay`. The partition report says CPU 91, `VITIS_EP_CPU` 26, DPU subgraphs 0. The 26 nodes are only QuantizeLinear and DequantizeLinear. Every projection, the score MatMul, Softmax, RoPE, and ReduceSum landed on CPU. Output gaps of 0.0 are CPU versus CPU.
- `opt_level` 0 and `opt_level` 1 repeated the qkv signedness abort. The documented default is already 0.
- A qkv-plus-norms cut with no attention primitives repeated the qkv abort. An attention-plus-MLP cut with no qkv node failed the same way on o_proj (`4096x1x1x4096` i8 against ui8). The qkv-to-attention edge is not the trigger.

A 4-rank conv pre-layout of the weight was not run. The measured line is signedness, not rank. INT8 layer work stopped there. No full-layer INT8 compile has finished with an NPU subgraph. `down_proj` tiling and output `DequantizeLinear` on CPU are measured only on the five-MatMul extract.

The pattern is the evidence. Each graph edit removed one VAIML abort and exposed the next. That is a toolchain limit on this INT8 layer, not a sequence of mistakes in the tensor ranks.

## 7. The toolchain hides the failure

Several "successes" were wrapper or log artifacts.

- A batch file that ends with `echo PY_EXIT:%ERRORLEVEL%` returns 0 to the caller, because `echo` clears `ERRORLEVEL`. Python had aborted with `-1073740791`. The runners now save `RC` before the echo and `exit /b %RC%`.
- A missing `aie_unsupported_original_ops.json` prints after the fatal MLIR error. Creating that file by hand was rejected. The compile dies before the report is written.
- `enable_cache_file_io_in_mem` set to 0 creates the cache directory and still does not produce `ep_report.json` if kernel selection aborts first.
- An XRT `0xc00002b6` ("The device has been removed") context reset while opening an existing `.rai` cache prints a generic "recompile with the latest VAIML" line even when the cache is valid. On the first layers 18-26 bisect attempt, opening `fullbake_layer_19_pos` died in about three seconds with `0xc00002b6` after that same cache had already loaded in the 11-35 run; the retry on the unchanged cache passed.
- Provider order is not placement. An ELF line is not a pass unless `session.run` returns a finite tensor. A zero feed is execution, not an INT4 numeric match.
- The string `unvectorized` is not a session switch in the 1.8 tree. AMD's BF16 config example lists it as a `preferred_data_storage` value. It was never applied to these graphs, and it was not going to repair shape inference.
- Replacing the seq-1 4096 GEMM with Conv1d or a BC1S Conv2d was measured on a branch that was not merged (`experiment/xdna-layout-probe`). Conv1d stayed on CPU. The Conv2d ELF was larger, and the max abs gap versus fp32 was worse than `GemmBfp16`. That closed a layout idea for one GEMM. It did not touch QK^T or the KV cache inside GroupQueryAttention.

## What is actually running

The bf16 36-layer split runs end to end on this machine. Embeddings and the final norm stay on CPU. Layers and the LM head are Vitis sessions. KV swap is host float32. The distribution gate fails, 15/17, at 1.5858 tokens per second. Packed INT4 on this execution provider still aborts in `superkernel_GemmBf16Int4`. One INT8 layer has not produced an NPU subgraph once the projections are connected to attention and the MLP.

## Explanations the runs retired

These were reasonable guesses. The runs contradicted them, and they should not be reused as causes.

- "1.72 tokens per second is the gate result." The gate run is 1.5858.
- "The NPU LM head causes the 0.18 KL peak." LM head only on the NPU passes (17/17 top-1, KL peak 0.016953 at step 16). All 36 layers on the NPU with a float32 LM head still fails with KL peak 0.186561 at step 16 and 15/17 top-1.
- "The 0.18 KL peak requires all 36 NPU layers." Layer 2 alone on the NPU (with the other 35 layers in float32 and `cpu_lm_head` false, so `lm_head` stayed on Vitis) fails at step 16 with KL 0.222947 (17/17 top-1, top-5 overlap 0.8, max abs 2.380). Layer 3 alone on the NPU passes (17/17 top-1, KL peak 0.010874 at step 16). Layers 4-17 on the NPU (with layer 2 and the remaining layers in float32) pass (17/17 top-1, step-16 KL 0.042343). Layers 0-1 on the NPU keep every KL under 0.05 (peak 0.0317 at step 16, one top-1 miss at step 13). Layers 18-35 on the NPU keep every KL under 0.05 (peak 0.0242 at step 13, one top-1 miss at step 0).
- "Making Layer 2 match float32 suffices to pass the gate." Putting Layer 2 in CPU float32 while keeping the other 35 layers and the LM head on the NPU caches (`bisect_cpu_only_layer_2.json`, 1.6417 tok/s, 19195.07 MB RSS) drops step-16 KL from 0.182684 to 0.066903 and restores step-9 top-1 (264), and it still fails `quality_pass`: step 0 misses top-1 (271 vs 1124, KL 0.040559), and steps 7 (0.073764), 9 (0.054854), 13 (0.079918), and 16 (0.066903) stay over 0.05. A float32 Layer 2 is the ceiling for a Layer-2-only numeric fix.
- "Layer 2's 0.222947 step-16 logit KL is a large step-16 output spike inside Layer 2, or `--dump-layer-gap 2` names one of the seven MatMuls." Side-by-side comparison of NPU Layer 2 against its CPU float32 upcast on the same inputs (`bisect_npu_layer_2_gap.json`, 0.4533 tok/s, 26590.8 MB RSS) shows `hidden_mean_abs` stays under 0.001 on every step, and Layer 2's step-16 `hidden_max_abs` is 0.01157 (`hidden_mean_abs` 0.000750), while the largest `hidden_max_abs` is at step 0 (0.06615, logit KL 0.002968). The 0.222947 step-16 logit KL is downstream amplification by the 33 float32 layers (`3-35`) of small Layer-2 state deltas (including `hidden_argmax_channel` 1838 on steps 7-11 and 13-16, and new-key-slot max abs 0.07838 vs new-value-slot 0.00516 at step 16, with key peak 0.13694 at step 1 and value peak 0.00742 at step 3). `layer_gap_rows` records whole-layer outputs (no cosine, no per-MatMul tensor), so it cannot separate the K half of fused `qkv` from `k_norm` and RoPE, and splitting `K=4096` or rewriting projections as `1x1 Conv2d` does not follow from it.
- "Power-of-two block-8 equilibration fixes Layer 2's step-16 KL, or the 15x-55x K-versus-V absolute gap measures a 15x-55x precision loss inside `k_norm`." Compiling Layer 2 with `npu_bf16_equilibrate=True` (`layer2-eq\layer_2_of_36.onnx`, 83 nodes, cache `vitis-cache\layer_02_eq8\`, report `bisect_npu_layer_2_eq.json`) scaled six MatMuls (`qkv`, `gate`, `up`, and the three `down` chunks at `K = 0, 4096, 8192`, changing `158 / 4096` channels on `input_layernorm` with factors `0.5, 1, 2, 4, 8` and `84 / 4096` on `post_attention_layernorm` with factors `0.5, 1, 2`, while leaving `o_proj`, `q_norm`, and `k_norm` at `0 / 128` changed). Its CPU float32 upcast matched the unequilibrated bf16 upcast at 0.0 max abs across present K, present V, and both hidden outputs, and on the NPU step-16 logit KL stayed at 0.225264 (versus 0.222947), step-16 `hidden_max_abs` moved from 0.011567 to 0.010354, and the step-16 new K slot (`0.07837677001953125`), step-16 new V slot (`0.005165`), and step-1 new K slot (`0.136944`) did not move at all (while step 0's new K slot moved from 0.07402 to 0.08926 and step 3's from 0.11154 to 0.06342). The VAIML log instantiates `GemmBfp16` with `lKernelArgs="bfloat16,0,1,1"` (which does not state a block size; block-8 was an assumption inside `_power_of_two_block_scale`), every `Cast_inv` in both the 2026-10-14 and new 83-node layer already has `to=1` (`FLOAT`), and VAIML places `q_norm` and `k_norm` `Mul_inv` on the AIE. On a CPU float32 draw of Layer 2 with both norms in float32, present K peaks at 7.785 and present V at 0.310, with absolute errors versus packed INT4 of 0.0591 and 0.00170 (~35x absolute ratio) and relative errors of about 0.76% and 0.55%; the absolute K-versus-V ratio in `bisect_npu_layer_2_gap.json` reflects tensor magnitude after `RoPE(k_norm(K))`, not a 15x-55x precision loss in `k_norm`.
- "Floating layers 0-2 clears the remaining KL failures." Putting layers 0-2 in CPU float32 while keeping layers 3-35 and the LM head on the NPU caches (`bisect_cpu_layers_0_2.json`, 1.5909 tok/s, 21314.70 MB RSS) clears the KL failures at steps 7 (0.024271), 9 (0.020384), and 16 (0.019566), and it still fails `quality_pass`: step 13 stays at KL 0.080061 (versus 0.079918 when only layer 2 is in float32), step 4 fails top-5 overlap (0.6 at KL 0.032180), and step 0 still misses top-1 (271 vs 1124 at KL 0.006545, 16/17 top-1). Floating layers 0-10 (`bisect_cpu_layers_0_10.json`, layers 11-35 and LM head on the NPU) removes both the step-13 KL of 0.080061 (down to 0.020533) and the step-4 top-5 overlap of 0.6 (back to 0.8, with step-16 KL 0.039322), leaving only the step-0 top-1 flip (271 vs 1124 at KL 0.019039, 16/17 top-1). Pairing layers 4-10 with 18-35 and the LM head on the NPU (`bisect_npu_layers_4_10_18_35.json`, layers 0-3 and 11-17 in float32) pushes step 13 over 0.05 to KL 0.066192 (still agreeing on top-1 token 3209, so most of the 0.080061 spike in 3-35 and not all of it), raises step 16 from 0.052387 to 0.074938, misses step 0 (271 vs 1124 at KL 0.013332), and keeps step-4 top-5 overlap at 0.8 (so that pair is not the 0.6 overlap failure).
- "Layers 4-17 are a flat 0.042 spread across 14 layers, or a single bad layer inside 4-10." Splitting 4-17 shows layers 4-10 on the NPU (`bisect_npu_layers_4_10.json`) fail at step 16 with KL 0.052387 (17/17 top-1, every other step under 0.05), while layers 11-17 on the NPU (`bisect_npu_layers_11_17.json`) pass with step-16 KL 0.016944 (17/17 top-1). Splitting 4-10 further shows neither half fails alone: layers 4-6 (`bisect_npu_layers_4_6.json`) pass with step-16 KL 0.036817 (17/17 top-1) and layers 7-10 (`bisect_npu_layers_7_10.json`) pass with step-16 KL 0.016523 (17/17 top-1, the same floor as the LM head alone). The 0.052387 failure appears when 4-6 and 7-10 run together, is pulled down to 0.042343 when 11-17 also run on the NPU, and is pushed up to 0.074938 when 18-35 instead of 11-17 run with 4-10 on the NPU.
- "The step-0 top-1 flip (271 vs 1124) is spread across 18-35." Splitting 18-35 shows layers 18-26 and the LM head on the NPU (`bisect_npu_layers_18_26.json`) pass the gate (17/17 top-1, step 0 agrees on 271 at KL 0.001890, step 13 KL 0.009760), and layers 11-26 and the LM head together (`bisect_npu_layers_11_26.json`, 1.0700 tok/s, 39373.2 MB RSS) also pass as one 16-layer run (17/17 top-1, step 0 agrees on 271, step 13 KL 0.011495, step 16 KL 0.031163). Inside 27-35 (`bisect_npu_layers_27_35.json`, step 0 miss at KL 0.011949), layers 27-30 and the LM head (`bisect_npu_layers_27_30.json`) pass (17/17 top-1, step 0 agrees on 271 at KL 0.003420, step 16 KL 0.014622), while layers 31-35 and the LM head (`bisect_npu_layers_31_35.json`, 51016.6 MB RSS) fail only at step 0 (271 vs 1124 at KL 0.012477, 16/17 top-1, step 13 KL 0.008885, step 16 KL 0.011956).
- "The step-16 gap is only the frozen seqlens row." Position ids were wired. The gate still fails. The older 17.69 gap was a different first token.
- "bf16 accumulation is the sole cause of the 15/17 miss." The CPU floor is 17/17 at KL 0.0155. The NPU executor is `GemmBfp16`. Those are different runs, and the missed ids were not decoded to text.
- "A Vitis-versus-CPU QDQ gap of 0.0 means the quantization matches INT4." That 0.0 is lowering fidelity. The calibration-token gap versus INT4 is max abs 3.397 and mean abs 0.730.
- "GQA is the only INT8 problem." `down_proj` K=12288 stayed on CPU on the projections-only graph, and the signedness abort hits o_proj with GQA and qkv both removed.
- "The bf16 bake is the intended INT4 path." It is the path that avoids issue 407. The flag defaults off.
- "Smaller weights are faster, and 180 CPU-NPU round trips dominate tokens per second." Neither was measured on this graph.
- "An argmax match on one vector means generation will pass the gate." The calibration token matched argmax 33975. That is one vector.
- "Static shapes are NPU placement." The static window, the ReduceSum norms, and the K-split all have CPU agreement and still meet issue 407 or a CPU-only partition when the weights stay packed or the overlay rejects the op.
- "Pull requests #5 through #9 and #12 are proven on the full model on silicon." #9 was merged and reverted. #12 restored the contract. The silicon evidence is the bake and the failed gate, plus toy graphs that are not Qwen layers. #5's Identity ELF is not attention.

## Still open

- Issue 407. No local edit of the generated superkernel fixes it.
- The 15/17 gate. It was not rerun and was not relaxed.
- A full-layer INT8 compile with an NPU subgraph.
- `down_proj` tiling and output `DequantizeLinear` placement on a connected INT8 layer.
- Quark on the 8B weights. The 1.8 env does not import it, and there is no calibration set.
- NPU-resident OrtValues on this wheel.
- Prefill. The compiled Qwen graphs are query length 1.

