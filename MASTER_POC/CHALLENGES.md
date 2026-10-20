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
- Provider order is not placement. An ELF line is not a pass unless `session.run` returns a finite tensor. A zero feed is execution, not an INT4 numeric match.
- The string `unvectorized` is not a session switch in the 1.8 tree. AMD's BF16 config example lists it as a `preferred_data_storage` value. It was never applied to these graphs, and it was not going to repair shape inference.
- Replacing the seq-1 4096 GEMM with Conv1d or a BC1S Conv2d was measured on a branch that was not merged (`experiment/xdna-layout-probe`). Conv1d stayed on CPU. The Conv2d ELF was larger, and the max abs gap versus fp32 was worse than `GemmBfp16`. That closed a layout idea for one GEMM. It did not touch QK^T or the KV cache inside GroupQueryAttention.

## What is actually running

The bf16 36-layer split runs end to end on this machine. Embeddings and the final norm stay on CPU. Layers and the LM head are Vitis sessions. KV swap is host float32. The distribution gate fails, 15/17, at 1.5858 tokens per second. Packed INT4 on this execution provider still aborts in `superkernel_GemmBf16Int4`. One INT8 layer has not produced an NPU subgraph once the projections are connected to attention and the MLP.

## Explanations the runs retired

These were reasonable guesses. The runs contradicted them, and they should not be reused as causes.

- "1.72 tokens per second is the gate result." The gate run is 1.5858.
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

## 8. Keeping the Split-Model Architecture: Dynamic KV Cache & NPU-Native Attention

Instead of abandoning the 40-component ONNX split architecture, the codebase extends the split-model contract so that both attention and dynamic KV cache management operate cleanly on the NPU:

- **NPU-Native Decomposed GQA (`decompose_gqa_to_npu_primitives`):** Replaces `com.microsoft::GroupQueryAttention` with 100% NPU-accepted ONNX primitives (`Reshape`, `Gather`, `Slice`, `Concat`, `Mul`, `Sub`, `Add`, `MatMul`, `Softmax`) and zero hostile operators (`Transpose`, `Where`, `Scatter`, `ScatterND`, `ScatterElements`). Head-major permutation is performed via single-head `Slice` + `Reshape` + `Concat`, RoPE rotation uses uncapped int64 `position_ids` with half-split `Gather` + `Slice` + `Mul`/`Sub`/`Add` + `Concat`, and attention scoring uses transpose-free column `MatMul(full_k, q_col)`.
- **Sink-Preserving & Rolling Static KV Window (`static_kv_concat_recipe(layout="sink_window")`, `right_aligned_attention_bias`):** Supports both pure sliding-window and sink-token-preserving (`[0:sink_size]` + `[sink_size+query_len:max_seq_len+query_len]`) static-shape graphs, paired with right-aligned additive `-10000.0` attention bias that unmasks populated sink and recent slots.
- **Dynamic KV Cache Management (`KVCacheManager`):** Tracks continuous `seen_tokens` and uncapped RoPE `position_ids` across unbounded rolling decode steps (`commit_present_swap(..., rolling=True)`), promotes initial tokens into pinned sink slots as the static window fills, and supports dynamic multi-bucket window resizing (`resize_static_window`, `ensure_bucket_capacity`).
- **Static-Window Split Runtime (`SplitModelInferenceEngine` & `benchmark_npu_decode.py`):** Automatically detects static-window split layers, binds active/shadow KV buffers via `KVUpdateMode.CONCAT_SWAP`, streams prompt tokens through `query_len=1` NPU layer graphs, and executes unbounded rolling generation with resident NPU layer sessions.

