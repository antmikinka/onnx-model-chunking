# bf16 NPU gate limitation

The position-ids decode on the baked graphs failed the distribution gate. Report: `C:\Users\antmi\qwen3-work\npu-decode-position-report.json`.

On the same 17 token ids, CPU MatMul of the upcast bf16 weights was 17/17 top-1 with KL peak 0.0155. The NPU run was 15/17 top-1 with KL peak 0.1827. Misses were step 0 (271 vs 1124) and step 9 (264 vs 220). `decode_tps` was 1.5858. `quality_pass` is false. The gate was not widened.

The NPU MatMul log on the layout probe names `GemmBfp16`. Packed INT4 on this EP still aborts in `superkernel_GemmBf16Int4` (https://github.com/amd/RyzenAI-SW/issues/407). These two runs are not a controlled split between accumulation formats. Quark is not installed, and there is no calibration set.
