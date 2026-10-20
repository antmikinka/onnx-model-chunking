# What this repository is

This page is for someone who has never seen the project. The measured numbers are in [WORK_RECORD.md](WORK_RECORD.md). The order in which each wall showed up is in [CHALLENGES.md](CHALLENGES.md).

## The short version

We have a Qwen3-8B language model, stored as an ONNX file, and a laptop whose processor includes an AMD neural processing unit (NPU). The repository is the software that cuts that model into pieces and tries to run those pieces on the NPU.

The NPU path that runs today is a workaround. The model’s original 4-bit weights do not compile. We expand them to a wider number format, compile that, and compare the answers with the original model. On a 17-token check the NPU agreed with the original on 15 tokens. That fails the bar we set. A separate attempt to run the same layer in 8-bit integers never produced a full layer on the NPU.

Rebooting this machine into Ubuntu and cloning AMD’s IRON or MLIR-AIE repositories would not make the current ONNX files pass that bar. Those repositories program the NPU a different way. They do not load this split, and they do not patch the compiler that rejected it. The last section says what each of those projects actually is.

## The machine

One Windows PC, dual-boot with Ubuntu. The processor is a Ryzen AI 9 HX 370. AMD calls that generation Strix. The NPU generation is XDNA2, also written AIE2P. The Windows driver we used is 32.0.203.329.

The software stack on Windows is Ryzen AI 1.8.0 and ONNX Runtime 1.27.0 with the VitisAI execution provider. Sessions are started from the conda environment `ryzen-ai-1.8.0` after `conda activate`. Starting that environment’s Python by its full path leaves the Ryzen AI variables unset, and the NPU provider does not come up the same way.

The NPU is a grid of small vector cores. It is not the main CPU, and it is not the GPU. A model reaches it only if a compiler turns the model’s operations into a program those cores can run. On this Windows stack that compiler is inside the VitisAI execution provider. Its logs call the front half VAIML and the code generator Peano.

## The model, in pieces

Qwen3-8B is a 36-layer transformer. The file we started from was exported by onnxruntime-genai. It is one graph plus a 7,241,596,928-byte weight file named `model.onnx.data`. That weight file is not in git. It lives under `C:\Users\antmi\qwen3-work`, which is not this repository. Nothing in this project rewrites those bytes.

The weights for the big matrix multiplies are stored packed: 4-bit integers, groups of 32, in the ONNX operator `MatMulNBits` from the `com.microsoft` domain. There are 181 of those operators. The attention operator is the fused `GroupQueryAttention`. The graph’s sequence length is symbolic, meaning “whatever length you pass at runtime.”

This repository’s job is to cut that graph into about 40 ONNX files: an embedding table, 36 decoder layers, a final norm, a vocabulary projection (the LM head), and a small leftover graph. Cutting the file does not change the math. It also does not, by itself, make the NPU accept the file.

## Words that show up everywhere

- **ONNX.** A file format for a neural network: operations, shapes, and pointers at weights.
- **Execution provider.** The ONNX Runtime plugin that runs a graph. We use VitisAI for the NPU and CPU for everything it refuses. Naming VitisAI first does not mean an operation landed on the NPU. The log has to say so, and `session.run` has to return a finite tensor.
- **NPU compile.** The provider turns the graph into a cache (a `.rai` package and an ELF for the cores) the first time a session opens. Later runs reuse the cache key. Reusing the wrong cache has crashed the process.
- **Decode.** Generating one token at a time. Our compiled Qwen graphs have query length 1. A prompt of many tokens at once (prefill) was not built for this model.
- **The gate.** A 17-token comparison against the original packed model. All three must hold: the top token agrees on at least 95 percent of steps, the top five tokens overlap by at least 0.8 on every step, and the KL divergence of the probability distributions stays under 0.05 on every step. The largest absolute difference is written down and is not the pass test.

## What the code in git actually does

`MASTER_POC` is the working tree. The pieces that matter:

- `chunking_utils/onnx/splitter.py` cuts a model into component files. If asked, it bakes each saved component. It does not bake the 7 GB master.
- `chunking_utils/onnx/static_shapes.py` freezes a Qwen layer to batch 1, one new token, and a past window of 16. It rewrites the fused norms into plain arithmetic, splits the widest matrix on its input side, and can replace packed 4-bit multiplies with plain multiplies in bfloat16. That replacement is off unless the caller asks.
- `chunking_utils/onnx/kv_concat.py` and the pipeline KV code publish the key/value cache by a slice, a concatenate, and a pointer swap on ordinary CPU memory.
- `chunking_utils/onnx/quark_optimize.py` is an optional AMD Quark step before the split. It is off by default. If Quark is not installed, turning it on fails the run instead of pretending success. Quark is not installed in the 1.8 environment, and the 8B model was not quantized with it.
- `benchmark_npu_decode.py` runs the 17-token gate. `benchmark_cpu_fp32_isolation.py` runs the same tokens on CPU after widening the baked weights back to float32. Its default path also writes a position input into the baked layer files.

Pull requests #1 through #8 and #11 and #12 are merged. #9 was merged and then reverted. #12 put that work back, with the bfloat16 bake gated off. #10 is still open and is only the Quark stage, on an older base. #13 is the branch that holds this explanation.

## Why the original file does not run on the NPU

Five separate facts, each measured.

**The NPU will not hold our runtime buffers.** Asking ONNX Runtime 1.27 to place a tensor on `vitisai`, `ryzenai`, or `npu` fails. The buffers stay on the CPU. The provider copies them during the run. The key/value cache is updated by swapping two CPU pointers after the run returns. An early test of that swap used a graph of two Identity operations. It produced a small ELF and the swapped bytes matched. That showed the swap works. It did not put the cache on the NPU.

**The exported graph uses operations and shapes this compiler rejects.** Scatter updates, `Where` masks, a symbolic sequence length, and IR version 14 were each a dead end. The graph we compile uses a fixed window, an additive mask of `-10000`, a slice-and-concatenate cache, IR version 10, and a host `position_ids` tensor when rotary embeddings are on. For one new token and 16 past tokens, the attention total length is 17 and the published cache is sliced back to 16. An empty position input rotates every token as if it were at position 16. The host now passes the real position.

**Packed 4-bit multiplies do not compile.** The compiler builds a kernel named `superkernel_GemmBf16Int4`. It calls that kernel with a bfloat16 weight buffer. The kernel expects an int8 buffer. Compilation stops. This is [amd/RyzenAI-SW issue 407](https://github.com/amd/RyzenAI-SW/issues/407). The fuse pass for `MatMulNBits` is already enabled. Float16 never reaches that kernel. A public check on 2026-10-15 found no newer Ryzen AI release that mentions the abort. The generated kernel source is written into a cache directory on each compile. Editing that cache does not fix the product compiler.

**The workaround that does compile changes the numbers.** `npu_bf16_fallback` expands each packed multiply into a bfloat16 multiply after the component is saved, and leaves the normalization arithmetic in float32. The flag defaults off. All 36 layers were compiled that way, with position ids. On CPU, widening those baked weights back to float32 and comparing with the original model agreed on all 17 tokens (KL peak 0.0155). On the NPU the same 17 tokens agreed on 15. The misses are step 0 (original token 271, NPU token 1124) and step 9 (264 versus 220). KL peak is 0.182684, on a step that still picked the same top token. Speed on that run was 1.5858 tokens per second. The gate fails. We did not lower the bar, and we did not rerun it.

**8-bit integers on one decoder layer never became a full NPU layer.** We built our own per-tensor 8-bit graph. It is not Quark. On CPU, one calibration token matched the original model’s top token and still differed by max absolute 3.397 and mean absolute 0.730. On the NPU, four of five matrix multiplies in layer 0 ran when attention was removed. The widest one, `down_proj` with input size 12288, stayed on CPU because the compiler found no legal tiling. Putting attention and the norms back in produced a new compiler abort each time we removed the previous one: an unknown attention type, then an unranked norm output, then an int8-versus-uint8 reshape, then a legacy NPU target that compiled and placed every compute operation on the CPU. We stopped. There is no full-layer 8-bit compile with an NPU subgraph.

## What a successful run looks like, and what does not

A line that says an ELF was generated is not success. Success for a placement is `session.run` returning a finite tensor, plus a log line that names the NPU for that subgraph. Success for the model is the gate above. We have the first for the bfloat16 bake. We do not have the second.

A gap of 0.0 between the NPU and the CPU on the 8-bit graph means the NPU did the same arithmetic as that CPU session. It does not mean the 8-bit model matches the original 4-bit model.

## Would Ubuntu, IRON, or MLIR-AIE resolve this?

No. Not for the files this repository compiles, and not by cloning those projects on the Ubuntu side of this dual boot.

The NPU chip is the same hardware either way. The program that rejected our graphs is the ONNX compiler shipped with Ryzen AI. IRON and MLIR-AIE are a different program that can also target the chip.

**What those repositories are.** [MLIR-AIE](https://github.com/Xilinx/mlir-aie) is an open compiler for the NPU’s core grid. [IRON](https://github.com/amd/IRON) is a Python interface on top of it: you write the matrix multiply, the norm, and the attention yourself, and IRON lowers that through MLIR-AIE and the Peano compiler, then XRT loads it. AMD’s own getting-started page says IRON complements the Ryzen AI software platform rather than replacing it. The model example in the IRON tree is Llama 3.2 1B, written as IRON operators. It is not an importer for these Qwen ONNX files. A packed-uint4 kernel proposal (IRON pull request 101) is not on IRON’s main branch, is a different matrix-vector operator, and does not mention issue 407.

**What the Ubuntu install of Ryzen AI 1.8 actually documents.** AMD’s [Linux install page](https://ryzenai.docs.amd.com/en/latest/linux.html) supports Strix, which is this machine, for CNN models in int8 and bfloat16, encoder NLP models in bfloat16, and LLMs in an NPU-only flow. The [LLM page](https://ryzenai.docs.amd.com/en/latest/llm_linux.html) runs a prebuilt AMD model, such as Phi-3.5-mini, downloaded already compiled from Hugging Face, through their `model_benchmark` tool. Hybrid CPU/NPU/GPU flow is not supported on Linux in that release. Building your own model (“model generate”) is not supported on Linux in that release. Their published example speed for Phi-3.5-mini is about 49 tokens per second. That number is their benchmark of their model. It is not a measurement of Qwen3-8B, and it is not this repository.

**What dual boot changes in practice.** Ubuntu does not see the Windows driver, the conda environment, or the `.rai` caches. You install a Linux XRT package, the `amdxdna` driver, and a separate Ryzen AI virtualenv, then reboot to use them. The two operating systems do not run this compile at the same time. The 7 GB weight file can sit on a disk both sides can read. The compiled caches cannot be reused across the two installs.

**Which of our failures that leaves untouched.**

- Issue 407 is inside the ONNX provider’s generated kernel. We have not compiled `MatMulNBits` with the Linux Ryzen AI package, and the public release notes do not say the Linux package contains a different Peano. Until that compile is run, the Linux tarball is not a fix.
- The 15/17 miss already happens on a graph that compiles and runs. Rebooting does not change those baked weights. A different kernel could produce different numbers. That would be a new measurement, not a correction of this one.
- The 8-bit reshape, the unranked norm, and the unknown attention type are VAIML frontend errors. IRON never enters VAIML. Avoiding a compiler by not calling it is a rewrite, not a patch.
- The CPU-only buffers are a limit of onnxruntime-vitisai 1.27 on Windows. IRON moves data through XRT for the kernels you write. That does not relocate the ONNX Runtime tensors in this repository.

**What a real IRON effort would be.** Someone would reimplement Qwen3-8B with IRON operators, or adopt another runtime that already did (FastFlowLM and Lemonade are reported as Linux NPU LLM runners as of March 2026; we have not run them). A third-party ONNX provider posted as [RyzenAI-SW issue 356](https://github.com/amd/RyzenAI-SW/issues/356) claims to execute `MatMul`, `Gemm`, and `MatMulInteger` through IRON on Fedora. We have not run it. It does not claim `MatMulNBits` or `GroupQueryAttention`, so it would not load this split unchanged.

That work can be worth doing. It does not close the gate on the graphs this repository already builds, and it does not rewrite `model.onnx.data`.

## Where the project stands

The bfloat16 36-layer split runs on Windows, end to end, and fails the distribution gate at 15/17 and 1.5858 tokens per second. Packed 4-bit weights still abort in `superkernel_GemmBf16Int4`. One 8-bit decoder layer has no NPU subgraph once attention and the MLP are connected. Quark has not been pointed at the 8B weights. Prefill has not been built. The Ubuntu side of this machine has not been used for any of these runs.
