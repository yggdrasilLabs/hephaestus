# GPU verification

Nothing in this repository's test suite runs on a GPU. The CUDA path (the
session on the CUDA provider, the key/value cache kept in GPU memory, the
half-precision graphs on GPU kernels) is written and tested on the CPU
provider, where the same code runs with host memory as the "device". This
page is the one command that checks it on a real GPU, and what its output
has to say before the path counts as working.

## What you need

- x86_64 Linux with an NVIDIA GPU and its driver, the CUDA 13.0 runtime
  libraries and cuDNN 9 on the library path (for example inside
  `nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04` with `--gpus all`), and a
  Rust toolchain with the build packages of `Dockerfile.gpu`'s builder
  stage. Cargo resolves the whole workspace, so it needs access to the
  private `scylla-client` git dependency, as the image build does.
- A model directory laid out as the server loads it: `config.json`,
  `tokenizer.json`, the chat template, and `onnx/<graph>` with its external
  data files, for every graph to check (for
  `onnx-community/Qwen3-4B-Instruct-2507-ONNX`: `onnx/model_fp16.onnx` and,
  if the full-precision graph is wanted, `onnx/model.onnx`).
- A prompts file:
  `{"max_new_tokens": 64, "prompts": [{"text": "...", "system": "..."}, ...]}`
  (`system` optional). Use four or more prompts of clearly different
  lengths, the longest near the length production sends.

## The command

```bash
ORT_CUDA_VERSION=13 \
HEPHAESTUS_TEXTGEN_BATCH_MODEL_DIR=/models/Qwen3-4B-Instruct-2507-ONNX \
HEPHAESTUS_TEXTGEN_BATCH_PROMPTS=/models/prompts.json \
HEPHAESTUS_TEXTGEN_GPU_GRAPHS=model_fp16.onnx,model.onnx \
HEPHAESTUS_TEXTGEN_BATCH_WIDTH=4 \
cargo test -p hephaestus-core --release --features cuda --lib \
  gpu_generation_matches_cpu_on_a_real_model -- --ignored --nocapture
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `HEPHAESTUS_TEXTGEN_BATCH_MODEL_DIR` | *(required)* | The model directory |
| `HEPHAESTUS_TEXTGEN_BATCH_PROMPTS` | *(required)* | The prompts file |
| `HEPHAESTUS_TEXTGEN_GPU_GRAPHS` | `model_fp16.onnx,model.onnx` | Graph files to check, one after another |
| `HEPHAESTUS_TEXTGEN_BATCH_WIDTH` | `4` | Batch width of the batched runs |
| `HEPHAESTUS_TEXTGEN_BATCH_LOGIT_STEPS` | `8` | Steps per prompt whose full logits are compared; later steps compare the two best tokens |
| `HEPHAESTUS_TEXTGEN_GPU_SKIP_CPU` | unset | `1` skips the CPU reference run (a 4B model on the CPU takes seconds per token, and the float32 graph needs its 16 GB in host memory) |
| `HEPHAESTUS_TEXTGEN_GPU_PROVIDER` | `cuda` | `cpu` runs the whole procedure on the CPU provider, to try the command out without a GPU |
| `CUDA_*` | — | Not read by the test: it uses GPU 0 with ONNX Runtime's default arena settings |

Not verified here: where `cargo test` leaves
`libonnxruntime_providers_cuda.so`. If the load fails with a message about
that library, add its directory to `LD_LIBRARY_PATH`
(`find target -name 'libonnxruntime_providers_cuda.so'`); the image build
in `Dockerfile.gpu` copies the libraries next to the binary for the same
reason.

## What it does

For every graph:

1. Generates every prompt alone on the CPU provider. This is the reference.
2. Loads the graph on the CUDA provider. Loading fails, and the test with
   it, if the provider cannot be registered, the session cannot be created
   on the GPU, a helper graph cannot be placed entirely on the GPU, or the
   first forward pass returns its cache anywhere but in GPU memory.
3. Generates every prompt alone on the GPU and compares it with the CPU
   run, step by step.
4. Generates every prompt through a batch three times (all together, in
   reverse order, joining three steps apart) and compares each with step 3.
5. Generates every prompt alone on the GPU with the cache in host memory
   (`TEXT_GENERATION_DEVICE_CACHE=off`) and compares it with step 3.
6. Prints tokens per second for the prompt pass and for decoding.

## Reading the output

Per graph there is one line `session on cuda, cache on Some("Cuda:0")`,
then for every comparison one line per prompt:

```
model_fp16.onnx cuda vs cpu: prompt 0 equal over 64 tokens, max |logit difference| 3.1e-2
model_fp16.onnx cuda batch staggered vs solo: prompt 2 near tie at step 41 of 64: reference margin there 1.2e-3, max |logit difference| up to there 9.8e-3
model_fp16.onnx cuda vs cpu: prompt 1 DIFFERS at step 7 of 64: reference margin there 2.4e0, max |logit difference| up to there 3.9e0
```

and speed lines:

```
model_fp16.onnx cuda solo, device cache: prefill 3912 tokens in 1.412 s = 2771 tokens/s; decode 252 tokens in 9.870 s = 25.5 tokens/s
model_fp16.onnx cuda solo, host cache: prefill ... ; decode ...
model_fp16.onnx cuda batch together (widest 4): 256 tokens in 4.2 s = 61.0 tokens/s, prompt passes included
```

(The numbers above are made up to show the format.)

- **equal**: the run picked the tokens of its reference.
- **near tie**: the tokens part at a step where the reference's two best
  logits are no further apart than four times the largest logit difference
  between the two runs, and that difference is at most 1. Rounding decides
  such a step; GPU and CPU kernels, and a padded row against the same
  request alone, round differently. Not a failure, but read the numbers.
- **DIFFERS**: the tokens part and the logits of the two runs are more
  than 1 apart. A wrong computation. The test fails.

## Pass criteria

The path counts as verified on that GPU when, for every graph:

1. the test passes (`clear differences: 0`);
2. the line `session on cuda, cache on Some("Cuda:0")` is printed;
3. `cuda vs cpu` lines are `equal`, or near ties with a maximum logit
   difference of the size half precision explains (around 1e-2 to 1e-1 for
   `model_fp16.onnx`, around 1e-4 or less for `model.onnx`);
4. `host cache vs device cache` lines are all `equal` with a logit
   difference of 0 or very close to it (the same kernels on the same
   values);
5. `batch ... vs solo` lines are `equal` or near ties of the same size as
   in the CPU run of the existing test
   `batched_generation_matches_solo_on_a_real_model`;
6. decoding with the device cache is not slower than with the host cache.
   If it is not clearly faster on prompts of a thousand tokens or more,
   the cache is being copied somewhere it should not be.

While it runs, `nvidia-smi` should show the process holding roughly the
weights plus twice the cache (see "GPU memory" in the README), and GPU
utilisation near 100% during the decode steps.

If loading fails on a helper graph (`cannot run the cache-rows graph on
execution provider 'cuda'`), the CUDA provider has no kernel for one of
`Concat`, `Gather`, `Where` or `Cast` in the cache's float type. The
server then still runs with `TEXT_GENERATION_DEVICE_CACHE=off`, at the
cost of copying the cache to the host and back at every step.
