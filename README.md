# Nemotron-3.5-Super-VL abliterated on one DGX Spark (TensorFold or vLLM)

An abliterated (uncensored) build of NVIDIA's
[Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
served on **one** NVIDIA DGX Spark. Refusals fall from 98 % to 2 %; tasks, math, code and speed stay where the
original is (on vLLM, NVIDIA's separate drafter accepts a few percent fewer of its guesses; see below).

The checkpoint,
[`armin1/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated`](https://huggingface.co/armin1/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated),
keeps NVIDIA's tensor formats, so both engines load it exactly as published: no conversion step, no patches. Pick one:

- **TensorFold 1.1.2** (default): the official release, a 1,048,576-token window, two requests at once, drafts from
  the checkpoint's own MTP head.
- **vLLM** with NVIDIA's LiLiCorr drafter: NVIDIA's own recipe, one request at a time, and JPEG and video input too.

Either way: an OpenAI-compatible API with text, images and tool calls.

## Numbers

Measured on one DGX Spark (GB10, 128 GB), greedy, thinking off, one request at a time unless noted. *Original* is
NVIDIA's unmodified checkpoint, served the same way on the same engine. Tools: TensorFold's `tools/bench_openai.py`
and `tools/bench_concurrent.py`, a 130-item task suite, 25 executed code tasks, GSM8K (300), a draft-acceptance probe
(nine fixed prompts: 3 code, 3 explanations, 2 prose, 1 German; 384 tokens each) and a prompt-length sweep with a
fresh prompt per size. Raw results: [`results/`](results/).

### TensorFold 1.1.2

Both measured 2026-10-23.

| | Original | This model |
|---|---:|---:|
| Task suite: math / JSON / instructions / tools / agent | 49 / 15 / 8 / 20 / 10 | 49 / 15 / 8 / 20 / 10 |
| Code, executed (25) | 25 | 25 |
| GSM8K (300) | 293 | 292 |
| Decode, code (`fibonacci-raw`) | 51.1 tok/s | 56.2 tok/s |
| Decode, chat (`gpu-chat-no-think`) | 39.2 tok/s | 38.8 tok/s |
| Two requests at once, code / chat (total) | 72.8 / 62.8 tok/s | 81.9 / 63.0 tok/s |
| MTP tokens per round: code / explain / prose / German | 4.74 / 3.33 / 2.63 / 3.66 | 4.74 / 3.29 / 2.78 / 3.46 |
| Lowest free memory during the runs | 28.2 GB | 28.0 GB |

The higher code figure is the text this model happened to write on that one prompt, where more drafts landed; over
the nine-prompt probe the acceptance is the same. Read the speed as equal. The probe, repeated through this
repository's own `prepare.sh` and `start.sh` from a clean copy, gave the same tokens per round (51.1 / 41.8 / 39.4 /
44.3 tok/s); free memory then bottomed at 19.9 GB.

### vLLM + LiLiCorr

Same image (by digest), flags and tools on both dates: the original 2026-10-17/18, this model 2026-10-23.

| | Original | This model |
|---|---:|---:|
| LiLiCorr tokens per round: code / explain / prose / German | 5.20 / 4.01 / 2.92 / 3.37 | 5.05 / 3.78 / 2.81 / 3.30 |
| Decode on the same prompts: code / explain / prose / German | 48.8 / 39.0 / 27.9 / 33.5 tok/s | 46.8 / 36.2 / 27.0 / 32.3 tok/s |
| Decode, code (`fibonacci-raw`) / chat (`gpu-chat-no-think`) | 51.2 / 39.3 tok/s | 60.0 / 34.6 tok/s |
| Prompt / generation, 4K tokens | 2,224 / 45.0 tok/s | 2,276 / 45.8 tok/s |
| Prompt / generation, 64K tokens | 2,875 / 27.4 tok/s | 2,922 / 27.3 tok/s |
| Prompt / generation, 259K tokens | 1,940 / 26.8 tok/s | 1,968 / 26.1 tok/s |
| Prompt / generation, 524K tokens | 1,349 / 21.3 tok/s | 1,347 / 23.9 tok/s |
| Lowest free memory during the runs | 17.6 GB | 16.2 GB |

Prompt processing is the same. NVIDIA's LiLiCorr drafter was trained on the original model; on this one it accepts
fewer of its guesses on code (-3 %) and explanations (-6 %), and a second run through this repository's own scripts
from a clean copy gave the same (5.03 / 3.72 / 2.98 / 3.44 tokens per round). Prose and German moved as much between
the two runs as away from the original: read those as equal. Decode on the probe is up to 7 % slower; single prompts
swing both ways (code +17 %, chat -12 % above) with the text each model writes. TensorFold's MTP drafter shows no such
loss.

### Faithfulness to the original

Teacher-forced on vLLM: both checkpoints score the same 64,319 positions of code, prose, chat, long-context and
reasoning text. KL is taken over each position's top 20 tokens, so it is a lower bound.

| | KL vs original | Top-1 agreement | ΔNLL |
|---|---:|---:|---:|
| The original against a second run of itself (noise floor) | 0.0138 | — | — |
| **This model** | **0.0157** | **95.74 %** | **+0.0013** |

The output distribution sits just past the measurement's own noise floor.

### Refusals

| | TensorFold, original | TensorFold, this model | vLLM, original | vLLM, this model |
|---|---:|---:|---:|---:|
| Harmful and jailbreak prompts refused (of 204) | 200 (98.0 %) | **4 (2.0 %)** | 201 (98.5 %) | **4 (2.0 %)** |
| Ordinary prompts answered (of 100) | 69 | **99** | 66 | **100** |

Counted by a keyword rule over the final answer, thinking off: a calibration aid, not a safety evaluation. The second
row checks that the model does not start refusing ordinary requests; it answers more of them than the original.

## Requirements

- An NVIDIA DGX Spark (GB10, 128 GB unified memory). Other machines are untested; the pinned TensorFold build and vLLM
  image are aarch64.
- Docker with the NVIDIA Container Toolkit (DGX OS has both).
- About 75 GB of free disk for the checkpoint (vLLM: about 26 GB more for its image and the drafter), and about
  100 GB of free memory while it serves: stop other models first.

## Quick start

```sh
git clone https://github.com/Arminova/Nemotron-3.5-Super-VL-Abliterated-Single-DGX-Spark
cd Nemotron-3.5-Super-VL-Abliterated-Single-DGX-Spark
./prepare.sh          # the checkpoint (~70 GiB, resumable) and TensorFold 1.1.2 (SHA-256 checked)
./start.sh            # 3-5 min until ready (up to 10 the first time); http://127.0.0.1:8888/v1
tools/check.sh        # optional: a text, a tool-call and an image request
./stop.sh
```

vLLM instead (or put `ENGINE=vllm` in a `.env` file):

```sh
ENGINE=vllm ./prepare.sh    # adds the vLLM image (~24 GiB) and NVIDIA's LiLiCorr drafter (~1.3 GiB)
ENGINE=vllm ./start.sh      # 12-16 min until ready
```

```sh
curl http://127.0.0.1:8888/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "nemotron-3.5-super-vl-abliterated", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The model thinks before it answers by default; add `"chat_template_kwargs": {"enable_thinking": false}` to a request
to skip that.

## Which engine

| | TensorFold (default) | vLLM |
|---|---|---|
| Ready after | 3-5 min (up to 10 the first time) | 12-16 min, every start |
| Requests at once | 2 (`PARALLEL`) | 1 |
| Window | 1,048,576 tokens | 786,432 tokens (1,048,576 after the first start, see below) |
| Drafts | the checkpoint's MTP head | NVIDIA's LiLiCorr drafter |
| Input | text, PNG images | text, JPEG and PNG images, video |
| Lowest free memory seen | 20 GB | 12 GB |

TensorFold's release reads JPEG only when libjpeg-turbo 3.1.4.1 is installed, which it does not ship; PNG works as is.

## Settings

Everything is in [`config.sh`](config.sh); override a value with an environment variable or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE` | `tensorfold` | `tensorfold` or `vllm` |
| `MODELS_DIR` / `MODEL_DIR` | `~/models` / `$MODELS_DIR/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated` | where the checkpoint goes |
| `BIND`, `PORT` | `127.0.0.1`, `8888` | where the API listens; `BIND=0.0.0.0` for your network |
| `API_KEY` | empty | require `Authorization: Bearer <key>` |
| `SERVED_MODEL_NAME` | `nemotron-3.5-super-vl-abliterated` | the model name clients send |
| `CONTEXT` | `1048576` | TensorFold: the window (prompt plus reply) |
| `PARALLEL` | `2` | TensorFold: requests served at once |
| `VLLM_CONTEXT` | `786432` | vLLM: the window; `1048576` works once vLLM has started once |
| `MIN_FREE_GIB` | `100` | the free memory `start.sh` asks for (`MEM_CHECK=0` skips the check) |
| `DRAFTER_DIR` | `$MODELS_DIR/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr` | vLLM: where the drafter goes |

## How it works

- **Checkpoint.** NVIDIA's NVFP4 release with its refusal behaviour removed by a lightweight weight edit (the
  procedure is not disclosed). Every tensor keeps NVIDIA's format and layout, so an engine that runs NVIDIA's
  checkpoint runs this one; the vision tower, embeddings and the 1M-token window are unchanged. Pinned at revision
  `1e1c8ecf`.
- **TensorFold.** `prepare.sh` downloads TensorFold's official 1.1.2 release for aarch64, checks it against its
  SHA-256 and unpacks it next to the scripts. `start.sh` runs its `tensorfold-native` inside NVIDIA's CUDA 13.3
  runtime image, which supplies the cuBLASLt the vision tower needs. Nemotron's CUDA kernels and MTP drafting are
  built into the release; nothing is compiled.
- **vLLM.** The image NVIDIA's model card names, pinned by digest, with the card's "Quick Start" flags for one DGX
  Spark: the LiLiCorr drafter proposing 7 tokens, one sequence, the card's Mamba cache settings, 75 % of the memory.
  A Docker volume keeps vLLM's caches between starts. The window is 786,432 tokens: on the very first start vLLM
  compiles its kernels and its CUDA graphs take ~2 GiB more, which leaves room for ~931K tokens at 75 % memory, so a
  full 1,048,576 window refuses to start there. From the second start on the compiled kernels are cached and
  `VLLM_CONTEXT=1048576` fits.

## Good to know

- **Uncensored.** It answers requests the original refuses, and it can produce harmful, offensive or false content.
  You are responsible for how you use it and for following the law and the model's license.
- The refusal counts come from a keyword rule, not a judge model. Treat the model as largely, not perfectly,
  uncensored.
- Vision is structurally unchanged but was not benchmarked separately for this build; long-context retrieval was not
  re-run either.
- The numbers come from our own harness: compare them with each other, not with scores published elsewhere.
- Tested on a DGX Spark only.

## Credits

See [CREDITS.md](CREDITS.md). The scripts are licensed under the Apache License 2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)); the model is governed by NVIDIA's [OpenMDW License, version 1.1](https://openmdw.ai/license/1-1/).
Not affiliated with or endorsed by NVIDIA.
