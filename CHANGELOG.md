# Changelog

## 0.1.0 (unreleased)

- The abliterated Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (`armin1/…-NVFP4-abliterated` @ `1e1c8ecf`) on one DGX Spark,
  with either engine reading the same checkpoint as published:
  - TensorFold 1.1.2, the official aarch64 release (SHA-256 pinned), in `nvidia/cuda:13.3.1-runtime-ubuntu24.04`:
    1,048,576-token window, 2 requests at once, MTP drafts, `--vision`;
  - vLLM, the image NVIDIA's card names (pinned by digest), with NVIDIA's LiLiCorr drafter @ `292bf1ef` and the card's
    one-Spark flags, window 786,432 (`VLLM_CONTEXT`): the first start with an empty compile cache has room for ~931K
    tokens only, so the full 1,048,576 refused to start there.
- Checks, head-to-head against NVIDIA's original on the same engine: TensorFold 1.1.2 task suite 49/15/8/20/10 both,
  code 25/25 both, GSM8K 292 vs 293 of 300, decode and MTP acceptance equal; vLLM + LiLiCorr prompt processing equal,
  LiLiCorr acceptance 3-6 % lower on code and explanations (decode up to 7 % slower on the nine-prompt probe);
  KL 0.0157 against a 0.0138 noise floor (vLLM); refusals 4/204 vs 200/204 (TensorFold) and 4/204 vs 201/204 (vLLM).
  Numbers in the README.
- Fresh-clone test on a DGX Spark (`prepare.sh --no-download` on the checkpoint files as uploaded; the TensorFold
  archive and both images fetched and checked): TensorFold ready in 311 s, vLLM in 880 s from an empty compile cache;
  `tools/check.sh` text, tool call and PNG image pass on both; the drafter probe matched the bench on both.
