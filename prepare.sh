#!/bin/bash
# prepare.sh: everything start.sh needs, once. Checks the machine, downloads the checkpoint as published (~70 GiB,
# resumable; nothing is converted), then gets the engine ENGINE names: TensorFold's 1.1.2 release archive (checked by
# its SHA-256) and NVIDIA's CUDA runtime image, or for vLLM the pinned image (~24 GiB) and NVIDIA's LiLiCorr drafter
# (~1.3 GiB). Safe to run again: finished steps are skipped.
#   ./prepare.sh [--no-download]            ENGINE=vllm ./prepare.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
. "$D/config.sh"
download=1
for a in "$@"; do
  case "$a" in
    --no-download) download=0 ;;
    *) echo "usage: [ENGINE=tensorfold|vllm] ./prepare.sh [--no-download]"; exit 2 ;;
  esac
done
say() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*"; exit 1; }
case "$ENGINE" in tensorfold | vllm) ;; *) die "ENGINE must be tensorfold or vllm (it is: $ENGINE)" ;; esac

say "checking the machine"
command -v docker > /dev/null || die "docker is not installed"
[ "$(uname -m)" = aarch64 ] || die "this is pinned to the DGX Spark's aarch64 builds (this machine: $(uname -m))"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1 || true)
[ -n "$gpu" ] || die "nvidia-smi found no GPU"
case "$gpu" in *GB10*) ;; *) echo "  note: built and tested on a DGX Spark (GB10); this GPU is: $gpu" ;; esac
total=$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
[ "$total" -ge 115 ] || die "this needs a 128 GB machine (MemTotal ${total} GiB)"

hf_get() {  # hf_get REPO REVISION DIR: download a Hugging Face repo at a revision into DIR (resumable)
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_DISABLE_TELEMETRY=1 \
    ${HF_TOKEN:+-e HF_TOKEN} -v "$3":/out python:3.12-slim sh -c \
    "pip install -q --no-cache-dir --disable-pip-version-check --no-warn-script-location huggingface_hub==$HF_HUB_VERSION &&
     PATH=/tmp/.local/bin:\$PATH hf download '$1' --revision '$2' --local-dir /out"
}
if [ $download = 1 ]; then
  mkdir -p "$MODEL_DIR"
  need=$((75 - $(du -s --block-size=1G "$MODEL_DIR" | cut -f1)))
  free=$(df --output=avail --block-size=1G "$MODEL_DIR" | tail -1 | tr -d ' ')
  [ "$need" -le 0 ] || [ "$free" -ge "$need" ] || die "$MODEL_DIR needs ${need} GB more free disk (has ${free} GB)"
  say "downloading $CHECKPOINT @ ${CHECKPOINT_REVISION:0:8} into $MODEL_DIR (~70 GiB, resumable)"
  hf_get "$CHECKPOINT" "$CHECKPOINT_REVISION" "$MODEL_DIR"
  if [ "$ENGINE" = vllm ]; then
    mkdir -p "$DRAFTER_DIR"
    say "downloading the drafter $DRAFTER @ ${DRAFTER_REVISION:0:8} into $DRAFTER_DIR (~1.3 GiB)"
    hf_get "$DRAFTER" "$DRAFTER_REVISION" "$DRAFTER_DIR"
  fi
fi
n=$(compgen -G "$MODEL_DIR/model-000[0-9][0-9]-of-00088.safetensors" | wc -l)
[ "$n" = 88 ] && [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ] ||
  die "$MODEL_DIR: expected 88 weight files, config.json and tokenizer.json (found $n weight files)"
say "checkpoint complete: 88 weight files"

if [ "$ENGINE" = tensorfold ]; then
  if [ -x "$TF_DIR/bin/tensorfold-native" ] && [ "$(cat "$TF_DIR/.sha256" 2> /dev/null)" = "$TF_SHA256" ]; then
    say "TensorFold $TF_VERSION is in $TF_DIR"
  else
    say "downloading TensorFold $TF_VERSION ($TF_URL)"
    tmp=$(mktemp "$D/.tensorfold.XXXXXX.tar.gz")
    trap 'rm -f "$tmp"' EXIT
    curl -fL --retry 3 -o "$tmp" "$TF_URL"
    echo "$TF_SHA256  $tmp" | sha256sum -c --quiet - || die "the TensorFold archive does not match TF_SHA256"
    rm -rf "$TF_DIR" && mkdir -p "$TF_DIR"
    tar -xzf "$tmp" -C "$TF_DIR" --strip-components=1
    echo "$TF_SHA256" > "$TF_DIR/.sha256"
    say "TensorFold $TF_VERSION unpacked into $TF_DIR (SHA-256 checked)"
  fi
  say "pulling $TF_IMAGE"
  docker pull -q "$TF_IMAGE" > /dev/null
  v=$(docker run --rm -v "$TF_DIR":/tf:ro -e LD_LIBRARY_PATH=/tf/lib --entrypoint /tf/bin/tensorfold-native \
    "$TF_IMAGE" --version 2>&1 | head -1) || die "TensorFold does not run in $TF_IMAGE: $v"
  say "engine ready: $v"
else
  [ -f "$DRAFTER_DIR/config.json" ] || die "$DRAFTER_DIR: no drafter (run without --no-download)"
  say "pulling $VLLM_IMAGE (~24 GiB the first time)"
  docker pull -q "$VLLM_IMAGE" > /dev/null
  say "engine ready: vLLM ($VLLM_IMAGE)"
fi
pre=
[ "$ENGINE" = vllm ] && pre='ENGINE=vllm '
say "done: ${pre}./start.sh serves it"
