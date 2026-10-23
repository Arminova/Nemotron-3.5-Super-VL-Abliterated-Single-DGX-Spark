#!/bin/bash
# start.sh: serves the model on one DGX Spark, an OpenAI-compatible API on BIND:PORT (127.0.0.1:8888 by default).
#   ENGINE=tensorfold (default): TensorFold 1.1.2's native engine, a 1,048,576-token window, PARALLEL requests at once,
#     the checkpoint's built-in MTP drafts, images. Ready in 3-5 min (up to 10 the first time the files are read).
#   ENGINE=vllm: vLLM with NVIDIA's LiLiCorr drafter, a VLLM_CONTEXT window (786,432), one request at a time, images
#     and video. Ready in 12-16 min (vLLM loads slowly and compiles; the compile is cached after the first start).
# Runs detached as container NAME and waits until the model answers. Settings: config.sh (or environment / .env).
#   ./start.sh            ENGINE=vllm ./start.sh            ./stop.sh            docker logs -f nemotron-supervl-abliterated
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
. "$D/config.sh"
say() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*"; exit 1; }
case "$ENGINE" in tensorfold | vllm) ;; *) die "ENGINE must be tensorfold or vllm (it is: $ENGINE)" ;; esac

[ -f "$MODEL_DIR/config.json" ] || die "no checkpoint in $MODEL_DIR: run ./prepare.sh first"
if [ "$ENGINE" = tensorfold ]; then
  [ -x "$TF_DIR/bin/tensorfold-native" ] || die "no TensorFold in $TF_DIR: run ./prepare.sh first"
else
  [ -f "$DRAFTER_DIR/config.json" ] || die "no drafter in $DRAFTER_DIR: run ENGINE=vllm ./prepare.sh first"
  docker image inspect "$VLLM_IMAGE" > /dev/null 2>&1 || die "no vLLM image: run ENGINE=vllm ./prepare.sh first"
fi
docker rm -f "$NAME" > /dev/null 2>&1 || true
avail=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "${MEM_CHECK:-1}" = 1 ] && [ "$avail" -lt "$MIN_FREE_GIB" ]; then
  die "only ${avail} GiB of memory is free; this model needs ${MIN_FREE_GIB}. Stop other models (or MEM_CHECK=0)"
fi
host=$BIND
[ "$host" = 0.0.0.0 ] && host=127.0.0.1
[ "$BIND" = 0.0.0.0 ] && [ -z "${API_KEY:-}" ] && echo "  note: BIND=0.0.0.0 serves the whole network with no API key"

if [ "$ENGINE" = tensorfold ]; then
  say "starting $SERVED_MODEL_NAME on TensorFold $TF_VERSION, $BIND:$PORT (context $CONTEXT, $PARALLEL at once)"
  docker run -d --name "$NAME" --gpus all --ipc=host -p "$BIND:$PORT:$PORT" \
    -v "$TF_DIR":/tf:ro -v "$MODEL_DIR":/model:ro -e LD_LIBRARY_PATH=/tf/lib \
    --entrypoint /tf/bin/tensorfold-native "$TF_IMAGE" \
    serve /model --host 0.0.0.0 --port "$PORT" --name "$SERVED_MODEL_NAME" --context "$CONTEXT" \
    --parallel "$PARALLEL" --vision ${API_KEY:+--api-key "$API_KEY"} --no-update-check > /dev/null
  limit=1500
else
  # NVIDIA's "Quick Start" command for one DGX Spark (LiLiCorr, 7 drafted tokens, one sequence), window VLLM_CONTEXT
  say "starting $SERVED_MODEL_NAME on vLLM + LiLiCorr, $BIND:$PORT (context $VLLM_CONTEXT, one at a time; ~15 min)"
  docker run -d --name "$NAME" --gpus all --ipc=host -p "$BIND:$PORT:$PORT" \
    -v "$MODEL_DIR":/model:ro -v "$DRAFTER_DIR":/draft:ro -v "$VLLM_CACHE":/root/.cache \
    --entrypoint vllm "$VLLM_IMAGE" serve /model \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED_MODEL_NAME" ${API_KEY:+--api-key "$API_KEY"} \
    --max-num-seqs 1 \
    --max-model-len "$VLLM_CONTEXT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --max-num-batched-tokens 8192 \
    --speculative_config.model /draft \
    --speculative_config.num_speculative_tokens 7 \
    --speculative_config.method dflash \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --enable-mamba-shared-prefix-checkpoint \
    --video-pruning-rate 0.5 \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
    --block-size 64 \
    --prefix-match-unit 64 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 > /dev/null
  limit=2700
fi
t=0
until curl -sf "http://$host:$PORT/v1/models" ${API_KEY:+-H "Authorization: Bearer $API_KEY"} > /dev/null 2>&1; do
  sleep 10
  t=$((t + 10))
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { docker logs --tail 20 "$NAME" 2>&1; die "the server exited"; }
  [ $t -ge $limit ] && die "not ready after ${t}s (still running: docker logs $NAME)"
done
say "ready after ${t}s: http://$BIND:$PORT/v1 (model $SERVED_MODEL_NAME, $ENGINE). Stop with ./stop.sh"
