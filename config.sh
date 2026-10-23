# config.sh: every pin and default in one place. prepare.sh, start.sh and stop.sh read it; override any value
# with an environment variable of the same name (or put it in a .env file next to this one).
[ -f "$(dirname "${BASH_SOURCE[0]}")/.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/.env"

# the engine: tensorfold (default) or vllm. Both serve the same checkpoint files.
ENGINE=${ENGINE:-tensorfold}

# the checkpoint, exactly as published
CHECKPOINT=${CHECKPOINT:-armin1/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-1e1c8ecf1976ee0ec387c114c1964a6f3506e804}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated}
HF_HUB_VERSION=${HF_HUB_VERSION:-2.2.0}

# TensorFold: the official 1.1.2 release (native CUDA engine, Nemotron kernels built in), checked by its SHA-256,
# run inside NVIDIA's CUDA 13.3 runtime image (the vision tower needs its cuBLASLt)
TF_VERSION=${TF_VERSION:-1.1.2}
TF_URL=${TF_URL:-https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION/tensorfold-$TF_VERSION-linux-aarch64.tar.gz}
TF_SHA256=${TF_SHA256:-4da070ed641e9dc6d41fa91d690a2fdf42e5023e1f84dfc69bcad8e893e6c367}
TF_DIR=${TF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tensorfold-$TF_VERSION}
TF_IMAGE=${TF_IMAGE:-nvidia/cuda:13.3.1-runtime-ubuntu24.04}
CONTEXT=${CONTEXT:-1048576}
PARALLEL=${PARALLEL:-2}

# vLLM: the image NVIDIA's model card names (tag nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2), pinned by digest,
# and NVIDIA's LiLiCorr drafter
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd}
DRAFTER=${DRAFTER:-nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr}
DRAFTER_REVISION=${DRAFTER_REVISION:-292bf1efc06ba39567da7351f2ad7a522afa3959}
DRAFTER_DIR=${DRAFTER_DIR:-$MODELS_DIR/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr}
# docker volume that keeps vLLM's compile cache between starts
VLLM_CACHE=${VLLM_CACHE:-nemotron-supervl-vllm-cache}
# vLLM's window. The first start (nothing compiled yet) spends ~2 GiB more on CUDA graphs and leaves room for ~931K
# tokens at 75 % memory, so the full 1,048,576 fails there; 786,432 always fits. Once the cache exists, 1048576 works.
VLLM_CONTEXT=${VLLM_CONTEXT:-786432}

# serving
NAME=${NAME:-nemotron-supervl-abliterated}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-nemotron-3.5-super-vl-abliterated}
BIND=${BIND:-127.0.0.1}
PORT=${PORT:-8888}
# MemAvailable the start needs (GiB): ~93 GB in use at the lowest point on TensorFold, ~104 on vLLM
MIN_FREE_GIB=${MIN_FREE_GIB:-100}
