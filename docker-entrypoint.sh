#!/usr/bin/env bash
# Launch vLLM with the flags Unlimited-OCR requires.
# Any extra `docker run` args are appended last, so they can override defaults.
set -Eeuo pipefail

MODEL="${MODEL:-baidu/Unlimited-OCR}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-baidu/Unlimited-OCR}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# num_attention_heads = 10 in this checkpoint, so vLLM only accepts a TP size
# that divides 10. Fail fast with a clear message instead of a pydantic
# ValidationError minutes into startup (baidu/Unlimited-OCR issue #76).
case "${TENSOR_PARALLEL_SIZE}" in
    1|2|5|10) ;;
    *)
        echo "FATAL: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} is invalid." >&2
        echo "       num_attention_heads=10 must be divisible by the TP size." >&2
        echo "       Allowed: 1, 2, 5, 10. Use replicas for more GPUs." >&2
        exit 78
        ;;
esac

args=(
    vllm serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    # Loads modeling_unlimitedocr.py from the checkpoint. Mandatory: the
    # architecture is not built into vLLM.
    --trust-remote-code
    # Without this the model loops forever on <|det|> coordinate tokens on long
    # documents. Clients still pass ngram_size / window_size per request.
    --logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor
    # Each page is a distinct image; a shared prefix practically never hits, so
    # the cache only burns VRAM. Recipe requires it off.
    --no-enable-prefix-caching
    --mm-processor-cache-gb 0
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
)

# --revision only resolves against the hub; a local directory has none.
case "${MODEL}" in
    /*|./*) ;;
    *) [ -n "${MODEL_REVISION:-}" ] && args+=(--revision "${MODEL_REVISION}") ;;
esac

if [ -n "${VLLM_API_KEY:-}" ]; then
    auth_note="api key required (VLLM_API_KEY is set)"
else
    auth_note="NO AUTH -- publish to 127.0.0.1 only, or set VLLM_API_KEY"
fi

cat <<EOF
Unlimited-OCR via vLLM
  model            ${MODEL}
  revision         ${MODEL_REVISION:-<unpinned>}
  served as        ${SERVED_MODEL_NAME}
  listening        ${HOST}:${PORT}
  tensor parallel  ${TENSOR_PARALLEL_SIZE}
  max model len    ${MAX_MODEL_LEN}
  auth             ${auth_note}
EOF

exec "${args[@]}" "$@"
