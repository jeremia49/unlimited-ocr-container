# Unlimited-OCR served through vLLM.
#
# The `unlimited-ocr` architecture is not in a stable vLLM pip wheel yet, so the
# dedicated release image is the only viable base. Do not swap this for
# vllm/vllm-openai:latest -- it cannot load model_type "unlimited-ocr".
#
#   Default (CUDA 13.0):  vllm/vllm-openai:unlimited-ocr
#   Hopper (CUDA 12.9):   vllm/vllm-openai:unlimited-ocr-cu129
#
# Pin by digest in production; both tags are mutable upstream.
ARG BASE_IMAGE=vllm/vllm-openai:unlimited-ocr

FROM ${BASE_IMAGE}

# baidu/Unlimited-OCR @ 07dea832e22aefee32ad281d4b80551282e1c168 (2026-07-29).
# Pinned so a silent upstream reupload cannot change what this image serves.
ARG MODEL_REPO=baidu/Unlimited-OCR
ARG MODEL_REVISION=07dea832e22aefee32ad281d4b80551282e1c168

# false -> weights come from a mounted HF cache at runtime (image stays ~small).
# true  -> 6.67 GB of weights are baked in (self-contained, image ~+7 GB).
ARG BAKE_MODEL=false

ARG GPU_MEM_UTIL=0.85
ARG MAX_MODEL_LEN=32768
ARG MAX_BATCHED_TOK=16384

ENV MODEL=${MODEL_REPO} \
    MODEL_REVISION=${MODEL_REVISION} \
    SERVED_MODEL_NAME=baidu/Unlimited-OCR \
    HOST=0.0.0.0 \
    PORT=8000 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL} \
    MAX_MODEL_LEN=${MAX_MODEL_LEN} \
    MAX_BATCHED_TOK=${MAX_BATCHED_TOK} \
    HF_HOME=/models \
    HF_HUB_CACHE=/models/hub

# No VOLUME for /models: declaring it here would make the layer that downloads
# the weights a no-op, and it would also leak an anonymous volume per container.
# Bind-mount or named-volume it at run time instead.
#
# Optional bake. HF_TOKEN arrives as a build secret, never as a build arg, so it
# does not land in image history. Excludes the 82 MB demo GIF, the PDF, and the
# SGLang wheel: none are needed to serve.
RUN --mount=type=secret,id=hf_token,required=false \
    set -eu; \
    if [ "${BAKE_MODEL}" = "true" ]; then \
        if [ -f /run/secrets/hf_token ]; then \
            HF_TOKEN="$(cat /run/secrets/hf_token)"; export HF_TOKEN; \
        fi; \
        mkdir -p "${HF_HUB_CACHE}"; \
        if command -v hf >/dev/null 2>&1; then dl="hf download"; else dl="huggingface-cli download"; fi; \
        $dl "${MODEL_REPO}" --revision "${MODEL_REVISION}" \
            --exclude "assets/*" "*.pdf" "wheel/*"; \
        unset HF_TOKEN; \
    else \
        mkdir -p "${HF_HUB_CACHE}"; \
        echo "BAKE_MODEL=false: mount a Hugging Face cache at /models at run time."; \
    fi

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

# /health sits outside vLLM's GUARDED_PREFIX ("/v1", "/v2", "/inference"), so it
# answers even when VLLM_API_KEY is set. curl is at /usr/bin/curl in this image;
# there is no `python`, only `python3`, so do not probe with `python`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" >/dev/null || exit 1

# Replaces the base image ENTRYPOINT: the recipe flags are mandatory and must
# not be forgettable at `docker run` time.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD []
