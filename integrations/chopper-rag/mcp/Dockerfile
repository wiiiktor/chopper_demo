# Chopper MCP server (stdio) for Glama's automated checks. CPU only; the small model is baked in so the checks
# need no network. Run: docker build -t chopper-mcp . && docker run -i --rm chopper-mcp
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 CHOPPER_VARIANT=small CHOPPER_DEVICE=cpu HF_HUB_DISABLE_TELEMETRY=1
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install "chopper-rag>=0.1.5" \
 && python -c "from chopper_rag.core import Chopper; Chopper.from_pretrained(variant='small', device='cpu')"
ENV HF_HUB_OFFLINE=1
CMD ["chopper-mcp"]
