# Matches torch==2.11.0+cu128 and torchvision==0.26.0+cu128 exactly
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

WORKDIR /

# Install dependencies (torch/torchvision already in base image — excluded from requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /

# Pre-bake model weights (requires HF_TOKEN for this gated model)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python3 -c "\
from diffusers import FluxImg2ImgPipeline; \
import torch; \
FluxImg2ImgPipeline.from_pretrained( \
    'black-forest-labs/FLUX.2-klein-4B', \
    torch_dtype=torch.bfloat16 \
)"

# Start the worker
CMD ["python3", "-u", "handler.py"]