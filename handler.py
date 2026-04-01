import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
from diffusers import AutoPipelineForImage2Image

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# HF_TOKEN must be set as an env variable in RunPod endpoint settings
# Settings → Environment Variables → HF_TOKEN = hf_xxxxxxxxxxxx
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — FLUX.2-klein-4B is a gated model and will fail to download.")

pipe = None
load_error = None

try:
    print("Loading FLUX.2-klein-4B img2img pipeline...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    ).to("cuda")
    print(f"Pipeline ready. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    load_error = str(e)
    print(f"FATAL: Model failed to load — {load_error}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_image(b64_string: str) -> Image.Image:
    raw = base64.b64decode(b64_string)
    return Image.open(BytesIO(raw)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Inference handler ─────────────────────────────────────────────────────────
def handler(event):
    print("Worker Start")

    # Return load error immediately so RunPod marks the job failed with a clear message
    if pipe is None:
        return {"error": f"Model failed to load at startup: {load_error}"}

    input = event["input"]

    prompt = input.get("prompt")
    if not prompt:
        return {"error": "Missing required field: 'prompt'"}

    input_image_b64 = input.get("image")
    if not input_image_b64:
        return {"error": "Missing required field: 'image' — this endpoint only supports img2img"}

    strength       = float(input.get("strength", 0.75))
    width          = int(input.get("width", 1024))
    height         = int(input.get("height", 1024))
    num_steps      = int(input.get("num_steps", 28))
    guidance_scale = float(input.get("guidance_scale", 3.5))
    seed           = input.get("seed", None)

    print(f"Prompt: {prompt}")
    print(f"Strength: {strength} | Steps: {num_steps} | CFG: {guidance_scale} | Size: {width}x{height}")

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    try:
        init_image = _decode_image(input_image_b64)
        init_image = init_image.resize((width, height))

        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory — try a smaller image size or lower num_steps"}
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    result_b64 = _encode_image(output.images[0])
    print(f"Done. VRAM after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return {
        "text":   f"Canvas updated based on: \"{prompt}\"",
        "image":  result_b64,
        "action": "replace",
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})