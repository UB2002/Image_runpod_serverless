import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
import traceback

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — FLUX.2-klein-4B is gated and may fail to download.")

pipe = None
load_error = None

try:
    print("Loading FLUX.2-klein-4B img2img pipeline...")

    from diffusers import FluxImg2ImgPipeline   # Better to use specific pipeline

    pipe = FluxImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    ).to("cuda")

    # Optional: Use this if you still get OOM on smaller GPUs (slower but safer)
    # pipe.enable_model_cpu_offload()

    print(f"Pipeline loaded successfully. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

except Exception as e:
    load_error = str(e)
    print(f"FATAL: Model failed to load — {load_error}")
    print(traceback.format_exc())

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
    print("=== Handler started ===")
    
    if pipe is None:
        return {"error": f"Model failed to load at startup: {load_error}"}

    try:
        input_data = event.get("input", {})
        
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: 'prompt'"}

        image_b64 = input_data.get("image")
        if not image_b64:
            return {"error": "Missing required field: 'image' — this endpoint only supports img2img"}

        # Safer defaults for FLUX.2-klein (prevents OOM)
        width = int(input_data.get("width", 768))      # Lower default
        height = int(input_data.get("height", 768))
        num_steps = int(input_data.get("num_steps", 8))  # Klein works great with low steps
        guidance_scale = float(input_data.get("guidance_scale", 3.0))
        seed = input_data.get("seed")

        print(f"Prompt: {prompt[:150]}...")
        print(f"Image size: {width}x{height}, Steps: {num_steps}, Guidance: {guidance_scale}")

        # Clear cache before inference
        torch.cuda.empty_cache()

        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(int(seed))
        else:
            generator = None

        # Decode and resize input image
        print("Decoding and resizing input image...")
        init_image = _decode_image(image_b64)
        init_image = init_image.resize((width, height), Image.LANCZOS)

        print("Running FLUX.2-klein img2img inference...")
        output = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            # height=height,      # optional - pipeline usually takes it from image
            # width=width,
        )

        print("Encoding output image...")
        result_b64 = _encode_image(output.images[0])

        print(f"Inference completed successfully. VRAM after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return {
            "text": f"Canvas updated based on: \"{prompt}\"",
            "image": result_b64,
            "action": "replace",
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("*** CUDA OUT OF MEMORY detected ***")
        return {"error": "GPU out of memory. Try smaller image size (e.g. 768x768 or 512x512) and fewer steps (4-12)."}

    except Exception as e:
        error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
