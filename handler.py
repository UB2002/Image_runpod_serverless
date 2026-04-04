import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
from diffusers import FluxKontextPipeline

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

pipe = None
load_error = None

try:
    print("Loading FLUX.1 Kontext pipeline (~24GB disk, ~12-16GB VRAM)...")

    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    print("✅ Pipeline loaded!")

except Exception as e:
    load_error = str(e)
    print(f"❌ Failed: {load_error}")


def _decode_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

def _encode_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def handler(event):
    if pipe is None:
        return {"error": f"Model failed to load: {load_error}"}

    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt")
        image_b64 = inp.get("image")

        if not prompt:
            return {"error": "Missing 'prompt'"}
        if not image_b64:
            return {"error": "Missing 'image' (base64)"}

        input_image = _decode_image(image_b64)
        seed = inp.get("seed")
        generator = torch.Generator("cuda").manual_seed(int(seed)) if seed else None

        with torch.inference_mode():
            output = pipe(
                image=input_image,
                prompt=prompt,
                guidance_scale=float(inp.get("guidance_scale", 2.5)),
                num_inference_steps=int(inp.get("num_inference_steps", 28)),
                generator=generator,
            )

        return {
            "image": _encode_image(output.images[0]),
            "text": "Image edited successfully.",
            "action": "replace",
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "OOM — try reducing num_inference_steps or image size"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
