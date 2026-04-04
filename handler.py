import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
import traceback

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
HF_TOKEN = os.environ.get("HF_TOKEN")

pipe = None
load_error = None

try:
    print("Loading QwenImageEditPlusPipeline... This model is heavy (~25-40GB peak).")

    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )

    # Critical: Offload BEFORE any .to("cuda")
    pipe.enable_model_cpu_offload()          # Best for 24GB GPUs
    # pipe.enable_sequential_cpu_offload()   # Uncomment if still OOM (slower)

    pipe.set_progress_bar_config(disable=True)

    print(f"✅ Pipeline loaded successfully! Current VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

except Exception as e:
    load_error = str(e)
    print(f"❌ FATAL: Failed to load model — {load_error}")
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
        return {"error": f"Model failed to load: {load_error}"}

    try:
        input_data = event.get("input", {})

        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: 'prompt'"}

        # Support "images" (list) or "image" (single) for flexibility
        images_b64 = input_data.get("images") or input_data.get("image")
        if not images_b64:
            return {"error": "Missing 'images' or 'image' field (base64 string or list)."}

        if isinstance(images_b64, str):
            images_b64 = [images_b64]

        if len(images_b64) > 2:
            return {"error": "This pipeline supports maximum 2 reference images."}

        input_images = [_decode_image(b64) for b64 in images_b64]

        # Parameters
        num_inference_steps = int(input_data.get("num_inference_steps", 30))
        true_cfg_scale = float(input_data.get("true_cfg_scale", 4.0))
        guidance_scale = float(input_data.get("guidance_scale", 1.0))
        negative_prompt = input_data.get("negative_prompt", "")
        seed = input_data.get("seed")

        generator = torch.Generator("cuda").manual_seed(int(seed)) if seed is not None else None

        print(f"Prompt: {prompt[:150]}...")
        print(f"Reference images: {len(input_images)} | Steps: {num_inference_steps}")

        torch.cuda.empty_cache()

        inputs = {
            "image": input_images,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = pipe(**inputs)
            output_image = output.images[0]

        result_b64 = _encode_image(output_image)

        print(f"✅ Inference completed. VRAM after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return {
            "text": "Image edited successfully based on prompt.",
            "image": result_b64,
            "action": "replace",
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU Out of Memory. Try reducing num_inference_steps (e.g. 20) or use a 24GB+ GPU."}

    except Exception as e:
        error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
