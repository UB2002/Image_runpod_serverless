import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
import traceback

MODEL_ID = "stepfun-ai/Step1X-Edit-v1p2-preview" 

pipe = None
load_error = None

try:
    print("Loading Step1X-Edit pipeline (~10GB disk, ~16GB VRAM)...")

    # Requires a custom diffusers branch
    import subprocess
    subprocess.run([
        "pip", "install", "-q",
        "git+https://github.com/Peyton-Chen/diffusers.git@dev/MergeV1-2"
    ], check=True)

    from diffusers import Step1XEditPipelineV1P2

    pipe = Step1XEditPipelineV1P2.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    print("✅ Step1X-Edit loaded!")

except Exception as e:
    load_error = str(e)
    print(f"❌ Failed: {load_error}\n{traceback.format_exc()}")


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

        # Reasoning mode — the model THINKS about your prompt before editing
        enable_thinking = inp.get("enable_thinking", True)
        enable_reflection = inp.get("enable_reflection", True)

        with torch.inference_mode():
            result = pipe(
                image=input_image,
                prompt=prompt,
                num_inference_steps=int(inp.get("num_inference_steps", 28)),
                true_cfg_scale=float(inp.get("true_cfg_scale", 4.0)),
                guidance_scale=float(inp.get("guidance_scale", 6.0)),
                generator=generator,
                enable_thinking_mode=enable_thinking,
                enable_reflection_mode=enable_reflection,
            )

        response = {
            "image": _encode_image(result.images[0]),
            "text": "Image edited successfully.",
            "action": "replace",
        }

        # Optionally return the model's reasoning
        if enable_thinking and hasattr(result, "reformat_prompt"):
            response["reformat_prompt"] = result.reformat_prompt

        return response

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "OOM — try reducing num_inference_steps or image resolution"}
    except Exception as e:
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
