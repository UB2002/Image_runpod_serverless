import runpod
import torch
import base64
from io import BytesIO
from PIL import Image
from diffusers import FluxImg2ImgPipeline

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

print("Loading FLUX.2-klein-4B img2img pipeline...")
pipe = FluxImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")
print("Pipeline ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_image(b64_string: str) -> Image.Image:
    """Decode a raw base-64 string (no data-URL prefix) into a PIL Image."""
    raw = base64.b64decode(b64_string)
    return Image.open(BytesIO(raw)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    """Encode a PIL Image to a raw base-64 PNG string (no data-URL prefix)."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Inference handler ─────────────────────────────────────────────────────────
def handler(event):
    """
    RunPod serverless handler — img2img only.

    Input  (event["input"]):
      prompt          str   required  – edit instruction from the FastAPI backend
      image           str   required  – raw base-64 canvas snapshot (no data-URL prefix)
      strength        float optional  – denoising strength, 0.0–1.0  (default 0.75)
      width           int   optional  – resize canvas to this width   (default 1024)
      height          int   optional  – resize canvas to this height  (default 1024)
      num_steps       int   optional  – inference steps               (default 28)
      guidance_scale  float optional  – CFG scale                    (default 3.5)
      seed            int   optional  – RNG seed for reproducibility

    Output (matches FastAPI ChatResponse schema):
      text            str   – human-readable status message
      image           str   – raw base-64 PNG (FastAPI adds the data-URL prefix)
      action          str   – always "replace"
    """
    print("Worker Start")
    input = event["input"]

    # ── Validate required fields ──────────────────────────────────────────────
    prompt = input.get("prompt")
    if not prompt:
        return {"error": "Missing required field: 'prompt'"}

    input_image_b64 = input.get("image")
    if not input_image_b64:
        return {"error": "Missing required field: 'image' — this endpoint only supports img2img"}

    # ── Optional params ───────────────────────────────────────────────────────
    strength       = float(input.get("strength", 0.75))
    width          = int(input.get("width", 1024))
    height         = int(input.get("height", 1024))
    num_steps      = int(input.get("num_steps", 28))
    guidance_scale = float(input.get("guidance_scale", 3.5))
    seed           = input.get("seed", None)

    print(f"Received prompt: {prompt}")
    print(f"Strength: {strength} | Steps: {num_steps} | CFG: {guidance_scale}")

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    # ── Run img2img ───────────────────────────────────────────────────────────
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
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # FastAPI backend expects raw base-64 (it adds "data:image/png;base64," itself)
    result_b64 = _encode_image(output.images[0])

    return {
        "text":   f"Canvas updated based on: \"{prompt}\"",
        "image":  result_b64,
        "action": "replace",
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

    ## pip install runpod torch diffusers transformers accelerate pillows