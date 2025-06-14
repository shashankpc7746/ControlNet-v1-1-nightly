import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import os

# === CONFIGURATION ===
controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"  # ✅ More stable and recommended
sd_model_path = "runwayml/stable-diffusion-v1-5"
pose_image_path = "poses/pose_input.png"  # ✅ Use forward slashes for cross-platform safety
output_path = "results/controlnet_output.png"  # ✅ Store in a proper folder
prompt = "Thomas Shelby form Peaky Blinders"  # ✅ Use a descriptive prompt

# === Ensure Output Directory Exists ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Load Pose Image ===
def load_pose_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))  # Required for most SD pipelines
    return image

# === Load ControlNet Model ===
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()

# === Inference ===
pose_image = load_pose_image(pose_image_path)
result = pipe(prompt, image=pose_image, num_inference_steps=30).images[0]
result.save(output_path)

print(f"✅ Image saved at: {output_path}")
