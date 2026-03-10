from diffusers import StableDiffusionXLPipeline
import torch

# ---------------------------
# 1️⃣ Prompt texte + contexte
# ---------------------------
prompt_text = "Un chat qui joue du piano dans une forêt enchantée"
context_text = "style cartoon, couleurs vives, ambiance joyeuse"
full_prompt = f"{prompt_text}, {context_text}"

output_file = "sdxl_cpu_result.png"

# ---------------------------
# 2️⃣ Charger le pipeline SDXL
# ---------------------------
# Attention : SDXL nécessite un pipeline dédié
device = "cpu"  # for CPU usage
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,  # utiliser float32 sur CPU
)
pipe = pipe.to(device)

# ---------------------------
# 3️⃣ Générer l'image
# ---------------------------
generator = torch.Generator(device=device).manual_seed(42)

image = pipe(
    prompt=full_prompt,
    negative_prompt="",  # optionnel
    num_inference_steps=25,  # plus élevé = meilleure qualité
    guidance_scale=7.5,
    generator=generator,
).images[0]

# ---------------------------
# 4️⃣ Sauvegarder l'image
# ---------------------------
image.save(output_file)
print(f"Image générée : {output_file}")