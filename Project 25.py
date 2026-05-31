from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image
from config import HF_API_KEY

MODELS = [
    "Bytedance/SDXL-Lightning",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5"
]

client = InferenceClient(api_key=HF_API_KEY)

print("=== Text-to-Image Payload Customization ===")
print(f"Primary Model: {MODELS[0]}")
print("Type 'quit' to exit.\n")

while True:

    prompt = input("Prompt: ").strip()

    if prompt.lower() in ["quit", "exit", "q"]:
        break

    if not prompt:
        continue

    negative_prompt = input(
        "Negative Prompt (optional): "
    ).strip()

    try:
        width = int(input("Width (default 1024): ") or 1024)
        height = int(input("Height (default 1024): ") or 1024)
        steps = int(input("Inference Steps (default 25): ") or 25)
        guidance = float(input("Guidance Scale (default 7.5): ") or 7.5)
        seed = int(input("Seed (default 42): ") or 42)

    except ValueError:
        print("Invalid input. Using default values.")
        width = 1024
        height = 1024
        steps = 25
        guidance = 7.5
        seed = 42

    print("\nGenerating image...")
    image = None

    for model in MODELS:

        try:
            image = client.text_to_image(
                prompt,
                model=model,

                # Payload customization
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed
            )

            print(f"Generated using: {model}")
            break

        except Exception as e:
            print(f"{model} failed.")
            continue

    if image:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"generated_{timestamp}.png"

        image.save(filename)

        print("\nGeneration Settings")
        print("-" * 30)
        print(f"Prompt: {prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"Size: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Guidance Scale: {guidance}")
        print(f"Seed: {seed}")
        print(f"Saved As: {filename}")

        image.show()

    else:
        print("All models failed.")

print("\nGoodbye!")