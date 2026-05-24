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

client = InferenceClient(api_key = HF_API_KEY)

print(f"Primary Model: {MODELS[0]}")
print(f"Type 'quit' to exit")

while True:
    prompt = input("Enter your prompt: ").strip()
    if prompt.lower() in ['quit', 'exit', 'q']:
        break
    if not prompt:
        continue

    print('Generating...')
    image = None

    for model in MODELS:
        try:
            image = client.text_to_image(model=model, inputs=prompt)
            break
        except Exception:
            print(f"Executing next...")
            continue

    if image:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"generated_{timestamp}.png"
        image.save(filename)
        print(f"Image saved as {filename}")
        image.show()
        print()
    else:
        print('Error: All models failed. Check your API key')
    
print("Goodbye!")