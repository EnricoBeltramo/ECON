from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import uuid
import os

SAVE_DIR = 'resultsStableXL'

os.makedirs(SAVE_DIR,exist_ok=True)

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

#img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
#mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

img_url = "overture-creations-5sI6fQgYIuo5.png"
mask_url = "overture-creations-5sI6fQgYIuo_mask5.png"


image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a nudist young  girl, very little tits, very little nipples"
generator = torch.Generator(device="cuda").manual_seed(1)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=9.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.9999999,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]


fileName = str(uuid.uuid4()) + '.jpg'

print(fileName)

image.save(os.path.join(SAVE_DIR, fileName)) 

print('done')