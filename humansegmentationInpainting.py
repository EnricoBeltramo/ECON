from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import uuid
import os
import shutil


from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
from prompts import *
from diffusers.utils import load_image
import torch.nn as nn
import numpy as np
import uuid
import cv2
import os


img_url = "overture-creations-5sI6fQgYIuo5.png"
#img_url = "0a64d9c7ac4a86aa0c29195bc6f55246.jpg"
#img_url = "baf3c945fa6b4349c59953a97740e70f.jpg"
img_url = "test4.png"

basename = str(uuid.uuid4())

SEED = 6
SAVE_DIR = 'results/resultsSeg2'
DILATE_STEP = 2

GUIDANCE_SCALE=7.5
INFERENCE_STEPS=20  # steps between 15 and 30 work well for us
STRENGTH=1-1e-2  # make sure to use `strength` below 1.0

LOOPS = 1


shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR,exist_ok=True)



processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")


generator = torch.Generator(device="cuda").manual_seed(SEED)


imageToParse = load_image(img_url)

def cropAndMask(image):

  nomask = [0,1,2,3,11,12,13,14,15]
  mask = [4,5,6,7,8,9,10,16,17]

  # 1 Cappello 0,73 0,68
  # 2 Capelli 0,91 0,82
  # 3 Occhiali da sole 0,73 0,63
  # 4 Soprabiti 0,87 0,78
  # 5 Gonna 0,76 0,65
  # 6 Pantaloni 0,90 0,84
  # 7 Abito 0,74 0,55
  # 8 Cintura 0,35 0,30
  # 9 Scarpa sinistra 0,74 0,58
  # 10 Scarpa destra 0,75 0,60
  # 11 Faccia 0,92 0,85
  # 12 Gamba sinistra 0,90 0,82
  # 13 Gamba destra 0,90 0,81
  # 14 Braccio sinistro 0,86 0,74
  # 15 Braccio destro 0,82 0,73
  # 16 Borsa 0,91 0,84
  # 17 Sciarpa 0,63 0,29


  # Ottieni le dimensioni originali dell'immagine
  width, height = image.size

  # Determina la dimensione del lato del quadrato
  side_length = min(width, height)

  # Calcola le coordinate del punto in alto a sinistra per il taglio
  left = (width - side_length) // 2
  top = 0

  # Calcola le coordinate del punto in basso a destra per il taglio
  right = left + side_length
  bottom = top + side_length

  # Esegui il taglio dell'immagine per ottenere il quadrato
  square_image = image.crop((left, top, right, bottom))

  image_square_resize = square_image.resize((1024, 1024))

  inputs = processor(images=image_square_resize, return_tensors="pt")

  outputs = model(**inputs)
  logits = outputs.logits.cpu()

  upsampled_logits = nn.functional.interpolate(
      logits,
      size=image_square_resize.size[::-1],
      mode="bilinear",
      align_corners=False,
  )

  pred_seg = upsampled_logits.argmax(dim=1)[0]
  pred_seg = pred_seg.cpu().numpy().astype(np.uint8)

  pred_seg_background = pred_seg.copy()
  pred_seg[np.isin(pred_seg, nomask)] = 0
  pred_seg[np.isin(pred_seg, mask)] = 255

  N = 5

  # Crea un kernel per la dilatazione (puoi personalizzarne la dimensione)
  kernel = np.ones((N, N), np.uint8)

  # Esegui la dilatazione sulla maschera
  pred_seg = cv2.dilate(pred_seg, kernel, iterations=DILATE_STEP)

  image_cv_masked = np.array(image_square_resize).copy()
  image_cv_masked[pred_seg == 255] = [255, 255, 255]

  image_cv_nobackground = np.array(image_square_resize).copy()

  pred_seg_background = cv2.erode(pred_seg_background, kernel, iterations=DILATE_STEP)

  #image_cv_nobackground[pred_seg_background == 0] = [128,128,128]
  image_square_resize = Image.fromarray(image_cv_nobackground)

  image_masked = Image.fromarray(image_cv_masked)
  maskImage = Image.fromarray(pred_seg)


  return image_square_resize, image_masked, maskImage

def inpainting(image_square_resize,mask_image):


  image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image_square_resize,
    mask_image=mask_image,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=INFERENCE_STEPS,  # steps between 15 and 30 work well for us
    strength=STRENGTH,  # make sure to use `strength` below 1.0
    generator=generator,
  ).images[0]


  return image

def saveImages(basename,image_square_resize,image_masked,mask,image):
  fileName = basename + '_cropped.jpg'
  image_square_resize.save(os.path.join(SAVE_DIR, fileName)) 

  fileName = basename + '_parsed.jpg'
  image.save(os.path.join(SAVE_DIR, fileName)) 

  fileName = basename + '_masked.jpg'
  image_masked.save(os.path.join(SAVE_DIR, fileName)) 

  fileName = basename + '_mask.jpg'
  mask.save(os.path.join(SAVE_DIR, fileName)) 

  print(fileName)


for ind in range(LOOPS):
  image_square_resize, image_masked, mask = cropAndMask(imageToParse)
  image = inpainting(image_square_resize, mask)
  imageToParse = image

saveImages(basename,image_square_resize,image_masked,mask,image)


print('done')