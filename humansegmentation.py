
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
from diffusers.utils import load_image
import torch.nn as nn
import numpy as np
import uuid
import cv2
import os

SAVE_DIR = 'resultsSeg'
os.makedirs(SAVE_DIR,exist_ok=True)

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"

# image = Image.open(requests.get(url, stream=True).raw)

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

img_url = "overture-creations-5sI6fQgYIuo5.png"

image = load_image(img_url).resize((1024, 1024))

inputs = processor(images=image, return_tensors="pt")


outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]
pred_seg = pred_seg.cpu().numpy().astype(np.uint8)


pred_seg[np.isin(pred_seg, nomask)] = 0
pred_seg[np.isin(pred_seg, mask)] = 255

N = 5

# Crea un kernel per la dilatazione (puoi personalizzarne la dimensione)
kernel = np.ones((N, N), np.uint8)

# Esegui la dilatazione sulla maschera
pred_seg = cv2.dilate(pred_seg, kernel, iterations=3)

image_cv_masked = np.array(image)
image_cv_masked[pred_seg == 255] = [255, 255, 255]


image_masked = Image.fromarray(image_cv_masked)
image = Image.fromarray(pred_seg)

basename = str(uuid.uuid4())

fileName = basename + '.jpg'
fileNameMasked = basename  + 'masked.jpg'

print(fileName)

image.save(os.path.join(SAVE_DIR, fileName)) 
image_masked.save(os.path.join(SAVE_DIR, fileNameMasked))

print('done')